//! Automatic GPU tile-cached layers for static subtrees and contiguous
//! static sibling runs (`gpui::Window::paint_cached_layer`).
//!
//! When an NSH view or contiguous run of sibling views remains unchanged for
//! [`STABLE_FRAMES`] consecutive paint passes, runs under a pure 2D translation
//! without rounded/polygon clips or color filters, and contains no uncacheable
//! primitives (hosted native items, active shaders, animated/loading images,
//! unsettled canvases, active scrollbars/spinners, clip paths, gradients, or
//! drop shadows), it is painted into a GPU tile-cached layer instead of
//! re-walking and re-submitting primitives on every animation tick elsewhere in
//! the surface.

use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    hash::Hasher,
};

use gpui::{Bounds, CachedLayerId, CachedLayerOptions, ContentMask, Pixels};
use nsh_core::{EventKind, Filter, ViewKind};
use rustc_hash::FxHasher;

use crate::{
    geom::Mat4,
    render::DrawCx,
    view::{NshView, ViewState},
};

/// Consecutive unchanged paint passes required before promoting a view or
/// sibling run to a GPU tile-cached layer.
pub(crate) const STABLE_FRAMES: u32 = 2;

/// Minimum view count in a cached subtree or sibling run (unless it contains a
/// rasterized canvas or multiple text nodes).
const MIN_LAYER_NODES: u32 = 6;

/// Minimum bounding box area (points²) before promoting a layer so tiny views do
/// not consume 512×512 GPU tiles.
const MIN_LAYER_AREA: f32 = 6_000.0;

/// Minimum width and height (points) for a cached layer.
const MIN_LAYER_SIDE: f32 = 24.0;

/// Maximum amount a full unclipped layer may extend beyond the viewport side
/// while keeping unclipped bounds so scroll translation reuses GPU tiles.
const MAX_UNCLIPPED_OVERSHOOT: f32 = 1024.0;

/// Maximum side length (points) for an unclipped cached layer.
const MAX_UNCLIPPED_SIDE: f32 = 2048.0;

/// Content-space quantization step (points) for banding oversized layers along
/// a scrolling axis so `layer_bounds.origin` moves 1:1 with the content instead
/// of staying pinned to the viewport edge (which would miss on every scroll frame).
const OVERSIZED_BAND_STEP: f32 = 512.0;

/// Extra margin (points) around the current clip inside which static sibling
/// children stay grouped in their cached run to avoid run-membership churn on
/// small scrolls.
const RUN_KEEP_MARGIN: f32 = 512.0;

/// Maximum number of retired layer ids kept in the thread-local cleanup ring.
const MAX_REMOVED_RING: usize = 2048;

thread_local! {
    static REMOVED: RefCell<(u64, VecDeque<(u64, CachedLayerId)>)> =
        const { RefCell::new((0, VecDeque::new())) };
}

/// Enqueues `id` for removal from GPUI windows on their next paint pass.
pub(crate) fn enqueue_removed(id: CachedLayerId) {
    let _ = REMOVED.try_with(|cell| {
        let mut state = cell.borrow_mut();
        state.0 = state.0.saturating_add(1);
        let seq = state.0;
        state.1.push_back((seq, id));
        while state.1.len() > MAX_REMOVED_RING {
            state.1.pop_front();
        }
    });
}

/// Removes any retired cached layer ids queued since `watermark` from `window`.
pub(crate) fn flush_removed(window: &mut gpui::Window, watermark: &Cell<u64>) {
    let seen = watermark.get();
    let latest = REMOVED.with(|cell| {
        let state = cell.borrow();
        let current = state.0;
        if current == seen {
            return None;
        }
        let ids: Vec<CachedLayerId> = state
            .1
            .iter()
            .filter_map(|&(seq, id)| (seq > seen).then_some(id))
            .collect();
        Some((current, ids))
    });
    if let Some((current, ids)) = latest {
        for id in ids {
            window.remove_cached_layer(id);
        }
        watermark.set(current);
    }
}

/// Returns the deterministic cached layer id for single subtree `uid`.
/// Uses even ids (`uid << 1`); sibling runs use odd ids (`hash | 1`).
#[inline]
pub(crate) const fn subtree_layer_id(uid: u64) -> CachedLayerId {
    CachedLayerId::new(uid << 1)
}

/// Returns the deterministic cached layer id for a static sibling run in
/// `parent_uid` starting with child `first_child_uid`.
#[inline]
fn run_layer_id(parent_uid: u64, first_child_uid: u64) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write_u64(parent_uid);
    hasher.write_u64(first_child_uid);
    hasher.finish() | 1
}

/// Memoized summary of a view's visual extent and layer-cache eligibility.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SubtreeCacheInfo {
    /// Tight visual bounds in the view's own coordinates (`[min_x, min_y, max_x, max_y]`).
    pub(crate) local_bounds: [f32; 4],
    /// Visible nodes in this subtree (0 when hidden/concealed/zero-opacity).
    pub(crate) node_count: u32,
    /// Number of visible text nodes in this subtree.
    pub(crate) text_count: u32,
    /// True if any visible node in this subtree renders a non-empty canvas.
    pub(crate) has_canvas: bool,
    /// True if this subtree contains any element or effect that must draw directly.
    pub(crate) has_uncacheable: bool,
}

impl SubtreeCacheInfo {
    const EMPTY: Self = Self {
        local_bounds: [0.0; 4],
        node_count: 0,
        text_count: 0,
        has_canvas: false,
        has_uncacheable: false,
    };
}

/// Computes (and memoizes by node revision) cache eligibility and tight local
/// visual bounds for `view`.
pub(crate) fn subtree_info(view: &NshView) -> SubtreeCacheInfo {
    if let Some(info) = view.cached_info() {
        return info;
    }

    let (clips, mut bounds, mut has_uncacheable, text_count_self, has_canvas_self) = {
        let state = view.state();
        if state.style.hidden || state.concealed || state.opacity() <= 0.0 {
            let empty = SubtreeCacheInfo::EMPTY;
            view.set_cached_info(empty);
            return empty;
        }
        let uncacheable = self_uncacheable(&state);
        let bounds = self_local_bounds(&state);
        let text = u32::from(state.text.is_some());
        let canvas = state.canvas.as_ref().is_some_and(|c| c.drawing.is_some());
        (state.clips(), bounds, uncacheable, text, canvas)
    };

    let mut node_count = 1u32;
    let mut text_count = text_count_self;
    let mut has_canvas = has_canvas_self;

    view.with_children(|children| {
        for child in children {
            let cinfo = subtree_info(child);
            has_uncacheable |= cinfo.has_uncacheable;
            if cinfo.node_count == 0 {
                continue;
            }
            node_count = node_count.saturating_add(cinfo.node_count);
            text_count = text_count.saturating_add(cinfo.text_count);
            has_canvas |= cinfo.has_canvas;
            if !clips && !has_uncacheable {
                let local = {
                    let p = view.state();
                    let cs = child.state();
                    p.child_matrix(&cs)
                };
                if !union_transformed_rect(&mut bounds, cinfo.local_bounds, &local) {
                    has_uncacheable = true;
                }
            }
        }
    });

    if !bounds.iter().all(|v| v.is_finite()) {
        has_uncacheable = true;
    }

    let info = SubtreeCacheInfo {
        local_bounds: bounds,
        node_count,
        text_count,
        has_canvas,
        has_uncacheable,
    };
    view.set_cached_info(info);
    info
}

fn self_uncacheable(state: &ViewState) -> bool {
    if !state.natives.is_empty()
        || matches!(
            state.kind,
            ViewKind::Input | ViewKind::WebView | ViewKind::Video | ViewKind::Native
        )
        || state.shader.is_some()
        || state.image.as_ref().is_some_and(|i| !i.is_cacheable())
        || state.canvas.as_ref().is_some_and(|c| !c.is_cacheable())
        || state.style.clip_path().is_some()
    {
        return true;
    }
    if let Some(s) = state.scroll.as_ref()
        && (s.live.alpha > 0.0
            || s.zooms()
            || (s.zoom - 1.0).abs() > 1e-5
            || s.spinner(
                state.frame.size.width,
                state.events.contains(EventKind::Refresh),
            )
            .is_some())
    {
        return true;
    }
    if let Some(e) = state.style.effects.as_deref()
        && (!e.background_images.is_empty()
            || e.filters
                .iter()
                .any(|f| matches!(f, Filter::DropShadow { .. })))
    {
        return true;
    }
    false
}

fn self_local_bounds(state: &ViewState) -> [f32; 4] {
    let w = state.frame.size.width.0.max(0.0);
    let h = state.frame.size.height.0.max(0.0);
    let mut b = [0.0f32, 0.0, w, h];

    let outline = state.style.outline;
    if outline.is_visible() {
        let pad = (outline.width.0 + outline.offset.0).max(0.0) + 1.0;
        b[0] = b[0].min(-pad);
        b[1] = b[1].min(-pad);
        b[2] = b[2].max(w + pad);
        b[3] = b[3].max(h + pad);
    }

    if let Some(effects) = state.style.effects.as_deref() {
        for shadow in &effects.box_shadows {
            if !shadow.inset && shadow.is_visible() {
                let pad = shadow.spread.0.max(0.0) + shadow.blur.0.max(0.0) * 1.5 + 1.0;
                b[0] = b[0].min(shadow.offset_x.0 - pad);
                b[1] = b[1].min(shadow.offset_y.0 - pad);
                b[2] = b[2].max(w + shadow.offset_x.0 + pad);
                b[3] = b[3].max(h + shadow.offset_y.0 + pad);
            }
        }
    }

    if !state.clips()
        && let Some(drawn) = &state.text
        && let Some(shadow) = drawn.text.shadow
        && shadow.color.0 & 0xff != 0
    {
        let pad = shadow.radius.0.max(0.0) * 0.8 + 2.0;
        let ox = drawn.content.origin.x.0 + shadow.offset_x.0;
        let oy = drawn.content.origin.y.0 + shadow.offset_y.0;
        let cw = drawn.content.size.width.0.max(0.0);
        let ch = drawn.content.size.height.0.max(0.0);
        b[0] = b[0].min(ox - pad);
        b[1] = b[1].min(oy - pad);
        b[2] = b[2].max(ox + cw + pad);
        b[3] = b[3].max(oy + ch + pad);
    }

    if let Some(ov) = state.overflow {
        b[0] = b[0].min(ov.origin.x.0);
        b[1] = b[1].min(ov.origin.y.0);
        b[2] = b[2].max(ov.origin.x.0 + ov.size.width.0.max(0.0));
        b[3] = b[3].max(ov.origin.y.0 + ov.size.height.0.max(0.0));
    }

    b
}

fn union_transformed_rect(dst: &mut [f32; 4], src: [f32; 4], m: &Mat4) -> bool {
    let corners = [
        (src[0], src[1]),
        (src[2], src[1]),
        (src[0], src[3]),
        (src[2], src[3]),
    ];
    for (x, y) in corners {
        let Some((mx, my)) = m.map(x, y) else {
            return false;
        };
        if !mx.is_finite() || !my.is_finite() {
            return false;
        }
        dst[0] = dst[0].min(mx);
        dst[1] = dst[1].min(my);
        dst[2] = dst[2].max(mx);
        dst[3] = dst[3].max(my);
    }
    true
}

#[inline]
pub(crate) fn is_pure_translation(m: &Mat4) -> Option<(f32, f32)> {
    if !m.is_affine() {
        return None;
    }
    let [a, b, c, d, tx, ty] = m.affine();
    ((a - 1.0).abs() < 1e-5
        && (d - 1.0).abs() < 1e-5
        && b.abs() < 1e-6
        && c.abs() < 1e-6
        && tx.is_finite()
        && ty.is_finite())
    .then_some((tx, ty))
}

#[inline]
fn meets_threshold(
    node_count: u32,
    text_count: u32,
    has_canvas: bool,
    width: f32,
    height: f32,
) -> bool {
    if width < MIN_LAYER_SIDE || height < MIN_LAYER_SIDE {
        return false;
    }
    if width * height < MIN_LAYER_AREA {
        return false;
    }
    node_count >= MIN_LAYER_NODES || has_canvas || (text_count >= 2 && node_count >= 2)
}

/// Where content drawn at `at` (points, window coordinates) lands relative
/// to a layer composited at `origin`, in 1/64 device pixels. GPUI captures
/// a layer's content where it is drawn and composites its tiles at the
/// layer origin rounded to whole device pixels, while text is placed on the
/// device pixel grid by where it is drawn: a layer moved by part of a pixel
/// (a scroll by a trackpad's fractional delta) is reused only when its
/// content still lands where drawing it again would put it.
#[inline]
fn device_offset(scale: f32, at: f32, origin: Pixels) -> i32 {
    let offset = at * scale - (f32::from(origin) * scale).round();
    (offset * 64.0).round() as i32
}

#[inline]
fn snap_rect_outward(x0: f32, y0: f32, x1: f32, y1: f32) -> Option<(f32, f32, f32, f32)> {
    let l = x0.floor();
    let t = y0.floor();
    let r = x1.ceil();
    let b = y1.ceil();
    if !l.is_finite() || !t.is_finite() || !r.is_finite() || !b.is_finite() || r <= l || b <= t {
        return None;
    }
    Some((l, t, r - l, b - t))
}

#[inline]
fn band_axis(v0: f32, vlen: f32, c0: f32, clen: f32) -> Option<(f32, f32)> {
    let vis0 = v0.max(c0);
    let vis1 = (v0 + vlen).min(c0 + clen);
    if vis1 <= vis0 {
        return None;
    }
    if vlen <= clen + MAX_UNCLIPPED_OVERSHOOT && vlen <= MAX_UNCLIPPED_SIDE {
        return Some((v0, vlen));
    }
    let rel0 = (vis0 - v0).max(0.0);
    let b0 = ((rel0 / OVERSIZED_BAND_STEP).floor() * OVERSIZED_BAND_STEP).max(0.0);
    // Fixed-height band anchored at `b0`: guarantees `[b0, b1]` covers `[rel0, rel1]`
    // while only transitioning once per `OVERSIZED_BAND_STEP` (when `b0` steps),
    // never a second time when `rel1` crosses a grid line.
    let b1 = (b0 + clen.ceil() + OVERSIZED_BAND_STEP).min(vlen);
    if !b0.is_finite() || !b1.is_finite() || b1 <= b0 {
        return None;
    }
    Some((v0 + b0, b1 - b0))
}

#[inline]
fn choose_layer_bounds(
    visual: Bounds<Pixels>,
    outer_clip: Bounds<Pixels>,
) -> Option<Bounds<Pixels>> {
    let visible = visual.intersect(&outer_clip);
    if visible.is_empty() {
        return None;
    }
    let (x, w) = band_axis(
        f32::from(visual.left()),
        f32::from(visual.size.width),
        f32::from(outer_clip.left()),
        f32::from(outer_clip.size.width),
    )?;
    let (y, h) = band_axis(
        f32::from(visual.top()),
        f32::from(visual.size.height),
        f32::from(outer_clip.top()),
        f32::from(outer_clip.size.height),
    )?;
    Some(crate::render::bounds(x, y, w, h))
}

#[inline]
fn inflate_bounds(bounds: Bounds<Pixels>, margin: f32) -> Bounds<Pixels> {
    let l = f32::from(bounds.left()) - margin;
    let t = f32::from(bounds.top()) - margin;
    let r = f32::from(bounds.right()) + margin;
    let b = f32::from(bounds.bottom()) + margin;
    crate::render::bounds(l, t, (r - l).max(0.0), (b - t).max(0.0))
}

/// Attempts to paint or composite `view`'s subtree via a single GPU cached layer.
/// Returns `true` when handled (composited or captured into a cached layer, or
/// completely clipped away); returns `false` when caller should paint directly.
pub(crate) fn try_paint_cached_subtree(cx: &mut DrawCx<'_>, view: &NshView, world: Mat4) -> bool {
    if cx.in_cached_capture {
        if view.take_had_cached_layer() {
            cx.window.remove_cached_layer(subtree_layer_id(view.uid()));
        }
        return false;
    }
    let streak = view.observe_stability(cx.paint_pass);
    let Some((tx, ty)) = (streak >= STABLE_FRAMES)
        .then(|| is_pure_translation(&world))
        .flatten()
    else {
        if view.take_had_cached_layer() {
            cx.window.remove_cached_layer(subtree_layer_id(view.uid()));
        }
        return false;
    };
    if cx.rounded.is_some() || cx.has_clip_polygon() || cx.matrix.is_some() {
        if view.take_had_cached_layer() {
            cx.window.remove_cached_layer(subtree_layer_id(view.uid()));
        }
        return false;
    }
    let info = subtree_info(view);
    if info.node_count == 0 || info.has_uncacheable {
        if view.take_had_cached_layer() {
            cx.window.remove_cached_layer(subtree_layer_id(view.uid()));
        }
        return false;
    }
    let Some((lx, ly, w, h)) = snap_rect_outward(
        info.local_bounds[0],
        info.local_bounds[1],
        info.local_bounds[2],
        info.local_bounds[3],
    ) else {
        if view.take_had_cached_layer() {
            cx.window.remove_cached_layer(subtree_layer_id(view.uid()));
        }
        return false;
    };
    if !meets_threshold(info.node_count, info.text_count, info.has_canvas, w, h) {
        if view.take_had_cached_layer() {
            cx.window.remove_cached_layer(subtree_layer_id(view.uid()));
        }
        return false;
    }

    // Use local-snapped origin offset so scroll translation keeps size and
    // relative offset bit-exact even across fractional offsets.
    let visual = crate::render::bounds(tx + lx, ty + ly, w, h);
    let Some(layer_bounds) = choose_layer_bounds(visual, cx.clip) else {
        // Offscreen and guaranteed free of Hosted/animated descendants.
        view.release_cached_layers(cx.window);
        return true;
    };

    let mut hasher = FxHasher::default();
    hasher.write_u64(view.revision());
    hasher.write_u32(cx.opacity.to_bits());
    hasher.write_u8(u8::from(cx.dark));
    hasher.write_i32(device_offset(cx.scale, tx, layer_bounds.origin.x));
    hasher.write_i32(device_offset(cx.scale, ty, layer_bounds.origin.y));
    let revision = hasher.finish();

    view.set_had_cached_layer(true);
    paint_with_layer(
        cx,
        subtree_layer_id(view.uid()),
        layer_bounds,
        revision,
        None,
        |sub_cx| {
            crate::render::draw_view_uncached(sub_cx, view, world);
        },
    );
    true
}

struct RunEntry<'a> {
    child: &'a NshView,
    local: Mat4,
    world: Mat4,
    tx: f32,
    ty: f32,
    info: SubtreeCacheInfo,
}

/// Paints `parent`'s `children`, batching maximal contiguous runs of static
/// cacheable siblings into shared GPU cached layers while drawing animated or
/// uncacheable children directly in exact z-order.
pub(crate) fn paint_children_with_cache(
    cx: &mut DrawCx<'_>,
    parent: &NshView,
    world: Mat4,
    children: &[NshView],
) {
    if cx.in_cached_capture
        || cx.rounded.is_some()
        || cx.has_clip_polygon()
        || cx.matrix.is_some()
        || is_pure_translation(&world).is_none()
        || children.len() < 2
    {
        for id in parent.take_cached_runs() {
            cx.window.remove_cached_layer(CachedLayerId::new(id));
        }
        for child in children {
            let local = parent.state().child_matrix(&child.state());
            crate::render::draw_view(cx, child, world, local);
        }
        return;
    }

    let prev_runs = parent.take_cached_runs();
    let mut new_runs: Vec<u64> = Vec::new();
    let mut run: Vec<RunEntry<'_>> = Vec::new();
    let keep_clip = inflate_bounds(cx.clip, RUN_KEEP_MARGIN);

    for child in children {
        let local = {
            let ps = parent.state();
            let cs = child.state();
            (!(cs.style.hidden || cs.concealed || cs.opacity() <= 0.0))
                .then(|| ps.child_matrix(&cs))
        };
        let Some(local) = local else {
            child.release_cached_layers(cx.window);
            continue;
        };
        let child_world = Mat4::mul(&local, &world);
        let info = subtree_info(child);
        if info.node_count == 0 {
            child.release_cached_layers(cx.window);
            continue;
        }
        if !info.has_uncacheable
            && let Some((tx, ty)) = is_pure_translation(&child_world)
        {
            if let Some((lx, ly, w, h)) = snap_rect_outward(
                info.local_bounds[0],
                info.local_bounds[1],
                info.local_bounds[2],
                info.local_bounds[3],
            ) {
                let child_visual = crate::render::bounds(tx + lx, ty + ly, w, h);
                if child_visual.intersect(&keep_clip).is_empty() {
                    // Far outside viewport with zero Hosted/animated descendants.
                    child.release_cached_layers(cx.window);
                    continue;
                }
            }
            let streak = child.observe_stability(cx.paint_pass);
            if streak >= STABLE_FRAMES {
                run.push(RunEntry {
                    child,
                    local,
                    world: child_world,
                    tx,
                    ty,
                    info,
                });
                continue;
            }
        }

        flush_run(cx, parent.uid(), world, &mut run, &mut new_runs);
        crate::render::draw_view(cx, child, world, local);
    }

    flush_run(cx, parent.uid(), world, &mut run, &mut new_runs);

    for old_id in prev_runs {
        if !new_runs.contains(&old_id) {
            cx.window.remove_cached_layer(CachedLayerId::new(old_id));
        }
    }
    parent.set_cached_runs(new_runs);
}

fn flush_run(
    cx: &mut DrawCx<'_>,
    parent_uid: u64,
    parent_world: Mat4,
    run: &mut Vec<RunEntry<'_>>,
    new_runs: &mut Vec<u64>,
) {
    if run.is_empty() {
        return;
    }
    if run.len() == 1 {
        let entry = run.pop().expect("len == 1");
        crate::render::draw_view(cx, entry.child, parent_world, entry.local);
        return;
    }

    let anchor = (run[0].tx, run[0].ty);
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let mut total_nodes = 0u32;
    let mut total_texts = 0u32;
    let mut any_canvas = false;

    for entry in run.iter() {
        let dx = entry.tx - anchor.0;
        let dy = entry.ty - anchor.1;
        min_x = min_x.min(dx + entry.info.local_bounds[0]);
        min_y = min_y.min(dy + entry.info.local_bounds[1]);
        max_x = max_x.max(dx + entry.info.local_bounds[2]);
        max_y = max_y.max(dy + entry.info.local_bounds[3]);
        total_nodes = total_nodes.saturating_add(entry.info.node_count);
        total_texts = total_texts.saturating_add(entry.info.text_count);
        any_canvas |= entry.info.has_canvas;
    }

    let Some((rx, ry, w, h)) = snap_rect_outward(min_x, min_y, max_x, max_y) else {
        for entry in run.drain(..) {
            crate::render::draw_view_uncached(cx, entry.child, entry.world);
        }
        return;
    };
    if !meets_threshold(total_nodes, total_texts, any_canvas, w, h) {
        for entry in run.drain(..) {
            if entry.child.take_had_cached_layer() {
                cx.window
                    .remove_cached_layer(subtree_layer_id(entry.child.uid()));
            }
            crate::render::draw_view_uncached(cx, entry.child, entry.world);
        }
        return;
    }

    let visual = crate::render::bounds(anchor.0 + rx, anchor.1 + ry, w, h);
    let Some(layer_bounds) = choose_layer_bounds(visual, cx.clip) else {
        for entry in run.drain(..) {
            entry.child.release_cached_layers(cx.window);
        }
        return;
    };

    for entry in run.iter() {
        if entry.child.take_had_cached_layer() {
            cx.window
                .remove_cached_layer(subtree_layer_id(entry.child.uid()));
        }
    }

    let raw_id = run_layer_id(parent_uid, run[0].child.uid());
    new_runs.push(raw_id);

    let mut hasher = FxHasher::default();
    hasher.write_usize(run.len());
    hasher.write_u32(cx.opacity.to_bits());
    hasher.write_u8(u8::from(cx.dark));
    for entry in run.iter() {
        hasher.write_u64(entry.child.uid());
        hasher.write_u64(entry.child.revision());
        hasher.write_i32(device_offset(cx.scale, entry.tx, layer_bounds.origin.x));
        hasher.write_i32(device_offset(cx.scale, entry.ty, layer_bounds.origin.y));
    }
    let revision = hasher.finish();

    paint_with_layer(
        cx,
        CachedLayerId::new(raw_id),
        layer_bounds,
        revision,
        Some(run.len()),
        |sub_cx| {
            for entry in run.iter() {
                crate::render::draw_view_uncached(sub_cx, entry.child, entry.world);
            }
        },
    );
    run.clear();
}

/// `NSH_LAYER_CACHE_DEBUG=1`: every cached layer is outlined where it is
/// composited ([`debug_outline`]) and every capture is logged to stderr, to
/// tell stale cached content from what is drawn directly.
fn debug_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NSH_LAYER_CACHE_DEBUG").is_ok_and(|v| v != "0"))
}

/// The tile side GPUI's cached layers are rasterized in (device pixels,
/// gpui-fast's `fast::layers::paint::TILE_SIZE`), for the debug grid.
const DEBUG_TILE_SIZE: f32 = 512.0;

/// Outlines cached layer `id` over `bounds` (window coordinates): a color of
/// its own (the same from frame to frame), solid for a subtree and dashed
/// for a sibling run, 3 px on a frame that captured it again and 1 px on a
/// frame that reused it, with its tile grid in a fainter shade.
fn debug_outline(
    window: &mut gpui::Window,
    id: CachedLayerId,
    bounds: Bounds<Pixels>,
    scale: f32,
    run: bool,
    captured: bool,
) {
    let mut hasher = FxHasher::default();
    hasher.write_u64(id.as_u64());
    let hue = (hasher.finish() % 360) as f32 / 360.0;
    let color = gpui::hsla(hue, 0.9, 0.6, 1.0);
    let grid = gpui::hsla(hue, 0.9, 0.6, 0.35);
    let width = if captured { 3.0 } else { 1.0 };
    let style = if run {
        gpui::BorderStyle::Dashed
    } else {
        gpui::BorderStyle::Solid
    };
    window.paint_quad(gpui::quad(
        bounds,
        gpui::Corners::default(),
        gpui::transparent_black(),
        gpui::Edges::all(gpui::px(width)),
        color,
        style,
    ));
    // GPUI cuts tiles from the layer origin rounded to device pixels.
    let left = f32::from(bounds.left());
    let top = f32::from(bounds.top());
    let right = f32::from(bounds.right());
    let bottom = f32::from(bounds.bottom());
    let step = DEBUG_TILE_SIZE / scale;
    let mut x = (left * scale).round() / scale + step;
    while x < right {
        window.paint_quad(gpui::fill(
            crate::render::bounds(x, top, 1.0, bottom - top),
            grid,
        ));
        x += step;
    }
    let mut y = (top * scale).round() / scale + step;
    while y < bottom {
        window.paint_quad(gpui::fill(
            crate::render::bounds(left, y, right - left, 1.0),
            grid,
        ));
        y += step;
    }
}

fn paint_with_layer(
    cx: &mut DrawCx<'_>,
    id: CachedLayerId,
    layer_bounds: Bounds<Pixels>,
    revision: u64,
    run_members: Option<usize>,
    paint: impl FnOnce(&mut DrawCx<'_>),
) -> bool {
    let outer_clip = cx.clip;
    let opacity = cx.opacity;
    let dark = cx.dark;
    let paint_pass = cx.paint_pass;
    let track_measured = cx.measured.is_some();
    let app = &mut *cx.app;
    let mut sub_measured: Option<(u32, f64)> = None;
    let ran = cx
        .window
        .with_content_mask(Some(ContentMask { bounds: outer_clip }), |window| {
            window.paint_cached_layer(
                id,
                layer_bounds,
                revision,
                CachedLayerOptions::default(),
                |window| {
                    let mut sub_cx = crate::render::DrawCx::for_cached_capture(
                        window,
                        app,
                        layer_bounds,
                        dark,
                        opacity,
                        paint_pass,
                        track_measured,
                    );
                    paint(&mut sub_cx);
                    sub_measured = sub_cx.measured;
                },
            )
        });
    if let (Some((paths, ms)), Some((sub_paths, sub_ms))) = (&mut cx.measured, sub_measured) {
        *paths += sub_paths;
        *ms += sub_ms;
    }
    if debug_enabled() {
        let captured = ran.is_some();
        if captured {
            let kind = match run_members {
                Some(n) => format!("run({n})"),
                None => format!("subtree(uid {})", id.as_u64() >> 1),
            };
            eprintln!(
                "[nsh-layer-cache] pass {paint_pass} capture {kind} id={:016x} rev={revision:016x} \
                 bounds=({:.1},{:.1} {:.1}x{:.1}) clip=({:.1},{:.1} {:.1}x{:.1})",
                id.as_u64(),
                f32::from(layer_bounds.left()),
                f32::from(layer_bounds.top()),
                f32::from(layer_bounds.size.width),
                f32::from(layer_bounds.size.height),
                f32::from(outer_clip.left()),
                f32::from(outer_clip.top()),
                f32::from(outer_clip.size.width),
                f32::from(outer_clip.size.height),
            );
        }
        let scale = cx.scale;
        cx.window
            .with_content_mask(Some(ContentMask { bounds: outer_clip }), |window| {
                debug_outline(
                    window,
                    id,
                    layer_bounds,
                    scale,
                    run_members.is_some(),
                    captured,
                );
            });
    }
    if ran.is_some() {
        cx.cache_misses += 1;
        true
    } else {
        cx.cache_hits += 1;
        false
    }
}

#[cfg(test)]
mod tests {
    use nsh_core::{Bounds, BoxShadow, Color, Effects, NodeId, Points, ViewKind};

    use super::*;
    use crate::geom::rect;

    #[test]
    fn changed_bumps_ancestor_revisions_and_resets_stability() {
        let root = NshView::new(1 as NodeId, ViewKind::View);
        let parent = NshView::new(2 as NodeId, ViewKind::View);
        let child = NshView::new(3 as NodeId, ViewKind::View);
        parent.insert_before(&root, None);
        child.insert_before(&parent, None);

        assert_eq!(child.observe_stability(1), 0);
        assert_eq!(child.observe_stability(1), 0, "idempotent within one pass");
        assert_eq!(child.observe_stability(2), 1);
        assert_eq!(child.observe_stability(3), 2);
        let _ = subtree_info(&root);
        assert!(root.cached_info().is_some());

        let old_root_rev = root.revision();
        let old_parent_rev = parent.revision();
        child.changed();

        assert!(root.revision() > old_root_rev);
        assert!(parent.revision() > old_parent_rev);
        assert!(root.cached_info().is_none());
        assert_eq!(child.observe_stability(4), 0);
        assert_eq!(parent.observe_stability(4), 0);
    }

    #[test]
    fn subtree_info_marks_uncacheable_and_expands_bounds() {
        let root = NshView::new(10 as NodeId, ViewKind::View);
        let child = NshView::new(11 as NodeId, ViewKind::View);
        {
            let mut s = root.state_mut();
            s.frame = rect(0.0, 0.0, 100.0, 40.0);
        }
        {
            let mut cs = child.state_mut();
            cs.frame = rect(80.0, 10.0, 30.0, 40.0);
            cs.style.effects = Some(std::sync::Arc::new(Effects {
                box_shadows: vec![BoxShadow {
                    offset_x: Points(4.0),
                    offset_y: Points(6.0),
                    blur: Points(4.0),
                    spread: Points(2.0),
                    color: Color::BLACK,
                    inset: false,
                }],
                ..Effects::default()
            }));
        }
        child.insert_before(&root, None);

        let info = subtree_info(&root);
        assert!(!info.has_uncacheable);
        assert_eq!(info.node_count, 2);
        // Non-clipping root expands right/bottom for overflowing shadowed child.
        assert!(info.local_bounds[2] > 110.0, "{:?}", info.local_bounds);
        assert!(info.local_bounds[3] > 50.0, "{:?}", info.local_bounds);

        // Native view inside child marks ancestor subtree uncacheable.
        let native_child = NshView::new(12 as NodeId, ViewKind::WebView);
        native_child.state_mut().frame = Bounds::default();
        native_child.insert_before(&child, None);
        let updated = subtree_info(&root);
        assert!(updated.has_uncacheable);
    }

    #[test]
    fn clear_cache_info_invalidates_ancestors_without_resetting_stability() {
        let root = NshView::new(30 as NodeId, ViewKind::View);
        let parent = NshView::new(31 as NodeId, ViewKind::View);
        let leaf = NshView::new(32 as NodeId, ViewKind::View);
        parent.insert_before(&root, None);
        leaf.insert_before(&parent, None);

        assert_eq!(leaf.observe_stability(1), 0);
        assert_eq!(leaf.observe_stability(2), 1);
        assert_eq!(parent.observe_stability(2), 0);
        let _ = subtree_info(&root);
        assert!(root.cached_info().is_some());
        assert!(parent.cached_info().is_some());
        assert!(leaf.cached_info().is_some());

        let rev_root = root.revision();
        let rev_leaf = leaf.revision();
        leaf.clear_cache_info();

        assert_eq!(root.revision(), rev_root);
        assert_eq!(leaf.revision(), rev_leaf);
        assert!(root.cached_info().is_none());
        assert!(parent.cached_info().is_none());
        assert!(leaf.cached_info().is_none());
        assert_eq!(leaf.observe_stability(3), 2);
    }

    #[test]
    fn cached_in_subtree_flag_propagates_across_reparenting() {
        let root_a = NshView::new(40 as NodeId, ViewKind::View);
        let mid = NshView::new(41 as NodeId, ViewKind::View);
        let leaf = NshView::new(42 as NodeId, ViewKind::View);
        let root_b = NshView::new(43 as NodeId, ViewKind::View);
        mid.insert_before(&root_a, None);
        leaf.insert_before(&mid, None);

        assert!(!root_a.has_cached_in_subtree_flag());
        leaf.set_had_cached_layer(true);
        assert!(leaf.has_cached_in_subtree_flag());
        assert!(mid.has_cached_in_subtree_flag());
        assert!(root_a.has_cached_in_subtree_flag());

        mid.insert_before(&root_b, None);
        assert!(root_b.has_cached_in_subtree_flag());

        root_b.set_cached_runs(vec![123]);
        assert!(root_b.has_any_cached_layer_in_subtree());
    }

    #[test]
    fn canvas_settlement_clears_stale_uncacheable_info() {
        use nsh_core::canvas::{CanvasOp, Drawing};
        use std::sync::Arc;

        let root = NshView::new(50 as NodeId, ViewKind::View);
        let canvas_view = NshView::new(51 as NodeId, ViewKind::Canvas);
        root.state_mut().frame = rect(0.0, 0.0, 200.0, 200.0);
        {
            let mut cs = canvas_view.state_mut();
            cs.frame = rect(0.0, 0.0, 120.0, 120.0);
            let mut state = crate::canvas::CanvasState::default();
            state.drawing = Some(Arc::new(Drawing {
                ops: vec![CanvasOp::Save],
                ..Drawing::default()
            }));
            cs.canvas = Some(state);
        }
        canvas_view.insert_before(&root, None);

        let before = subtree_info(&root);
        assert!(before.has_uncacheable);

        canvas_view
            .state()
            .canvas
            .as_ref()
            .unwrap()
            .mark_settled_for_test();
        canvas_view.clear_cache_info();

        let after = subtree_info(&root);
        assert!(!after.has_uncacheable);
        assert!(after.has_canvas);
    }

    #[test]
    fn oversized_layer_bands_stay_content_anchored_across_scroll() {
        let clip = crate::render::bounds(860.0, 75.0, 120.0, 725.0);
        // 16,000pt tall code editor scrolling by 20pt steps within one 512pt band.
        let b1 = choose_layer_bounds(crate::render::bounds(860.0, -500.0, 120.0, 16_000.0), clip)
            .expect("visible");
        let b2 = choose_layer_bounds(crate::render::bounds(860.0, -520.0, 120.0, 16_000.0), clip)
            .expect("visible");
        let b3 = choose_layer_bounds(crate::render::bounds(860.0, -540.0, 120.0, 16_000.0), clip)
            .expect("visible");

        // X axis stays unclipped; Y axis stays quantized to a fixed-height band
        // anchored at b0 = 512 (height = clen.ceil() + 512 = 1237) across the full 512pt step.
        assert_eq!(f32::from(b1.size.width), 120.0);
        assert_eq!(f32::from(b1.size.height), 725.0 + OVERSIZED_BAND_STEP);
        assert_eq!(b1.size, b2.size);
        assert_eq!(b2.size, b3.size);

        // Even when rel1 crosses 1536 (at ty = -750, rel0 = 825 still in [512, 1024)),
        // the fixed-height band stays on the exact same b0 = 512 with zero extra transition.
        let b4 = choose_layer_bounds(crate::render::bounds(860.0, -750.0, 120.0, 16_000.0), clip)
            .expect("visible");
        assert_eq!(b1.size, b4.size);
        assert_eq!(
            device_offset(2.0, -500.0, b1.origin.y),
            device_offset(2.0, -750.0, b4.origin.y),
        );

        // Layer origin translates 1:1 with content, keeping device_offset invariant (cache hit).
        assert_eq!(f32::from(b1.origin.y) - f32::from(b2.origin.y), 20.0);
        assert_eq!(f32::from(b2.origin.y) - f32::from(b3.origin.y), 20.0);
        assert_eq!(
            device_offset(2.0, -500.0, b1.origin.y),
            device_offset(2.0, -520.0, b2.origin.y),
        );
        assert_eq!(
            device_offset(2.0, -520.0, b2.origin.y),
            device_offset(2.0, -540.0, b3.origin.y),
        );

        // Crossing a 512pt band boundary shifts the band cleanly and still covers the clip.
        let b_next =
            choose_layer_bounds(crate::render::bounds(860.0, -1020.0, 120.0, 16_000.0), clip)
                .expect("visible");
        assert!(f32::from(b_next.top()) <= f32::from(clip.top()));
        assert!(f32::from(b_next.bottom()) >= f32::from(clip.bottom()));
    }
}
