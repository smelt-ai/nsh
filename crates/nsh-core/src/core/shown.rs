//! Whether canvases show ([`EventKind::CanvasShown`]): a canvas listening
//! for it is told when it goes out of the scroll views around it, or comes
//! back, and `<Canvas>` runs its drawing code only while it shows.
//!
//! A canvas shows while its native view is made (one a scroll view holds
//! back, `cull.rs`, does not), it is laid out (`display: none` on it or a
//! view around it gives it no box) and its rectangle, widened by how far the
//! transforms around it may move it ([`Core::reach_of`]), meets the
//! viewport of each scroll view it is in. Where that cannot be told (a
//! transform without a bound, a list's cell, a view laid out on its own or
//! placed elsewhere) it counts as showing, as it was before it listened.
//! The core looks again when a commit laid out, scrolled or made views.

use super::{Core, Key, NO_KEY};
use crate::{EventPayload, PayloadValue, element::JsEvent, ops::EventKind, text::TextEngine};

impl<M: TextEngine> Core<M> {
    /// Canvas `key` listens for whether it shows (from now on: it showed).
    pub(super) fn track_shown(&mut self, key: Key) {
        let id = self.nodes[key as usize].id;
        if !self.shown.iter().any(|&(k, i, _)| k == key && i == id) {
            self.shown.push((key, id, true));
            self.shown_dirty = true;
        }
    }

    /// Tells the canvases whose showing changed since the last look.
    pub(super) fn sync_shown(&mut self) {
        if !std::mem::take(&mut self.shown_dirty) {
            return;
        }
        let mut tracked = std::mem::take(&mut self.shown);
        tracked.retain_mut(|(key, id, told)| {
            // Released (its key may be another node's now), or no longer
            // listening.
            let Some(node) = self.nodes.get(*key as usize) else {
                return false;
            };
            if node.id != *id || !node.events.contains(EventKind::CanvasShown) {
                return false;
            }
            let shows = self.shows(*key);
            if shows != *told {
                *told = shows;
                self.events.push(JsEvent::Event(
                    *id,
                    EventKind::CanvasShown,
                    EventPayload::Value(PayloadValue::Bool(shows)),
                ));
            }
            true
        });
        self.shown = tracked;
    }

    /// Whether `key` shows in the scroll views around it.
    fn shows(&self, key: Key) -> bool {
        let node = &self.nodes[key as usize];
        if !node.created || self.displays_none(key) {
            return false;
        }
        let f = node.frame;
        if f.origin.x.0.is_nan() {
            return true;
        }
        // Its rectangle in its parent's coordinates (left, top, right,
        // bottom), taken up through the views around it.
        let mut r = (
            f.origin.x.0,
            f.origin.y.0,
            f.origin.x.0 + f.size.width.0,
            f.origin.y.0 + f.size.height.0,
        );
        let mut k = key;
        loop {
            let n = &self.nodes[k as usize];
            if n.is_layout_root() || n.is_overlay() || n.sticky.is_some() || n.follow.is_some() {
                return true;
            }
            let Some((rx, ry)) = self.reach_of(k) else {
                return true;
            };
            r = (r.0 - rx, r.1 - ry, r.2 + rx, r.3 + ry);
            let parent = n.parent;
            if parent == NO_KEY {
                return true;
            }
            let p = &self.nodes[parent as usize];
            if !p.created {
                return false;
            }
            if p.list.is_some() {
                return true;
            }
            if let Some(scroll) = p.scroll.as_deref() {
                let (x, y) = scroll.offset;
                let (w, h) = (p.frame.size.width.0, p.frame.size.height.0);
                if w > 0.0 && h > 0.0 && !(r.0 < x + w && r.1 < y + h && r.2 > x && r.3 > y) {
                    return false;
                }
                r = (r.0 - x, r.1 - y, r.2 - x, r.3 - y);
            }
            let at = p.frame.origin;
            if at.x.0.is_nan() {
                return true;
            }
            r = (r.0 + at.x.0, r.1 + at.y.0, r.2 + at.x.0, r.3 + at.y.0);
            k = parent;
        }
    }
}
