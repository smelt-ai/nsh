//! Streaming text reveal for [`super::TextView`]: newly appended CJK characters
//! and Latin words fade in smoothly via inline opacity without re-laying out
//! settled text, stopping display frames completely once settled.

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    ops::Range,
    rc::Rc,
};

use nsh_core::{
    element::{Owner, Spawned, reduce_motion, request_animation_frame},
    prelude::*,
    time::Instant,
};

use crate::terminal::is_terminal;

/// Milliseconds a newly appended segment takes to fade in.
pub(super) const FADE_MS: f32 = 180.0;
/// Milliseconds between consecutive segment fade starts.
pub(super) const STAGGER_MS: f32 = 16.0;
/// Maximum stagger lag behind the current clock so long bursts never fall behind.
pub(super) const MAX_STAGGER_LAG_MS: f32 = 140.0;
/// Threshold below which an initial non-empty mount with `streaming_fade` fades
/// its first chunk smoothly instead of treating the content as settled history.
pub(super) const INITIAL_FADE_MAX_BYTES: usize = 160;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FadeSpan {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) begin_ms: f32,
}

#[derive(Default)]
struct ParagraphReveal {
    seen_bytes: usize,
    spans: Vec<FadeSpan>,
}

struct RevealInner {
    owner: Owner,
    epoch: Instant,
    now_ms: Cell<f32>,
    frame: Signal<u64>,
    scheduled: Cell<bool>,
    disposed: Cell<bool>,
    raf_task: RefCell<Option<Spawned<()>>>,
    prev_source: RefCell<Option<SharedString>>,
    generation: Cell<u64>,
    reset_generation: Cell<u64>,
    fade_on_reset: Cell<bool>,
    fade_new: Cell<bool>,
    last_start: Cell<f32>,
    max_end: Cell<f32>,
    paragraphs: RefCell<HashMap<String, ParagraphReveal>>,
}

/// Shared state tracking per-paragraph streaming fade spans for one `TextView`.
#[derive(Clone)]
pub(super) struct Reveal(Rc<RevealInner>);

impl Reveal {
    pub(super) fn new() -> Self {
        Self(Rc::new(RevealInner {
            owner: Owner::current(),
            epoch: nsh_core::time::now(),
            now_ms: Cell::new(0.0),
            frame: signal(0u64),
            scheduled: Cell::new(false),
            disposed: Cell::new(false),
            raf_task: RefCell::new(None),
            prev_source: RefCell::new(None),
            generation: Cell::new(0),
            reset_generation: Cell::new(0),
            fade_on_reset: Cell::new(false),
            fade_new: Cell::new(false),
            last_start: Cell::new(f32::NEG_INFINITY),
            max_end: Cell::new(0.0),
            paragraphs: RefCell::new(HashMap::new()),
        }))
    }

    pub(super) fn dispose(&self) {
        self.0.disposed.set(true);
        self.0.scheduled.set(false);
        *self.0.raf_task.borrow_mut() = None;
        self.0.paragraphs.borrow_mut().clear();
    }

    pub(super) fn frame_signal(&self) -> Signal<u64> {
        self.0.frame
    }

    pub(super) fn set_enabled(&self, enabled: bool) {
        self.0.fade_new.set(enabled);
    }

    fn refresh_now(&self) -> f32 {
        let ms = nsh_core::time::now()
            .saturating_duration_since(self.0.epoch)
            .as_secs_f32()
            * 1000.0;
        let now = ms.max(self.0.now_ms.get());
        self.0.now_ms.set(now);
        now
    }

    pub(super) fn sync_source(&self, source: Option<&SharedString>, streaming_fade: bool) {
        self.refresh_now();
        let active_motion = streaming_fade && !is_terminal() && !reduce_motion();
        let Some(source) = source else {
            self.0.paragraphs.borrow_mut().clear();
            *self.0.prev_source.borrow_mut() = None;
            self.0.fade_new.set(false);
            self.0.fade_on_reset.set(false);
            self.0.last_start.set(f32::NEG_INFINITY);
            self.0.max_end.set(0.0);
            return;
        };

        let grew = self
            .0
            .prev_source
            .borrow()
            .as_ref()
            .is_some_and(|prev| source.starts_with(prev.as_str()));
        *self.0.prev_source.borrow_mut() = Some(source.clone());
        let next_gen = self.0.generation.get().wrapping_add(1);
        self.0.generation.set(next_gen);

        if !grew {
            self.0.paragraphs.borrow_mut().clear();
            self.0.reset_generation.set(next_gen);
            self.0
                .fade_on_reset
                .set(active_motion && source.len() <= INITIAL_FADE_MAX_BYTES);
            self.0.last_start.set(f32::NEG_INFINITY);
            self.0.max_end.set(0.0);
        }
        self.0.fade_new.set(streaming_fade);
    }

    fn should_fade_unseen(&self) -> bool {
        if self.0.generation.get() == self.0.reset_generation.get() {
            self.0.fade_on_reset.get()
        } else {
            self.0.fade_new.get()
        }
    }

    pub(super) fn should_track_paragraph(&self, key: &str) -> bool {
        if is_terminal() || reduce_motion() {
            return false;
        }
        self.0.fade_new.get()
            || self.should_fade_unseen()
            || self
                .0
                .paragraphs
                .borrow()
                .get(key)
                .is_some_and(|p| !p.spans.is_empty())
    }

    pub(super) fn begin_paragraph(&self, key: &str, plain: &str) -> Vec<FadeSpan> {
        let now = self.refresh_now();
        let should_fade_new = self.should_fade_unseen();

        let (result, need_tick) = {
            let mut paragraphs = self.0.paragraphs.borrow_mut();
            let entry = paragraphs.entry(key.to_owned()).or_insert_with(|| {
                let seen_bytes = if should_fade_new { 0 } else { plain.len() };
                ParagraphReveal {
                    seen_bytes,
                    spans: Vec::new(),
                }
            });

            entry.spans.retain(|s| s.begin_ms + FADE_MS > now && s.start < plain.len());
            for span in &mut entry.spans {
                span.end = span.end.min(plain.len());
            }
            entry
                .spans
                .retain(|s| s.start < s.end && plain.is_char_boundary(s.start) && plain.is_char_boundary(s.end));

            let mut seen = entry.seen_bytes.min(plain.len());
            while seen > 0 && !plain.is_char_boundary(seen) {
                seen -= 1;
            }

            if plain.len() > seen {
                if should_fade_new {
                    let units = segment_reveal_units(&plain[seen..]);
                    let count = units.len();
                    if count > 0 {
                        let max_begin = now + MAX_STAGGER_LAG_MS;
                        let avail = (self.0.last_start.get() + STAGGER_MS)
                            .max(now)
                            .min(max_begin);
                        let step = if count <= 1 {
                            STAGGER_MS
                        } else {
                            ((max_begin - avail) / (count - 1) as f32).clamp(0.0, STAGGER_MS)
                        };
                        let mut max_end = self.0.max_end.get();
                        let mut last_start = avail;
                        for (i, range) in units.into_iter().enumerate() {
                            let begin_ms = avail + i as f32 * step;
                            last_start = begin_ms;
                            max_end = max_end.max(begin_ms + FADE_MS);
                            entry.spans.push(FadeSpan {
                                start: seen + range.start,
                                end: seen + range.end,
                                begin_ms,
                            });
                        }
                        self.0.last_start.set(last_start);
                        self.0.max_end.set(max_end);
                    }
                }
                seen = plain.len();
            }
            entry.seen_bytes = seen;
            let need_tick = self.0.max_end.get() > now && !entry.spans.is_empty();
            (entry.spans.clone(), need_tick)
        };

        if need_tick {
            self.ensure_ticking();
        }
        result
    }

    fn ensure_ticking(&self) {
        if self.0.scheduled.get() || self.0.disposed.get() || !self.0.owner.is_live() {
            return;
        }
        self.0.scheduled.set(true);
        let this = self.clone();
        let owner = self.0.owner;
        let task = owner.run(move || {
            request_animation_frame(move |at| {
                this.0.scheduled.set(false);
                if this.0.disposed.get() {
                    return;
                }
                let ms = at
                    .saturating_duration_since(this.0.epoch)
                    .as_secs_f32()
                    * 1000.0;
                let now = ms.max(this.0.now_ms.get());
                this.0.now_ms.set(now);
                this.0.frame.update(|n| *n = n.wrapping_add(1));
                if now < this.0.max_end.get() {
                    this.ensure_ticking();
                } else {
                    {
                        let mut paragraphs = this.0.paragraphs.borrow_mut();
                        for p in paragraphs.values_mut() {
                            p.spans.retain(|s| s.begin_ms + FADE_MS > now);
                        }
                    }
                    this.0.max_end.set(0.0);
                    *this.0.raf_task.borrow_mut() = None;
                }
            })
        });
        *self.0.raf_task.borrow_mut() = Some(task);
    }

    pub(super) fn alpha(&self, begin_ms: f32) -> f32 {
        let t = ((self.0.now_ms.get() - begin_ms) / FADE_MS).clamp(0.0, 1.0);
        if t >= 1.0 {
            1.0
        } else {
            1.0 - (1.0 - t).powi(3)
        }
    }
}

fn is_single_unit_char(c: char) -> bool {
    let cp = c as u32;
    matches!(
        cp,
        0x3040..=0x30FF // Hiragana & Katakana
            | 0x3400..=0x4DBF // CJK Extension A
            | 0x4E00..=0x9FFF // CJK Unified Ideographs
            | 0xAC00..=0xD7AF // Hangul Syllables
            | 0xF900..=0xFAFF // CJK Compatibility Ideographs
            | 0x1F300..=0x1FAFF // Wide symbols & emoji
            | 0x20000..=0x2FA1F // CJK Extensions B..F
    )
}

fn is_opening_punct(c: char) -> bool {
    matches!(c, '（' | '(' | '「' | '『' | '【' | '《' | '〈' | '[' | '{' | '“' | '‘')
}

fn is_closing_punct(c: char) -> bool {
    matches!(
        c,
        '，' | '。'
            | '！'
            | '？'
            | '、'
            | '；'
            | '：'
            | '）'
            | '》'
            | '〉'
            | '」'
            | '』'
            | '】'
            | '…'
            | '—'
            | '.'
            | ','
            | '!'
            | '?'
            | ';'
            | ':'
            | ')'
            | ']'
            | '}'
            | '”'
            | '’'
    )
}

fn is_horizontal_ws(c: char) -> bool {
    c == ' ' || c == '\t'
}

/// Splits newly arrived text `s` into visual reveal units:
/// - Single CJK characters, kana, hangul, or emoji (with attached punctuation/trailing space)
/// - Contiguous non-CJK words (with attached trailing punctuation/trailing space)
/// Pure whitespace slices yield an empty `Vec`.
pub(super) fn segment_reveal_units(s: &str) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut chars = s.char_indices().peekable();

    while let Some(&(idx, c)) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }

        let unit_start = idx;
        let mut has_visible = false;

        while let Some(&(_, ch)) = chars.peek() {
            if is_opening_punct(ch) {
                has_visible = true;
                chars.next();
            } else {
                break;
            }
        }

        if let Some(&(_, ch)) = chars.peek()
            && !ch.is_whitespace()
        {
            if is_single_unit_char(ch) {
                has_visible = true;
                chars.next();
            } else if is_closing_punct(ch) {
                while let Some(&(_, pc)) = chars.peek() {
                    if is_closing_punct(pc) {
                        has_visible = true;
                        chars.next();
                    } else {
                        break;
                    }
                }
            } else {
                while let Some(&(_, wc)) = chars.peek() {
                    if wc.is_whitespace()
                        || is_single_unit_char(wc)
                        || is_opening_punct(wc)
                        || is_closing_punct(wc)
                    {
                        break;
                    }
                    has_visible = true;
                    chars.next();
                }
            }
        }

        while let Some(&(_, ch)) = chars.peek() {
            if is_closing_punct(ch) {
                has_visible = true;
                chars.next();
            } else {
                break;
            }
        }

        while let Some(&(_, ch)) = chars.peek() {
            if is_horizontal_ws(ch) {
                chars.next();
            } else {
                break;
            }
        }

        let unit_end = chars.peek().map_or(s.len(), |&(i, _)| i);
        if has_visible && unit_end > unit_start {
            out.push(unit_start..unit_end);
        } else if unit_end == unit_start {
            chars.next();
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::segment_reveal_units;

    fn slices(s: &str) -> Vec<&str> {
        segment_reveal_units(s)
            .into_iter()
            .map(|r| &s[r])
            .collect()
    }

    #[test]
    fn segments_chinese_english_mixed_punctuation_and_bpe_chunks() {
        assert_eq!(
            slices("你好呀，cc。有什么想一起做的？"),
            vec![
                "你",
                "好",
                "呀，",
                "cc。",
                "有",
                "什",
                "么",
                "想",
                "一",
                "起",
                "做",
                "的？",
            ]
        );
        assert_eq!(slices("Hello, world! "), vec!["Hello, ", "world! "]);
        assert_eq!(slices("don't worry "), vec!["don't ", "worry "]);
        assert_eq!(slices("（你好）"), vec!["（你", "好）"]);
        assert_eq!(slices("..."), vec!["..."]);
        assert_eq!(slices("   \n\t  "), Vec::<&str>::new());
        assert_eq!(slices("stream"), vec!["stream"]);
        assert_eq!(slices("ing "), vec!["ing "]);
    }
}
