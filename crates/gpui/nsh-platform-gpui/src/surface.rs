//! One surface: an engine thread and the views it mounts on GPUI's main
//! thread.
//!
//! The engine thread hands each commit to [`Sink`], which queues it and
//! sends one wake-up to the main thread ([`crate::rt::invoke`]); the main
//! thread drains the queue and applies every batch in order
//! ([`Surface::drain`]). Drawing is coalesced: changes mark the surface
//! invalid, and [`flush_all`] (which ends every main-thread entry point)
//! notifies the GPUI view showing it, which draws the tree in its next
//! frame ([`crate::render`]). Native animations and JS frame requests tick
//! on the window's frames ([`Surface::tick`], called while drawing) while
//! there is something to do. A window that is not presented (minimized,
//! hidden, a suspended Wayland toplevel) draws no frames, so what changes
//! the app's state when an animation ends (a view detached, a screen's
//! `onDismiss`, an animation's `finished`, a scroll's end) is kept by a
//! timer instead, which ticks at its deadline if no frame did
//! ([`Surface::keep_deadline`]).

use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use crossbeam_channel::{Receiver, Sender};
use nsh_core::{EventKind, EventPayload, MountBatch, MountItem, NodeId, Points, ROOT_ID, Size};
use nsh_engine::{
    AppError, Engine, EngineConfig, EngineControl, Environment, MainThreadTask, MountSink,
};
use rustc_hash::FxHashMap;

use crate::{
    animation::UiAnimation,
    mount::Mounter,
    text::GpuiTextEngine,
    view::{Hosted, NshView},
};

/// What the engine thread sends the main thread.
enum Message {
    Batch(MountBatch, Option<Points>),
    Error(AppError),
    Frame,
    HotUpdateFailed,
    Handled(u64),
    /// A function the app runs on the main thread
    /// ([`MountSink::run_on_main_thread`]), after the batches before it.
    Run(MainThreadTask),
}

thread_local! {
    /// The surfaces of this (main) thread, by key.
    static SURFACES: RefCell<FxHashMap<u64, Weak<Surface>>> = RefCell::default();
}

static NEXT_KEY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Forwards batches from the engine thread to the main thread.
struct Sink {
    /// The surface to wake ([`Surface::by_key`]).
    key: u64,
    tx: Sender<Message>,
    /// A wake-up is on its way: no need for another.
    pending: Arc<AtomicBool>,
}

impl Sink {
    fn send(&self, message: Message) {
        let _ = self.tx.send(message);
        if !self.pending.swap(true, Ordering::AcqRel) {
            let key = self.key;
            crate::rt::invoke(move || {
                if let Some(surface) = Surface::by_key(key) {
                    surface.drain();
                }
            });
        }
    }
}

impl MountSink for Sink {
    fn submit(&self, batch: MountBatch) {
        self.submit_with_content_height(batch, None);
    }

    fn needs_content_height(&self) -> bool {
        true
    }

    fn submit_with_content_height(&self, batch: MountBatch, content_height: Option<Points>) {
        self.send(Message::Batch(batch, content_height));
    }

    fn report(&self, error: &AppError) {
        self.send(Message::Error(error.clone()));
    }

    fn request_frame(&self) {
        self.send(Message::Frame);
    }

    fn hot_update_failed(&self) {
        self.send(Message::HotUpdateFailed);
    }

    fn handled(&self, events: u64) {
        self.send(Message::Handled(events));
    }

    fn run_on_main_thread(&self, task: MainThreadTask) {
        self.send(Message::Run(task));
    }
}

/// How long after a deadline the timer that keeps it fires
/// ([`Surface::keep_deadline`]): more than a frame, so while the window
/// draws, its frames always come first.
const LATE: f64 = 0.05;

/// The rate (Hz) of what is drawn at a paced rate ([`Surface::pace`]) and
/// asks for none of its own: animations looping with no end and JS frame
/// loops running long ([`Surface::js_loop_paced`]).
pub(crate) const PACED_HZ: f32 = 30.0;

/// Animations asking for this rate (Hz) or more are drawn on the display's
/// frames; below it, at their rate ([`Surface::pace`]).
const DISPLAY_HZ: f32 = 60.0;

/// A JS frame loop (`requestAnimationFrame` asked again frame after frame)
/// running this long (seconds) is paced like an endless animation: short
/// bursts (a spring, a JS transition) keep full rate.
const JS_LOOP_PACE_AFTER: f64 = 1.0;

/// A JS frame request coming this long (seconds) after the last run of the
/// callbacks starts a new loop rather than continuing one.
const JS_LOOP_GAP: f64 = 0.25;

/// The surface stayed invalid this long (seconds) without being drawn: it is
/// not presented (another dock tab, a hidden panel) and the pacer stops
/// until it is drawn again.
const NOT_PRESENTED_AFTER: f64 = 0.25;

/// Seconds between paced frames at `hz`, rounded up to the microsecond: the
/// pacer's frames come exactly this far apart ([`Surface::arm_pacer`]), and
/// JS callbacks get their times, so a loop throttling itself to that rate
/// (`time - last >= 1000 / 30`) never sees two of them closer.
pub(crate) fn paced_period(hz: f32) -> f64 {
    (1e6 / f64::from(hz)).ceil() / 1e6
}

/// How a surface's animations want frames ([`Surface::animation_pace`]).
#[derive(Clone, Copy, Debug, PartialEq)]
enum AnimationPace {
    /// None runs.
    Still,
    /// On the display's frames.
    Display,
    /// This many seconds apart ([`paced_period`]).
    Paced(f64),
}

/// Called on every frame with the frame's time (seconds, [`Surface::clock`])
/// until it is done ([`Surface::on_frame`]).
pub(crate) type FrameClient = Box<dyn FnMut(&Rc<Surface>, f64) -> Next>;

/// What a frame client has left after a frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Next {
    /// Nothing: it is not called again.
    Done,
    /// Drawing only (a fade, a shader, a scroll indicator): the next frame,
    /// when there is one.
    Frame,
    /// Something that changes the app's state ends at this time (seconds,
    /// [`Surface::clock`]: a view detached, an event sent): it is called by
    /// then even while no frames are drawn ([`Surface::keep_deadline`]),
    /// and on every frame until then.
    At(f64),
}

impl From<bool> for Next {
    /// Whether it wants the next frame, for drawing only.
    fn from(more: bool) -> Self {
        if more { Next::Frame } else { Next::Done }
    }
}

/// Called with the new size (points) when the surface is resized.
pub(crate) type ResizeHandler = Box<dyn Fn(f32, f32)>;

/// A window shortcut: true if it took the key (down).
pub(crate) type ShortcutHandler = Box<dyn Fn(&crate::input::KeyEvent) -> bool>;

/// Called when the app reports an error (development hosts show it).
pub(crate) type ErrorHandler = Box<dyn Fn(&AppError)>;

/// A handler as kept: shared, so it is called with nothing borrowed (it may
/// restart the app or set another handler).
type Handler<F> = RefCell<Option<Rc<F>>>;

/// One engine and its views.
pub(crate) struct Surface {
    key: u64,
    /// The root view (node [`ROOT_ID`]): fills the window.
    pub(crate) root: NshView,
    /// Logical in-flow root geometry, applied with its mount batch.
    pub(crate) content_height: Cell<Option<Points>>,
    /// Views by node id, and what mounting needs between items.
    pub(crate) mounter: RefCell<Mounter>,
    /// The views by node id (the mounter's, readable while it applies a
    /// batch).
    views: Rc<crate::mount::Views>,
    /// The native animation graph.
    pub(crate) anim: RefCell<UiAnimation>,
    /// Stacks, screens and modals.
    pub(crate) nav: RefCell<crate::navigation::Nav>,
    /// Pointer input state ([`crate::input`]).
    pub(crate) input: crate::input::Input,
    /// The destination of a file dragged in from another application.
    pub(crate) file_drop: crate::file_drop::State,
    engine: RefCell<Option<Engine>>,
    control: RefCell<Option<EngineControl>>,
    rx: RefCell<Option<Receiver<Message>>>,
    pending: RefCell<Arc<AtomicBool>>,
    /// Pixels per point (the window's scale factor).
    scale: Cell<f32>,
    /// The GPUI view drawing the surface ([`crate::app::AppView`]): notified
    /// when the surface changed.
    pub(crate) entity: Cell<Option<gpui::EntityId>>,
    /// The GPUI window that view is in, as it was made or last drawn: what
    /// reaches the window between frames uses it (native components' props
    /// and commands, [`crate::native`]).
    pub(crate) main_window: Cell<Option<gpui::AnyWindowHandle>>,
    /// The GPUI text system (fonts the engine thread shapes with, and the
    /// main thread draws with).
    pub(crate) text_system: RefCell<Option<Arc<gpui::TextSystem>>>,
    /// Where the surface was drawn last, in window coordinates (points):
    /// input arrives in window coordinates.
    pub(crate) origin: Cell<(f32, f32)>,
    /// A draw is requested.
    invalid: Cell<bool>,
    /// Mouse buttons pressed on the surface (bit per button): their moves
    /// and releases are the surface's wherever they happen.
    pub(crate) input_pressed: Cell<u8>,
    /// The mouse is over the surface.
    pub(crate) input_hovered: Cell<bool>,
    /// A field took the surface's focus. The view takes the window's at the
    /// next paint, or the keys for the field go to whatever element of the
    /// embedding app had it ([`crate::events::install`]).
    pub(crate) window_focus_requested: Cell<bool>,
    /// The surface's window is not presented (minimized, on another
    /// workspace, a suspended Wayland toplevel): its caret does not blink
    /// ([`crate::text_input::window_visibility`]). Kept across the app's
    /// reloads, which do not change the window's visibility.
    pub(crate) window_hidden: Cell<bool>,
    /// The environment the engine was last told about.
    pub(crate) environment: Cell<Environment>,
    /// The app's appearance (`nsh.json`, `setColorScheme`): the color
    /// scheme is the system's resolved with it ([`crate::system`]).
    pub(crate) appearance: Cell<nsh_core::platform::Appearance>,
    /// JS asked for a display frame.
    js_frame: Cell<bool>,
    /// When the current unbroken JS frame loop first ran its callbacks.
    js_loop_since: Cell<Option<f64>>,
    /// When the JS frame callbacks last ran.
    js_last_run: Cell<Option<f64>>,
    /// When the surface last turned invalid ([`Surface::invalidate`]).
    invalid_since: Cell<f64>,
    /// The paced frame the pacer's timer waits for ([`Surface::arm_pacer`]).
    pacer_armed: Cell<Option<f64>>,
    /// The last paced frame's time, or the last run of the JS callbacks:
    /// the next paced frame is a whole number of periods after it.
    pacer_last: Cell<Option<f64>>,
    /// A paced frame ran the JS callbacks and is drawn once their results
    /// are in ([`Surface::frame_asked`]), not before.
    pacer_owed: Cell<bool>,
    /// Extra frame clients (transitions, glides, spinners), called on every
    /// frame until they are done, each with its deadline ([`Next::At`]; a
    /// new one's is when it was added, [`Surface::on_frame`]).
    frame_clients: RefCell<Vec<(FrameClient, Option<f64>)>>,
    /// When the last tick ran ([`Surface::clock`]).
    last_tick: Cell<f64>,
    /// A tick runs: what it adds is scheduled when it ends.
    ticking: Cell<bool>,
    /// A batch was applied since the last tick: it may have moved what the
    /// frame clients last reported (a transition retargeted), so a tick is
    /// due at once ([`Surface::deadline`]).
    restep: Cell<bool>,
    /// The deadline the scheduled timer waits for
    /// ([`Surface::keep_deadline`]).
    keeper: Cell<Option<f64>>,
    /// The GPU device shader views render with, opened when the first
    /// shader is mounted (`None` inside: there is none).
    shader_renderer: std::cell::OnceCell<Option<Rc<nsh_engine::shader::Renderer>>>,
    /// The model of a test session's app (`nsh test --target gpui`).
    #[cfg(feature = "dev")]
    test_model: RefCell<Option<std::sync::Arc<nsh_testing::agent::Model>>>,
    /// Frame statistics, while they are logged ([`crate::stats`]).
    pub(crate) stats: RefCell<Option<crate::stats::FrameStats>>,
    /// Sequence watermark of retired [`gpui::CachedLayerId`]s removed from the
    /// main window on paint.
    pub(crate) removed_layers_seq: Cell<u64>,
    /// Cache `(hits, misses)` recorded during the most recent paint pass.
    pub(crate) last_layer_cache_stats: Cell<(u32, u32)>,
    epoch: std::time::Instant,
    on_error: Handler<dyn Fn(&AppError)>,
    /// A dev session's error overlay: hears the errors besides `on_error`.
    #[cfg(feature = "dev")]
    on_dev_error: Handler<dyn Fn(&AppError)>,
    on_hot_update_failed: Handler<dyn Fn()>,
    /// Counts the apps stopped ([`Surface::stop`]): what an engine sent
    /// before is not applied to the next one.
    generation: Cell<u64>,
    /// Where `asset:/` image URIs resolve ([`crate::image::set_asset_base`]).
    pub(crate) asset_base: RefCell<Option<crate::image::AssetBase>>,
    /// The disk cache of downloaded images ([`crate::image::set_disk_cache`]).
    pub(crate) image_cache: RefCell<Option<nsh_core::image::DiskCache>>,
    /// The roots the app's `<Image>` files are read under
    /// ([`nsh_engine::Engine::media_files`]).
    pub(crate) media_files: RefCell<Arc<nsh_core::access::MediaFiles>>,
    /// What the running app may reach (checked for image redirects).
    pub(crate) access: RefCell<Arc<nsh_core::access::Access>>,
    /// Videos, web views and registered native components.
    pub(crate) native: crate::native::NativeState,
    /// The app's keyboard shortcuts ([`crate::keymap`]).
    pub(crate) keymap: crate::keymap::KeymapState,
    /// Hosted items drawn by the last frame ([`Hosted::placed`]).
    placed: RefCell<Vec<Rc<dyn Hosted>>>,
    /// The window's chrome: title bar, menus ([`crate::chrome`]).
    pub(crate) chrome: crate::chrome::Chrome,
    /// Called after the surface is resized (development overlays).
    pub(crate) on_resized: RefCell<Vec<ResizeHandler>>,
    /// Keys the window takes before anything focused sees them (the
    /// standalone app's full screen, the dev host's reload).
    pub(crate) shortcuts: RefCell<Vec<ShortcutHandler>>,
    /// How many batches were applied.
    pub(crate) batches: Cell<u64>,
    /// Events sent to the engine, and how many of them it has handled
    /// with what they changed applied here.
    events_sent: Cell<u64>,
    events_handled: Cell<u64>,
    /// The embedding app's theme ([`crate::AppView::set_theme`]): sent to
    /// every engine the surface starts, before its first frame.
    pub(crate) host_theme: RefCell<Option<nsh_base::Theme>>,
    /// The embedding app's theme overrides
    /// ([`crate::AppView::set_theme_overrides`]): over the host's theme if
    /// it gave one, else over the app's own.
    pub(crate) host_overrides: RefCell<Option<nsh_engine::Tokens>>,
    /// The window is see-through where the app draws nothing
    /// ([`crate::AppOptions::transparent`]): no window background.
    pub(crate) transparent: Cell<bool>,
    /// The embedding app's appearance (dark or not), standing in for the
    /// window's ([`crate::AppView::set_appearance`]).
    pub(crate) host_dark: Cell<Option<bool>>,
    /// The embedding app's locale and text size, standing in for the
    /// system's ([`crate::AppView::set_locale`], `set_font_scale`).
    pub(crate) host_environment: Cell<nsh_core::platform::HostEnvironment>,
    /// The app's windows of its own besides the main one, oldest first
    /// ([`crate::window`]).
    pub(crate) windows: RefCell<Vec<Rc<crate::window::AppWindow>>>,
    /// While one of those windows handles input, its root: the views input
    /// picks are in it ([`Surface::input_root`], [`crate::window::enter`]).
    pub(crate) input_root: RefCell<Option<NshView>>,
}

impl Surface {
    /// A surface with no app and no window yet.
    pub(crate) fn new(environment: Environment) -> Rc<Surface> {
        let key = NEXT_KEY.fetch_add(1, Ordering::Relaxed);
        let root = NshView::new(ROOT_ID, nsh_core::ViewKind::View);
        let mounter = Mounter::new(root.clone());
        let surface = Rc::new(Surface {
            key,
            root: root.clone(),
            content_height: Cell::new(None),
            views: mounter.views.clone(),
            mounter: RefCell::new(mounter),
            anim: RefCell::new(UiAnimation::default()),
            nav: RefCell::default(),
            input: Default::default(),
            file_drop: Default::default(),
            engine: RefCell::new(None),
            control: RefCell::new(None),
            rx: RefCell::new(None),
            pending: RefCell::new(Arc::new(AtomicBool::new(false))),
            scale: Cell::new(1.0),
            entity: Cell::new(None),
            main_window: Cell::new(None),
            text_system: RefCell::new(None),
            origin: Cell::new((0.0, 0.0)),
            invalid: Cell::new(false),
            input_pressed: Cell::new(0),
            input_hovered: Cell::new(false),
            window_focus_requested: Cell::new(false),
            window_hidden: Cell::new(false),
            environment: Cell::new(environment),
            appearance: Cell::default(),
            js_frame: Cell::new(false),
            js_loop_since: Cell::new(None),
            js_last_run: Cell::new(None),
            invalid_since: Cell::new(0.0),
            pacer_armed: Cell::new(None),
            pacer_last: Cell::new(None),
            pacer_owed: Cell::new(false),
            frame_clients: RefCell::default(),
            last_tick: Cell::new(f64::NEG_INFINITY),
            ticking: Cell::new(false),
            restep: Cell::new(false),
            keeper: Cell::new(None),
            shader_renderer: std::cell::OnceCell::new(),
            #[cfg(feature = "dev")]
            test_model: RefCell::default(),
            stats: RefCell::new(None),
            removed_layers_seq: Cell::new(0),
            last_layer_cache_stats: Cell::new((0, 0)),
            epoch: std::time::Instant::now(),
            on_error: RefCell::new(None),
            #[cfg(feature = "dev")]
            on_dev_error: RefCell::new(None),
            on_hot_update_failed: RefCell::new(None),
            generation: Cell::new(0),
            asset_base: RefCell::new(None),
            image_cache: RefCell::new(None),
            media_files: RefCell::default(),
            access: RefCell::new(Arc::default()),
            native: Default::default(),
            keymap: Default::default(),
            placed: RefCell::default(),
            on_resized: RefCell::default(),
            shortcuts: RefCell::default(),
            batches: Cell::new(0),
            events_sent: Cell::new(0),
            events_handled: Cell::new(0),
            chrome: Default::default(),
            host_theme: RefCell::new(None),
            host_overrides: RefCell::new(None),
            transparent: Cell::new(false),
            host_dark: Cell::new(None),
            host_environment: Cell::default(),
            windows: RefCell::default(),
            input_root: RefCell::default(),
        });
        SURFACES.with(|s| s.borrow_mut().insert(key, Rc::downgrade(&surface)));
        let weak = Rc::downgrade(&surface);
        root.set_invalidate(Rc::new(move || {
            if let Some(surface) = weak.upgrade() {
                surface.invalidate();
            }
        }));
        surface
    }

    /// The surface with `key`, if it is alive.
    pub(crate) fn by_key(key: u64) -> Option<Rc<Surface>> {
        SURFACES.with(|s| s.borrow().get(&key).and_then(Weak::upgrade))
    }

    /// Its key ([`Surface::by_key`]).
    pub(crate) fn key(&self) -> u64 {
        self.key
    }

    /// Pixels per point (the window's scale factor).
    pub(crate) fn scale(&self) -> f32 {
        self.scale.get()
    }

    /// The GPU device shader views render with ([`crate::shader`]), opened
    /// the first time one asks; without one the app hears that shader views
    /// draw nothing (`shadersAvailable()`).
    pub(crate) fn shader_renderer(&self) -> Option<Rc<nsh_engine::shader::Renderer>> {
        self.shader_renderer
            .get_or_init(|| {
                let renderer =
                    nsh_engine::shader::Renderer::new(nsh_engine::shader::PixelFormat::Bgra);
                if renderer.is_none() {
                    self.control(EngineControl::shaders_unavailable);
                }
                renderer.map(Rc::new)
            })
            .clone()
    }

    /// The root input picks views in: a window's of the app's own while it
    /// handles input ([`crate::window::enter`]), else the main one.
    pub(crate) fn input_root(&self) -> NshView {
        self.input_root
            .borrow()
            .clone()
            .unwrap_or_else(|| self.root.clone())
    }

    /// The surface's size in points.
    pub(crate) fn size(&self) -> (f32, f32) {
        let s = self.root.state();
        (s.frame.size.width.0, s.frame.size.height.0)
    }

    /// Seconds since the surface was made (the animation clock).
    pub(crate) fn clock(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }

    /// Starts `config`'s app, replacing what ran.
    pub(crate) fn start(self: &Rc<Self>, mut config: EngineConfig) {
        self.stop();
        *self.access.borrow_mut() = Arc::new(match &config.app {
            nsh_engine::App::Js(bundle) => bundle.policy.access(),
            nsh_engine::App::Rust(_) => nsh_engine::Policy::default().access(),
        });
        let env = self.environment.get();
        config.viewport = Size::new(env.window.width, env.window.height);
        config.environment = env;
        // A test session's app: the test's storage, network and trace.
        #[cfg(feature = "dev")]
        if let Some(model) = &*self.test_model.borrow() {
            model.set_up_engine(&mut config);
        }
        let (tx, rx) = crossbeam_channel::unbounded();
        let pending = Arc::new(AtomicBool::new(false));
        let sink = Sink {
            key: self.key,
            tx,
            pending: pending.clone(),
        };
        *self.rx.borrow_mut() = Some(rx);
        *self.pending.borrow_mut() = pending;
        let Some(text_system) = self.text_system.borrow().clone() else {
            log::error!("cannot start the engine: the surface is not in a GPUI app yet");
            return;
        };
        let family: Arc<str> = Arc::from(&*crate::text::ui_family_for(&text_system));
        let tokens = self.host_tokens();
        match Engine::start_with(
            config,
            move || GpuiTextEngine::new(text_system, family),
            sink,
            |control| {
                if tokens.is_some() {
                    control.set_tokens(tokens);
                }
            },
        ) {
            Ok(engine) => {
                // The app's downloaded images go through its disk cache.
                crate::image::set_disk_cache(self, engine.image_cache());
                *self.media_files.borrow_mut() = engine.media_files();
                // The test moves its clock through the engine.
                #[cfg(feature = "dev")]
                if let Some(model) = &*self.test_model.borrow() {
                    model.engine_started(engine.control());
                }
                if self.stats.borrow().is_some() {
                    engine.control().set_perf_monitor(Some(Default::default()));
                }
                *self.control.borrow_mut() = Some(engine.control());
                *self.engine.borrow_mut() = Some(engine);
            }
            Err(err) => log::error!("cannot start the engine: {err}"),
        }
    }

    /// Stops the app and removes its views.
    pub(crate) fn stop(&self) {
        self.generation.set(self.generation.get() + 1);
        // Dropping the engine joins its thread.
        self.control.borrow_mut().take();
        self.engine.borrow_mut().take();
        self.rx.borrow_mut().take();
        crate::window::close_all(self);
        self.mounter.borrow_mut().clear();
        self.content_height.set(None);
        self.native.clear();
        crate::system::clear(self);
        crate::key_focus::forget(self);
        *self.anim.borrow_mut() = UiAnimation::default();
        self.nav.borrow_mut().reset();
        crate::input::reset(self);
        crate::keymap::clear(self);
        crate::text_input::surface_stopped(self.key);
        self.js_frame.set(false);
        self.js_loop_since.set(None);
        self.js_last_run.set(None);
        // The timer waiting finds another generation.
        self.pacer_armed.set(None);
        self.pacer_last.set(None);
        self.pacer_owed.set(false);
        self.frame_clients.borrow_mut().clear();
        self.restep.set(false);
        // The timer waiting finds it no longer kept.
        self.keeper.set(None);
        self.events_sent.set(0);
        self.events_handled.set(0);
        self.invalidate();
    }

    /// Calls `f` with the engine's control, if an app runs.
    /// Logs frame statistics once a second, or stops
    /// ([`crate::AppView::set_fps_monitor`]). The engine measures its
    /// commits meanwhile, which `fpsEnabled()` follows.
    pub(crate) fn set_stats(&self, on: bool) {
        *self.stats.borrow_mut() = on.then(Default::default);
        self.control(|c| c.set_perf_monitor(on.then(Default::default)));
    }

    /// The tokens the embedding app gives the engine: its overrides over
    /// its theme.
    pub(crate) fn host_tokens(&self) -> Option<nsh_engine::Tokens> {
        let theme = self.host_theme.borrow();
        let overrides = self.host_overrides.borrow();
        match (&*theme, &*overrides) {
            (Some(theme), Some(overrides)) => Some(overrides.over(theme.tokens())),
            (Some(theme), None) => Some(theme.tokens().clone()),
            (None, overrides) => overrides.clone(),
        }
    }

    /// Sends [`Surface::host_tokens`] to the engine.
    pub(crate) fn host_tokens_changed(&self) {
        let tokens = self.host_tokens();
        self.control(|c| c.set_tokens(tokens));
    }

    pub(crate) fn control(&self, f: impl FnOnce(&EngineControl)) {
        if let Some(control) = &*self.control.borrow() {
            f(control);
        }
    }

    /// Sends an event to JS.
    pub(crate) fn event(&self, id: NodeId, kind: EventKind, payload: EventPayload) {
        self.control(|c| {
            c.dispatch_event(id, kind, payload);
            self.events_sent.set(self.events_sent.get() + 1);
        });
    }

    /// Events sent to the engine so far.
    pub(crate) fn events_sent(&self) -> u64 {
        self.events_sent.get()
    }

    /// Events the engine has handled, with what they changed applied.
    pub(crate) fn events_handled(&self) -> u64 {
        self.events_handled.get()
    }

    /// The view of node `id`.
    pub(crate) fn view(&self, id: NodeId) -> Option<NshView> {
        self.views.get(id)
    }

    /// Shows fatal app errors with `f` (development hosts, embedding apps).
    pub(crate) fn set_error_handler(&self, f: ErrorHandler) {
        *self.on_error.borrow_mut() = Some(Rc::from(f));
    }

    /// Shows fatal app errors with `f` too, besides the handler of
    /// [`Surface::set_error_handler`] (a dev session's overlay).
    #[cfg(feature = "dev")]
    pub(crate) fn set_dev_error_handler(&self, f: ErrorHandler) {
        *self.on_dev_error.borrow_mut() = Some(Rc::from(f));
    }

    /// Runs `f` when a hot update could not be applied in place.
    #[cfg(feature = "dev")]
    pub(crate) fn set_hot_update_failed(&self, f: impl Fn() + 'static) {
        *self.on_hot_update_failed.borrow_mut() = Some(Rc::new(f));
    }

    /// Updates the environment and tells the engine, with the embedding
    /// app's settings over the system's.
    pub(crate) fn set_environment(&self, mut environment: Environment) {
        self.host_environment.get().apply(&mut environment);
        if self.environment.replace(environment) != environment {
            self.control(|c| c.set_environment(environment));
            crate::system::environment_applied(self);
        }
    }

    /// The surface is drawn `width` × `height` points big at `scale`
    /// pixels per point.
    pub(crate) fn resized(self: &Rc<Self>, width: f32, height: f32, scale: f32) {
        let scale_changed = (self.scale.replace(scale) - scale).abs() > f32::EPSILON;
        {
            let mut s = self.root.state_mut();
            if !scale_changed && s.frame.size.width.0 == width && s.frame.size.height.0 == height {
                return;
            }
            s.frame.size.width.0 = width;
            s.frame.size.height.0 = height;
        }
        self.root.bump_revision();
        let mut env = self.environment.get();
        env.window = nsh_core::Size::new(nsh_core::Points(width), nsh_core::Points(height));
        env.pixel_ratio = scale;
        self.environment.set(env);
        self.control(|c| {
            c.resize(Size::new(Points(width), Points(height)));
            c.set_environment(env);
        });
        crate::navigation::surface_resized(self, width, height);
        crate::system::surface_resized(self, width, height);
        for f in self.on_resized.borrow().iter() {
            f(width, height);
        }
        self.invalidate();
    }

    /// Asks for the surface to be drawn again (once, however often asked
    /// before it is): [`flush_all`] notifies its view.
    pub(crate) fn invalidate(&self) {
        if !self.invalid.replace(true) {
            self.invalid_since.set(self.clock());
            // Entry points flush when they end; a change made anywhere else
            // (a GPUI callback, an observer) is flushed by the main-thread
            // queue, which flushes after its jobs: GPUI's retained frames
            // never show a stale tree.
            crate::rt::invoke(|| {});
        }
    }

    /// It changed since it was last drawn.
    pub(crate) fn invalidated(&self) -> bool {
        self.invalid.get()
    }

    /// The surface is being drawn: what changed so far is in this frame.
    pub(crate) fn drawing(&self) {
        self.invalid.set(false);
    }

    /// Hosted items drawn by the frame just drawn, with their window
    /// bounds: tells them, and the ones no longer drawn.
    pub(crate) fn items_placed(&self, drawn: Vec<(Rc<dyn Hosted>, gpui::Bounds<gpui::Pixels>)>) {
        place_items(&self.placed, drawn);
    }
}

/// Hosted items drawn by a window's frame just drawn, with their window
/// bounds: tells them, and those of `placed` (the window's last frame's)
/// no longer drawn.
pub(crate) fn place_items(
    placed: &RefCell<Vec<Rc<dyn Hosted>>>,
    drawn: Vec<(Rc<dyn Hosted>, gpui::Bounds<gpui::Pixels>)>,
) {
    let mut placed = placed.borrow_mut();
    for old in placed.iter() {
        if !drawn.iter().any(|(item, _)| Rc::ptr_eq(item, old)) {
            old.placed(None);
        }
    }
    placed.clear();
    for (item, rect) in drawn {
        item.placed(Some(rect));
        placed.push(item);
    }
}

impl Surface {
    /// Applies what the engine sent.
    pub(crate) fn drain(self: &Rc<Self>) {
        self.pending.borrow().store(false, Ordering::Release);
        let messages: Vec<Message> = match &*self.rx.borrow() {
            Some(rx) => rx.try_iter().collect(),
            None => return,
        };
        let generation = self.generation.get();
        let mut applied = false;
        for message in messages {
            // A handler restarted or stopped the app: the rest was the old
            // engine's.
            if self.generation.get() != generation {
                return;
            }
            match message {
                Message::Batch(batch, content_height) => {
                    self.apply(&batch);
                    if self.content_height.replace(content_height) != content_height {
                        self.invalidate();
                    }
                    applied = true;
                    self.batches.set(self.batches.get() + 1);
                }
                Message::Error(error) => {
                    log::error!("{error}");
                    #[cfg(feature = "dev")]
                    if let Some(model) = &*self.test_model.borrow() {
                        model.error(error.to_string());
                    }
                    #[cfg(feature = "dev")]
                    {
                        let handler = self.on_dev_error.borrow().clone();
                        if let Some(f) = handler {
                            f(&error);
                        }
                    }
                    let handler = self.on_error.borrow().clone();
                    if let Some(f) = handler {
                        f(&error);
                    }
                }
                Message::Frame => self.frame_asked(),
                Message::HotUpdateFailed => {
                    let handler = self.on_hot_update_failed.borrow().clone();
                    if let Some(f) = handler {
                        f();
                    }
                }
                Message::Handled(events) => {
                    self.events_handled.set(events);
                    crate::text_input::engine_caught_up(self);
                }
                Message::Run(task) => task.run(),
            }
        }
        if applied {
            self.write_animations();
            // Views may have come under (or gone from under) a still mouse.
            crate::input::rehover(self);
            if !self.frame_clients.borrow().is_empty() {
                self.restep.set(true);
            }
        }
        self.ensure_ticking();
    }

    fn apply(self: &Rc<Self>, batch: &MountBatch) {
        crate::access::prepare(batch);
        for item in &batch.items {
            self.anim.borrow_mut().apply(item);
            if !matches!(
                item,
                MountItem::Animation(_) | MountItem::AnimationBindings { .. }
            ) {
                self.mounter.borrow_mut().apply(self, item);
            }
        }
        self.mounter.borrow_mut().end_batch(self);
        #[cfg(feature = "dev")]
        if let Some(model) = &*self.test_model.borrow() {
            model.apply(batch);
        }
        crate::access::batch_applied(self);
        crate::selectable_view::batch_applied(self);
    }

    /// Writes the animated values to the views and reports finished
    /// animations.
    pub(crate) fn write_animations(&self) {
        let (finished, moved) = {
            let mut anim = self.anim.borrow_mut();
            anim.write(&self.mounter.borrow());
            let mut moved = Vec::new();
            anim.write_components(
                |id| crate::native::has_component(self, id),
                |id, name, value| moved.push((id, std::sync::Arc::<str>::from(name), value)),
            );
            (std::mem::take(&mut anim.finished), moved)
        };
        // No surface state borrowed while the app's components run.
        crate::native::component_animated(self, moved);
        crate::scroll::pin_animated_sticky(self);
        if !finished.is_empty() {
            self.control(|c| c.animations_finished(finished));
        }
    }

    /// Calls `f` on every frame (with the frame's time) until it is done
    /// ([`Next`]): transitions, scroll glides, navigation; the first time
    /// by now even while no frames are drawn.
    pub(crate) fn on_frame(self: &Rc<Self>, f: impl FnMut(&Rc<Surface>, f64) -> Next + 'static) {
        let now = self.clock();
        self.frame_clients
            .borrow_mut()
            .push((Box::new(f), Some(now)));
        self.ensure_ticking();
    }

    /// Calls `f` on every frame while it returns true, for what only draws
    /// (a fade, a field waiting to be shown): nothing it does comes due
    /// without frames ([`Next::Frame`]).
    pub(crate) fn on_draw(self: &Rc<Self>, mut f: impl FnMut(&Rc<Surface>, f64) -> bool + 'static) {
        self.frame_clients
            .borrow_mut()
            .push((Box::new(move |surface, now| f(surface, now).into()), None));
        self.ensure_ticking();
    }

    /// [`Surface::on_draw`] while the animation graph is borrowed (its
    /// writes into the views): asks for a frame without looking at it.
    pub(crate) fn add_frame_client(&self, mut f: impl FnMut(&Rc<Surface>, f64) -> bool + 'static) {
        self.frame_clients
            .borrow_mut()
            .push((Box::new(move |surface, now| f(surface, now).into()), None));
        self.invalidate();
    }

    /// Whether a native animation runs: the graph's (not one looping
    /// forever, which no test step waits for), or the platform's own
    /// (scroll glides).
    #[cfg(feature = "dev")]
    pub(crate) fn animating(&self) -> bool {
        self.anim.borrow().finishing() || !self.frame_clients.borrow().is_empty()
    }

    /// Keeps the model `nsh test --target` steps see of the app
    /// ([`nsh_testing::agent::Model`]): every batch applied and every error
    /// go to it from now on.
    #[cfg(feature = "dev")]
    pub(crate) fn set_test_model(&self, model: Option<std::sync::Arc<nsh_testing::agent::Model>>) {
        *self.test_model.borrow_mut() = model;
    }

    /// Something animates or JS wants a frame: the view keeps drawing
    /// frames ([`crate::render`]).
    pub(crate) fn wants_frames(&self) -> bool {
        self.js_frame.get()
            || self.anim.borrow().animating()
            || !self.frame_clients.borrow().is_empty()
    }

    /// The next frame is to come on the display's frames: a JS frame not
    /// paced ([`Surface::js_loop_paced`]), a frame client, or an animation
    /// drawn at the display's rate ([`Surface::animation_pace`]). Otherwise
    /// what wants frames gets them from the pacer ([`Surface::pace`]).
    pub(crate) fn wants_immediate_frame(&self) -> bool {
        (self.js_frame.get() && !self.js_loop_paced())
            || !self.frame_clients.borrow().is_empty()
            || self.animation_pace() == AnimationPace::Display
    }

    /// How the running animations want frames: one that ends by itself and
    /// asks for no rate, or one asking for [`DISPLAY_HZ`] or more, on the
    /// display's frames; one asking for less at its rate; the rest (looping
    /// with no end) at [`PACED_HZ`]. With several, the highest rate any asks
    /// for ([`nsh_core::animation::Graph::frame_rate`]).
    fn animation_pace(&self) -> AnimationPace {
        let anim = self.anim.borrow();
        if !anim.animating() {
            return AnimationPace::Still;
        }
        match anim.frame_rate() {
            Some(hz) if hz < DISPLAY_HZ => AnimationPace::Paced(paced_period(hz)),
            Some(_) => AnimationPace::Display,
            None if anim.finishing() => AnimationPace::Display,
            None => AnimationPace::Paced(paced_period(PACED_HZ)),
        }
    }

    /// Seconds between the paced frames wanted now, or none: nothing paced
    /// runs, frames come on the display's anyway, or the window is hidden.
    /// A paced JS loop waiting for its callbacks and paced animations share
    /// the frames, at the higher rate either asks for.
    fn pace(&self) -> Option<f64> {
        if self.window_hidden.get() || self.wants_immediate_frame() {
            return None;
        }
        let js = (self.js_frame.get() && self.js_loop_paced()).then(|| paced_period(PACED_HZ));
        let anim = match self.animation_pace() {
            AnimationPace::Paced(period) => Some(period),
            _ => None,
        };
        js.into_iter().chain(anim).reduce(f64::min)
    }

    /// Draws frames while animations run or JS wants a frame, and keeps
    /// their deadlines while no frames come. What is paced is drawn by the
    /// pacer only ([`Surface::arm_pacer`]): what the engine sends while it
    /// runs draws a frame only if it changed something drawn.
    pub(crate) fn ensure_ticking(self: &Rc<Self>) {
        let immediate = self.wants_immediate_frame();
        if immediate {
            self.invalidate();
        }
        if immediate || self.anim.borrow().animating() {
            self.keep_deadline();
        }
        self.arm_pacer();
    }

    /// A frame of the surface was drawn: true when the next is to come on
    /// the display's frames, else the pacer is armed for what is paced.
    pub(crate) fn frame_drawn(self: &Rc<Self>) -> bool {
        // The paced frame owed is this one.
        self.pacer_owed.set(false);
        if self.wants_immediate_frame() || self.invalidated() {
            return true;
        }
        self.arm_pacer();
        false
    }

    /// Arms the pacer's timer for the next paced frame, if any is wanted
    /// ([`Surface::pace`]) and none is armed. Paced frames keep a cadence:
    /// the next comes a whole number of periods after the last (or after
    /// the JS callbacks last ran, on a frame), whenever the timer fires, so
    /// timer delays add up to nothing and the JS callbacks see their frames
    /// a period apart.
    pub(crate) fn arm_pacer(self: &Rc<Self>) {
        if self.pacer_armed.get().is_some() {
            return;
        }
        let Some(period) = self.pace() else {
            return;
        };
        let now = self.clock();
        let at = match self.pacer_last.get() {
            Some(last) if last <= now => last + period * (((now - last) / period).floor() + 1.0),
            Some(last) => last + period,
            None => now + period,
        };
        self.pacer_armed.set(Some(at));
        let key = self.key;
        let generation = self.generation.get();
        crate::rt::after_ms(((at - now) * 1000.0).ceil() as u64, move || {
            let Some(surface) = Surface::by_key(key) else {
                return;
            };
            if surface.generation.get() == generation && surface.pacer_armed.get() == Some(at) {
                surface.paced_frame(at);
            }
        });
    }

    /// The paced frame of time `at` came due. A paced JS loop waiting runs
    /// its callbacks with `at` as their time, if the surface is presented
    /// (what the last run changed was drawn); as they run in the engine,
    /// what they change comes after, so the frame waits for it
    /// ([`Surface::frame_asked`]) and shows the animations' step with it:
    /// one frame where drawing first would make two. The JS loop draws no
    /// frame of its own, so the pacer arms itself for it; paced animations
    /// alone are drawn, and the frame drawn arms it ([`Surface::frame_drawn`]),
    /// so a surface no longer drawn stops it.
    fn paced_frame(self: &Rc<Self>, at: f64) {
        self.pacer_armed.set(None);
        self.pacer_last.set(Some(at));
        // The last one's canvases have not all landed in a whole period
        // ([`Surface::frame_asked`]), or the loop ended asking for nothing:
        // the animations' step waits no longer.
        if self.pacer_owed.replace(false) {
            self.draw_paced();
        }
        if self.wants_immediate_frame() {
            // A button went down: frames at the display's rate.
            self.ensure_ticking();
            return;
        }
        let animated = matches!(self.animation_pace(), AnimationPace::Paced(_));
        if self.js_frame.get() && self.js_loop_paced() && self.presented(self.clock()) {
            self.run_js_frame(at);
            self.pacer_owed.set(animated);
            self.arm_pacer();
        } else if animated {
            self.draw_paced();
        }
    }

    /// The engine asks for a frame ([`MountSink::request_frame`]): JS
    /// called `requestAnimationFrame`. Sent at the end of the engine's
    /// turn, after its batch: the JS callbacks a paced frame ran are done
    /// and what they changed is applied, so the frame owed is drawn now,
    /// unless canvases they drew again are being rasterized, whose landing
    /// draws it ([`crate::canvas`]).
    fn frame_asked(self: &Rc<Self>) {
        // Asked long after the last run: a new loop, not the last one
        // going on.
        let now = self.clock();
        if self
            .js_last_run
            .get()
            .is_none_or(|last| now - last > JS_LOOP_GAP)
        {
            self.js_loop_since.set(None);
        }
        self.js_frame.set(true);
        if self.pacer_owed.get() && !crate::canvas::rasterizing(self.key) {
            self.pacer_owed.set(false);
            self.draw_paced();
        }
    }

    /// Draws a paced frame: the surface and its windows of their own.
    fn draw_paced(&self) {
        self.invalidate();
        crate::window::invalidate_all(self);
    }

    /// Whether the surface is presented: its window shows, and it was drawn
    /// since it last changed, or not long ago ([`NOT_PRESENTED_AFTER`]).
    fn presented(&self, now: f64) -> bool {
        !self.window_hidden.get()
            && !(self.invalid.get() && now - self.invalid_since.get() > NOT_PRESENTED_AFTER)
    }

    /// The JS frame loop running now has run long enough to be paced, and
    /// no mouse button is held on the surface (a drag follows the pointer
    /// at full rate). Browsers run `requestAnimationFrame` at the display's
    /// rate but draw nothing for a callback that changed nothing; here the
    /// callbacks run in the engine after the frame that asked for them, so
    /// that frame never shows what they do: a loop at full rate is a whole
    /// window drawn at full rate, whatever the loop throttles itself to.
    fn js_loop_paced(&self) -> bool {
        self.input_pressed.get() == 0
            && self
                .js_loop_since
                .get()
                .is_some_and(|since| self.clock() - since >= JS_LOOP_PACE_AFTER)
    }

    /// Runs the JS frame callbacks, with `at` as their time (on a frame, its
    /// time; on a paced frame, the time it was paced by).
    fn run_js_frame(&self, at: f64) {
        self.js_frame.set(false);
        if self.js_loop_since.get().is_none() {
            self.js_loop_since.set(Some(at));
        }
        self.js_last_run.set(Some(at));
        self.pacer_last.set(Some(at));
        let time = self.epoch + std::time::Duration::from_secs_f64(at);
        self.control(|c| c.frame_at(time));
    }

    /// One animation frame (called while the surface is drawn, before its
    /// views are, or by [`Surface::keep_deadline`] when no frame came).
    pub(crate) fn tick(self: &Rc<Self>) {
        let now = self.clock();
        self.last_tick.set(now);
        self.restep.set(false);
        self.ticking.set(true);
        if self.anim.borrow().animating() {
            self.anim.borrow_mut().tick(now);
            self.write_animations();
        }
        // A paced loop's callbacks run on paced frames ([`Surface::paced_frame`]).
        if self.js_frame.get() && !self.js_loop_paced() {
            self.run_js_frame(now);
        }
        let clients = std::mem::take(&mut *self.frame_clients.borrow_mut());
        let mut kept: Vec<(FrameClient, Option<f64>)> = clients
            .into_iter()
            .filter_map(|(mut f, _)| match f(self, now) {
                Next::Done => None,
                Next::Frame => Some((f, None)),
                Next::At(at) => Some((f, Some(at))),
            })
            .collect();
        // Clients added while running stay after the ones kept.
        let mut added = std::mem::take(&mut *self.frame_clients.borrow_mut());
        kept.append(&mut added);
        *self.frame_clients.borrow_mut() = kept;
        self.ticking.set(false);
        self.keep_deadline();
    }

    /// When the next thing that changes the app's state comes due without
    /// a frame (seconds, [`Surface::clock`]): an animation of the graph or
    /// a frame client ending, or `now` after a batch moved what the frame
    /// clients reported. `None` when only drawing is left: an animation
    /// looping forever, a fade, a JS frame request. Batches are applied as
    /// they arrive ([`Surface::drain`]), never held for a frame.
    fn deadline(&self, now: f64) -> Option<f64> {
        let restep = self.restep.get().then_some(now);
        let clients = self
            .frame_clients
            .borrow()
            .iter()
            .filter_map(|(_, at)| *at)
            .reduce(f64::min);
        [self.anim.borrow().deadline(now), clients, restep]
            .into_iter()
            .flatten()
            .reduce(f64::min)
    }

    /// Makes what comes due happen on time while no frames are drawn: a
    /// window that is not presented (minimized, hidden, a suspended Wayland
    /// toplevel) or a view no longer rendered gets no frames, and
    /// animations must still end (a transition detaches its views when it
    /// does, a modal reports its dismissal, an animation its end to JS). A
    /// timer on the main thread ([`crate::rt::after_ms`]) waits for the
    /// next deadline ([`Surface::deadline`]) and a little more ([`LATE`]);
    /// if no tick ran since it came due, it ticks outside the paint, which
    /// steps everything to the time, ends what ended and gives the next
    /// deadline. Nothing in a tick needs GPUI's window: what it changes
    /// only marks the surface invalid, and the window draws it when it
    /// shows again. While frames come they always tick first and the timer
    /// only schedules the next; nothing is scheduled for what only draws
    /// (a spinner, a JS frame request), so a hidden window with an endless
    /// animation does no work.
    fn keep_deadline(self: &Rc<Self>) {
        // The tick schedules when it ends, with every client's deadline.
        if self.ticking.get() {
            return;
        }
        let now = self.clock();
        let Some(at) = self.deadline(now) else {
            return;
        };
        // One timer at a time: the one waiting for an earlier deadline
        // schedules the next when it fires.
        if self.keeper.get().is_some_and(|pending| pending <= at) {
            return;
        }
        self.keeper.set(Some(at));
        let key = self.key;
        let ms = ((at + LATE - now).max(0.0) * 1000.0).ceil() as u64;
        crate::rt::after_ms(ms, move || {
            let Some(surface) = Surface::by_key(key) else {
                return;
            };
            // An earlier deadline (or a restart) took its place.
            if surface.keeper.get() != Some(at) {
                return;
            }
            surface.keeper.set(None);
            if surface.last_tick.get() < at && surface.wants_frames() {
                surface.tick();
            } else {
                surface.keep_deadline();
            }
        });
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        let key = self.key;
        let _ = SURFACES.try_with(|s| s.borrow_mut().remove(&key));
    }
}

/// Notifies the views of this thread's surfaces that changed, and those
/// of their windows of their own.
pub(crate) fn flush_all(cx: &mut gpui::App) {
    let ids: Vec<gpui::EntityId> = SURFACES.with(|s| {
        let mut ids = Vec::new();
        for surface in s.borrow().values().filter_map(Weak::upgrade) {
            crate::window::changed_windows(&surface, &mut ids);
            if !surface.invalid.get() {
                continue;
            }
            if let Some(stats) = surface.stats.borrow_mut().as_mut() {
                stats.notified();
            }
            ids.extend(surface.entity.get());
        }
        ids
    });
    for id in ids {
        cx.notify(id);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Runs with the other window tests so all sinks share their GPUI queue.
    #[cfg(feature = "shot")]
    pub(crate) fn committed_content_height_reaches_host_observers(cx: &mut gpui::TestAppContext) {
        use gpui::AppContext as _;
        use nsh_core::{ApproxTextEngine, Core, Op, Prop, Tag};

        let (view, cx) = cx.add_window_view(|window, cx| {
            crate::AppView::new(crate::AppOptions::default(), window, cx)
        });
        cx.run_until_parked();
        let surface = view.read_with(cx, |view, _| view.surface().clone());
        let observed = cx.new(|cx| {
            cx.observe(&view, |seen: &mut Vec<_>, view, cx| {
                let view = view.read(cx);
                let mounted = view.surface().root.child_views().first().map(|child| {
                    let frame = child.state().frame;
                    Points(frame.origin.y.0 + frame.size.height.0)
                });
                seen.push((view.content_height(), mounted));
            })
            .detach();
            Vec::new()
        });
        let (tx, rx) = crossbeam_channel::unbounded();
        *surface.rx.borrow_mut() = Some(rx);
        let sink = Sink {
            key: surface.key,
            tx,
            pending: surface.pending.borrow().clone(),
        };
        let mut core = Core::new(
            ApproxTextEngine,
            nsh_core::geometry::size(Points(320.0), Points(480.0)),
        );
        core.apply(Op::Create {
            id: 2,
            tag: Tag::View,
        })
        .unwrap();
        core.apply(Op::Insert {
            parent: ROOT_ID,
            child: 2,
            before: None,
        })
        .unwrap();
        for height in [100, 160] {
            core.apply(Op::SetProp {
                id: 2,
                prop: Prop::Class(format!("h-[{height}px]").into()),
            })
            .unwrap();
            let previous = view.read_with(cx, |view, _| view.content_height());
            let batch = core.commit().unwrap();
            sink.submit_with_content_height(batch, core.content_height());
            // Queuing a commit alone must not publish geometry ahead of its views.
            assert_eq!(
                view.read_with(cx, |view, _| view.content_height()),
                previous
            );
            cx.run_until_parked();
            assert_eq!(
                observed.read_with(cx, |seen, _| seen.last().copied()),
                Some((Some(Points(height as f32)), Some(Points(height as f32))))
            );
        }
        view.update(cx, |view, _| view.stop());
        cx.run_until_parked();
        assert_eq!(
            observed.read_with(cx, |seen, _| seen.last().copied()),
            Some((None, None))
        );
    }

    fn error(text: &str) -> AppError {
        AppError::new(nsh_engine::AppErrorKind::Uncaught, text)
    }

    /// A surface draining `messages` as if an engine had sent them.
    fn sent(surface: &Surface, messages: Vec<Message>) {
        let (tx, rx) = crossbeam_channel::unbounded();
        for m in messages {
            tx.send(m).unwrap();
        }
        *surface.rx.borrow_mut() = Some(rx);
    }

    #[test]
    fn the_hosts_theme_overrides_go_over_its_theme_and_stay_when_it_changes() {
        let surface = Surface::new(nsh_engine::Environment::default());
        let color = |tokens: &nsh_engine::Tokens, name: &str| {
            tokens.color(name).map(|c| nsh_core::Color::from(c.light).0)
        };
        let theme = |json: &str| Some(nsh_base::Theme::from_json(json).unwrap());
        *surface.host_overrides.borrow_mut() = Some(
            nsh_base::theme_overrides(r##"{ "cssVars": { "light": { "primary": "#ff0000" } } }"##)
                .unwrap(),
        );
        // Over the app's own theme: only what they name.
        let tokens = surface.host_tokens().unwrap();
        assert_eq!(color(&tokens, "primary"), Some(0xff0000ff));
        assert_eq!(color(&tokens, "secondary"), None);
        *surface.host_theme.borrow_mut() = theme(
            r##"{ "cssVars": { "light": { "primary": "#00ff00", "secondary": "#0000ff" } } }"##,
        );
        let tokens = surface.host_tokens().unwrap();
        assert_eq!(color(&tokens, "primary"), Some(0xff0000ff));
        assert_eq!(color(&tokens, "secondary"), Some(0x0000ffff));
        *surface.host_overrides.borrow_mut() = None;
        let tokens = surface.host_tokens().unwrap();
        assert_eq!(color(&tokens, "primary"), Some(0x00ff00ff));
        *surface.host_theme.borrow_mut() = None;
        assert!(surface.host_tokens().is_none());
    }

    #[test]
    fn the_hosts_locale_and_text_size_stand_in_for_the_systems() {
        let surface = Surface::new(nsh_engine::Environment::default());
        let zh = nsh_engine::Locale::parse("zh-Hans-CN");
        surface
            .host_environment
            .set(nsh_core::platform::HostEnvironment {
                locale: zh,
                font_scale: Some(1.5),
            });
        // What the system reports goes under the host's settings.
        surface.set_environment(nsh_engine::Environment {
            font_scale: 1.25,
            ..Default::default()
        });
        let env = surface.environment.get();
        assert_eq!((Some(env.locale), env.font_scale), (zh, 1.5));
        surface.host_environment.set(Default::default());
        surface.set_environment(nsh_engine::Environment {
            font_scale: 1.25,
            ..Default::default()
        });
        let env = surface.environment.get();
        assert_eq!(
            (env.locale, env.font_scale),
            (nsh_engine::Locale::EN_US, 1.25)
        );
    }

    #[test]
    fn an_error_handler_restarting_the_app_drops_the_old_engines_messages() {
        let surface = Surface::new(nsh_engine::Environment::default());
        let weak = Rc::downgrade(&surface);
        surface.set_error_handler(Box::new(move |_| {
            let surface = weak.upgrade().unwrap();
            // What a restart does first; setting a handler again must not
            // find it borrowed.
            surface.stop();
            surface.set_error_handler(Box::new(|_| {}));
        }));
        sent(
            &surface,
            vec![
                Message::Error(error("boom")),
                Message::Batch(MountBatch::default(), None),
            ],
        );
        surface.drain();
        assert_eq!(surface.batches.get(), 0);
    }

    #[test]
    fn messages_apply_in_order_without_a_restart() {
        let surface = Surface::new(nsh_engine::Environment::default());
        let errors = Rc::new(Cell::new(0));
        let seen = errors.clone();
        surface.set_error_handler(Box::new(move |_| seen.set(seen.get() + 1)));
        sent(
            &surface,
            vec![
                Message::Error(error("first")),
                Message::Batch(MountBatch::default(), None),
                Message::Error(error("second")),
            ],
        );
        surface.drain();
        assert_eq!((errors.get(), surface.batches.get()), (2, 1));
    }

    #[cfg(feature = "dev")]
    #[test]
    fn a_dev_session_hears_errors_besides_the_embedding_apps_handler() {
        let surface = Surface::new(nsh_engine::Environment::default());
        let heard = Rc::new(RefCell::new(Vec::new()));
        let host = heard.clone();
        surface.set_error_handler(Box::new(move |m| {
            host.borrow_mut().push(format!("host {m}"))
        }));
        let dev = heard.clone();
        surface.set_dev_error_handler(Box::new(move |m| dev.borrow_mut().push(format!("dev {m}"))));
        sent(&surface, vec![Message::Error(error("boom"))]);
        surface.drain();
        assert_eq!(*heard.borrow(), ["dev boom", "host boom"]);
    }

    #[test]
    fn the_error_handler_hears_what_went_wrong_and_the_package() {
        let surface = Surface::new(nsh_engine::Environment::default());
        let heard = Rc::new(RefCell::new(Vec::new()));
        let seen = heard.clone();
        surface.set_error_handler(Box::new(move |e| seen.borrow_mut().push(e.clone())));
        let package = std::sync::Arc::new(nsh_engine::PackageInfo {
            id: "com.example.app".into(),
            name: "App".into(),
            version: "1.0.0".into(),
            build: 7,
            sha256: "0".repeat(64),
        });
        let stopped = AppError::new(nsh_engine::AppErrorKind::Stopped, "out of memory")
            .with_package(Some(package));
        sent(&surface, vec![Message::Error(stopped.clone())]);
        surface.drain();
        let heard = heard.borrow();
        assert_eq!(*heard, [stopped]);
        assert!(heard[0].is_fatal());
        assert_eq!(heard[0].package.as_ref().map(|p| p.build), Some(7));
    }

    #[test]
    fn endless_animations_and_long_js_loops_are_paced_while_the_rest_are_not() {
        use nsh_core::animation::{AnimationOp, Composite, Driver, Easing};

        let surface = Surface::new(nsh_engine::Environment::default());
        assert!(!surface.wants_frames());
        assert!(!surface.wants_immediate_frame());
        assert_eq!(surface.pace(), None);

        // A JS frame request starting a loop is immediate.
        surface.js_frame.set(true);
        assert!(surface.wants_frames());
        assert!(surface.wants_immediate_frame());
        assert_eq!(surface.pace(), None);

        // A loop running past `JS_LOOP_PACE_AFTER` is paced: its callbacks
        // run on the pacer's frames, at `PACED_HZ`.
        let now = surface.clock();
        surface
            .js_loop_since
            .set(Some(now - JS_LOOP_PACE_AFTER - 0.5));
        assert!(surface.wants_frames());
        assert!(!surface.wants_immediate_frame());
        assert_eq!(surface.pace(), Some(paced_period(PACED_HZ)));

        // A button held on the surface: the loop follows at full rate.
        surface.input_pressed.set(1);
        assert!(surface.wants_immediate_frame());
        assert_eq!(surface.pace(), None);
        surface.input_pressed.set(0);
        surface.js_frame.set(false);
        surface.js_loop_since.set(None);
        surface.js_last_run.set(None);

        // An endless loop asking for no rate is paced at `PACED_HZ`.
        surface.apply(&MountBatch {
            items: vec![
                MountItem::Animation(AnimationOp::CreateValue {
                    node: 1,
                    value: 0.0,
                }),
                MountItem::Animation(AnimationOp::StartComposite {
                    animation: 1,
                    key: 1,
                    spec: Composite::Loop {
                        child: Box::new(Composite::Run {
                            node: 1,
                            driver: Driver::Timing {
                                to: 360.0,
                                duration_ms: 900.0,
                                easing: Easing::Linear,
                                delay_ms: 0.0,
                                frame_rate: None,
                            },
                        }),
                        iterations: None,
                        reset: true,
                    },
                }),
            ],
        });
        assert!(surface.wants_frames());
        assert!(!surface.wants_immediate_frame());
        assert_eq!(surface.pace(), Some(paced_period(PACED_HZ)));
        // With a paced JS loop too, one pace for both.
        surface.js_frame.set(true);
        surface
            .js_loop_since
            .set(Some(surface.clock() - JS_LOOP_PACE_AFTER - 0.5));
        assert_eq!(surface.pace(), Some(paced_period(PACED_HZ)));
        surface.js_frame.set(false);
        surface.js_loop_since.set(None);

        // A rate below `DISPLAY_HZ` asked for paces at it.
        surface.apply(&MountBatch {
            items: vec![
                MountItem::Animation(AnimationOp::StopAnimation { animation: 1 }),
                MountItem::Animation(AnimationOp::CreateValue {
                    node: 2,
                    value: 0.0,
                }),
                MountItem::Animation(AnimationOp::Start {
                    node: 2,
                    animation: 2,
                    driver: Driver::Timing {
                        to: 1.0,
                        duration_ms: 200.0,
                        easing: Easing::Linear,
                        delay_ms: 0.0,
                        frame_rate: Some(25.0),
                    },
                }),
            ],
        });
        assert!(surface.wants_frames());
        assert!(!surface.wants_immediate_frame());
        assert_eq!(surface.pace(), Some(0.04));

        // A finite transition asking for no rate runs on the display's frames.
        surface.apply(&MountBatch {
            items: vec![
                MountItem::Animation(AnimationOp::StopAnimation { animation: 2 }),
                MountItem::Animation(AnimationOp::Start {
                    node: 2,
                    animation: 3,
                    driver: Driver::Timing {
                        to: 2.0,
                        duration_ms: 200.0,
                        easing: Easing::Linear,
                        delay_ms: 0.0,
                        frame_rate: None,
                    },
                }),
            ],
        });
        assert!(surface.wants_frames());
        assert!(surface.wants_immediate_frame());
        assert_eq!(surface.pace(), None);
    }

    /// `surface` runs a looping animation (`frame_rate` as it asks).
    fn loop_animation(surface: &Rc<Surface>, frame_rate: Option<f32>) {
        use nsh_core::animation::{AnimationOp, Composite, Driver, Easing};
        surface.apply(&MountBatch {
            items: vec![
                MountItem::Animation(AnimationOp::CreateValue {
                    node: 9,
                    value: 0.0,
                }),
                MountItem::Animation(AnimationOp::StartComposite {
                    animation: 9,
                    key: 9,
                    spec: Composite::Loop {
                        child: Box::new(Composite::Run {
                            node: 9,
                            driver: Driver::Timing {
                                to: 360.0,
                                duration_ms: 900.0,
                                easing: Easing::Linear,
                                delay_ms: 0.0,
                                frame_rate,
                            },
                        }),
                        iterations: None,
                        reset: true,
                    },
                }),
            ],
        });
    }

    #[test]
    fn an_engine_message_draws_no_frame_for_a_paced_animation() {
        let surface = Surface::new(nsh_engine::Environment::default());
        loop_animation(&surface, None);
        surface.invalid.set(false);
        // Something the engine sends that changes nothing drawn: the
        // looping animation's next frame is the pacer's, not this.
        sent(&surface, vec![Message::Handled(1)]);
        surface.drain();
        assert!(!surface.invalidated());
    }

    /// `surface`'s JS frame loop has run long enough to be paced and asks
    /// for its next frame.
    fn paced_js_loop(surface: &Surface) {
        let now = surface.clock();
        surface.js_frame.set(true);
        surface
            .js_loop_since
            .set(Some(now - JS_LOOP_PACE_AFTER - 0.5));
        surface.js_last_run.set(Some(now));
    }

    #[test]
    fn a_paced_js_loop_and_animation_draw_one_frame_a_period() {
        let surface = Surface::new(nsh_engine::Environment::default());
        loop_animation(&surface, None);
        paced_js_loop(&surface);
        surface.invalid.set(false);

        // The paced frame runs the callbacks with its time and waits for
        // what they do.
        let at = surface.clock();
        surface.paced_frame(at);
        assert!(!surface.js_frame.get());
        assert_eq!(surface.js_last_run.get(), Some(at));
        assert!(!surface.invalidated());
        // The engine's turn ends asking for the next: the frame is drawn
        // now, the one frame of this period.
        sent(&surface, vec![Message::Frame]);
        surface.drain();
        assert!(surface.invalidated());
        surface.drawing();
        assert!(!surface.frame_drawn());
        // A frame drawn for anything else does not run the callbacks.
        surface.tick();
        assert!(surface.js_frame.get());
        sent(&surface, vec![Message::Handled(1)]);
        surface.drain();
        assert!(!surface.invalidated());

        // The loop ends asking for nothing: the next paced frame draws the
        // step it held back.
        surface.paced_frame(at + paced_period(PACED_HZ));
        assert!(!surface.invalidated());
        surface.paced_frame(at + 2.0 * paced_period(PACED_HZ));
        assert!(surface.invalidated());
    }

    #[test]
    fn paced_frames_keep_their_cadence() {
        let surface = Surface::new(nsh_engine::Environment::default());
        loop_animation(&surface, None);
        let period = paced_period(PACED_HZ);
        // One more microsecond than a 30th of a second: a loop skipping
        // frames closer than `1000 / 30` ms never skips one.
        assert_eq!(period, 0.033334);
        assert!(period * 1000.0 >= 1000.0 / 30.0);

        // The timer fired late: the next is still a whole number of periods
        // after the last, not a period after it fired.
        let before = surface.clock();
        let last = before - 2.5 * period;
        surface.pacer_last.set(Some(last));
        assert!(!surface.frame_drawn());
        let armed = surface.pacer_armed.get().unwrap();
        let periods = (armed - last) / period;
        assert!((periods - periods.round()).abs() < 1e-9, "{periods}");
        assert!(armed > before && armed - period <= surface.clock());

        // Hidden, none is armed.
        surface.pacer_armed.set(None);
        surface.window_hidden.set(true);
        surface.ensure_ticking();
        assert_eq!(surface.pacer_armed.get(), None);
        surface.window_hidden.set(false);

        // Not drawn for long: the JS loop waits and the pacer stops.
        paced_js_loop(&surface);
        surface.invalid.set(true);
        surface
            .invalid_since
            .set(surface.clock() - NOT_PRESENTED_AFTER - 0.1);
        surface.paced_frame(armed);
        assert!(surface.js_frame.get());
        assert_eq!(surface.pacer_armed.get(), None);
    }

    #[test]
    fn an_endless_animation_asking_for_the_displays_rate_gets_it() {
        let surface = Surface::new(nsh_engine::Environment::default());
        loop_animation(&surface, Some(120.0));
        assert!(surface.wants_immediate_frame());
        assert_eq!(surface.pace(), None);

        let surface = Surface::new(nsh_engine::Environment::default());
        loop_animation(&surface, Some(DISPLAY_HZ));
        assert!(surface.wants_immediate_frame());

        let surface = Surface::new(nsh_engine::Environment::default());
        loop_animation(&surface, Some(20.0));
        assert!(!surface.wants_immediate_frame());
        assert_eq!(surface.pace(), Some(0.05));
    }
}
