//! Sandboxed guest JS VMs sharing one engine thread and one [`nsh_core::Core`].
//!
//! Each loaded plugin gets its own QuickJS runtime (`Script`), heap allocator,
//! `harden: true` sandbox, `Posts` queue, `HostChannel` and [`TenantId`] in
//! [`nsh_core::Core`], while sharing dense ID counters ([`nsh_js::SharedIds`])
//! and the surface's single shadow tree, overlay stack, focus chain and
//! accessibility tree.

use std::{
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use nsh_core::{
    ApplyError, Core, EventKind, EventPayload, NodeId, Op, PayloadValue, SHELL_TENANT, TenantId,
    TextEngine, element::JsEvent, js::Limits, platform::Environment, time::Instant,
};
use rustc_hash::FxHashMap;

use crate::{
    AppError, AppErrorKind, Bundle, EngineConfig, HostChannel, Message, MountSink, State,
    host_channel::HostAttachment,
    script::{self, Script},
};

/// Snapshot of engine-thread counts for diagnostics and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EngineStats {
    /// Total live nodes in [`nsh_core::Core`] (shell + all plugins).
    pub nodes: usize,
    /// Number of currently live guest plugin VMs.
    pub plugins: usize,
    /// Total wakeups executed by the engine loop since startup.
    pub wakeups: u64,
    /// Whether the engine currently has any timer or idle GC deadline armed.
    pub has_deadline: bool,
}

/// Engine-level resources retained by [`crate::Driver`] so sandboxed guest
/// plugin VMs can be spawned dynamically on the engine thread.
pub(crate) struct PluginHostEnv<M: TextEngine + 'static> {
    pub(crate) ui_sets: Vec<nsh_core::element::Registry>,
    pub(crate) text_engine: Rc<M>,
    pub(crate) stop: Arc<AtomicBool>,
    #[cfg(any(feature = "js-core", all(feature = "browser", target_arch = "wasm32")))]
    pub(crate) app_url: nsh_core::url_scheme::AppUrlCell,
    pub(crate) websocket: Option<Arc<dyn nsh_core::websocket::WebSocketTransport>>,
    pub(crate) refuse_connections: Option<nsh_core::element::RefuseConnections>,
    pub(crate) filesystem: Option<Arc<dyn nsh_core::filesystem::FileSystem>>,
    pub(crate) data_dir: Option<std::path::PathBuf>,
    pub(crate) wakeups: Arc<AtomicU64>,
}

impl<M: TextEngine + 'static> PluginHostEnv<M> {
    pub(crate) fn new(
        config: &EngineConfig,
        text_engine: Rc<M>,
        stop: Arc<AtomicBool>,
        #[cfg(any(feature = "js-core", all(feature = "browser", target_arch = "wasm32")))]
        app_url: nsh_core::url_scheme::AppUrlCell,
        wakeups: Arc<AtomicU64>,
    ) -> Self {
        Self {
            ui_sets: config.ui_sets.clone(),
            text_engine,
            stop,
            #[cfg(any(feature = "js-core", all(feature = "browser", target_arch = "wasm32")))]
            app_url,
            websocket: config.websocket.clone(),
            refuse_connections: config.refuse_connections.clone(),
            filesystem: config.filesystem.clone(),
            data_dir: crate::data_dir(config),
            wakeups,
        }
    }

    pub(crate) fn wakeup_count(&self) -> u64 {
        self.wakeups.load(Ordering::Relaxed)
    }
}

/// Sliding window length for guest CPU throttling.
const CPU_WINDOW: Duration = Duration::from_secs(1);

/// Default per-call CPU budget for sandboxed guest plugin VMs.
pub const DEFAULT_PLUGIN_CALL_BUDGET: Duration = Duration::from_millis(80);
/// Default load budget for sandboxed guest plugin VMs.
pub const DEFAULT_PLUGIN_LOAD_BUDGET: Duration = Duration::from_millis(250);
/// Default QuickJS heap ceiling for sandboxed guest plugin VMs (64 MiB).
pub const DEFAULT_PLUGIN_MEMORY_BYTES: usize = 64 << 20;
/// Default core node + text-run ceiling for sandboxed guest plugin VMs.
pub const DEFAULT_PLUGIN_MAX_NODES: usize = 10_000;
/// Default cumulative CPU time a guest plugin may burn in one [`CPU_WINDOW`].
pub const DEFAULT_PLUGIN_CPU_WINDOW: Duration = Duration::from_millis(250);

/// Configuration for loading one sandboxed guest plugin VM into an engine.
#[derive(Debug, Clone)]
pub struct PluginConfig {
    /// Opaque caller-chosen identifier (reloading the same id replaces it).
    pub id: String,
    /// The plugin's JS bundle and capability policy.
    pub bundle: Bundle,
    /// Resource ceilings for this plugin's QuickJS VM and Core node quota.
    pub limits: Limits,
    /// Dedicated `nsh/host` channel for this plugin.
    pub host: HostChannel,
    /// Maximum cumulative JS CPU time allowed inside any 1-second rolling
    /// window (`None` disables sliding-window throttling; per-call
    /// `limits.call_budget` still applies).
    pub cpu_window_budget: Option<Duration>,
}

impl PluginConfig {
    /// Creates a plugin config with hardened plugin defaults (64 MiB heap,
    /// 80 ms per-call fuse, 250 ms/s rolling CPU budget, 10k node cap).
    pub fn new(id: impl Into<String>, bundle: Bundle) -> Self {
        Self {
            id: id.into(),
            bundle,
            limits: Limits {
                memory_bytes: DEFAULT_PLUGIN_MEMORY_BYTES,
                load_budget: DEFAULT_PLUGIN_LOAD_BUDGET,
                call_budget: DEFAULT_PLUGIN_CALL_BUDGET,
                max_nodes: DEFAULT_PLUGIN_MAX_NODES,
                ..Limits::default()
            },
            host: HostChannel::new(),
            cpu_window_budget: Some(DEFAULT_PLUGIN_CPU_WINDOW),
        }
    }

    /// Overrides the VM and node resource limits.
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Attaches a dedicated [`HostChannel`] for `nsh/host`.
    pub fn with_host(mut self, host: HostChannel) -> Self {
        self.host = host;
        self
    }

    /// Overrides the rolling 1-second CPU budget (`None` disables it).
    pub fn with_cpu_window_budget(mut self, budget: Option<Duration>) -> Self {
        self.cpu_window_budget = budget;
        self
    }
}

/// Rolling 1-second CPU time accumulator for one guest plugin VM.
struct CpuWindow {
    budget: Option<Duration>,
    window_start: Instant,
    spent: Duration,
}

impl CpuWindow {
    fn new(budget: Option<Duration>) -> Self {
        Self {
            budget,
            window_start: Instant::now(),
            spent: Duration::ZERO,
        }
    }

    /// Charges `elapsed` to the current window; returns an error reason if the
    /// plugin exceeded its rolling budget.
    fn charge(&mut self, elapsed: Duration) -> Option<String> {
        let budget = self.budget?;
        let now = Instant::now();
        if now.saturating_duration_since(self.window_start) >= CPU_WINDOW {
            self.window_start = now;
            self.spent = Duration::ZERO;
        }
        self.spent = self.spent.saturating_add(elapsed);
        if self.spent > budget {
            Some(format!(
                "plugin exceeded rolling CPU budget ({} ms / {} ms window)",
                budget.as_millis(),
                CPU_WINDOW.as_millis()
            ))
        } else {
            None
        }
    }
}

/// Non-release host ops submitted by a VM before cycle collection runs when
/// no structural tree mutation occurred.
pub(crate) const GC_OP_THRESHOLD: u32 = 128;

/// Whether an op mutates the retained node/slot structure (which can create
/// reference cycles across detached subtrees). Excludes [`Op::Release`] because
/// cycle collection itself emits `Op::Release` when freeing detached nodes.
pub(crate) fn is_structural_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Create { .. }
            | Op::CreateText { .. }
            | Op::Insert { .. }
            | Op::Remove { .. }
            | Op::AttachSlot { .. }
            | Op::DetachSlot { .. }
    )
}

/// One live guest plugin VM running on the engine thread.
pub(crate) struct PluginVm {
    pub(crate) id: String,
    pub(crate) tenant: TenantId,
    pub(crate) script: Script,
    pub(crate) posts: nsh_core::element::Posts,
    pub(crate) _host: HostAttachment,
    pub(crate) ops_since_gc: u32,
    pub(crate) has_structural_churn: bool,
    cpu: CpuWindow,
}

impl PluginVm {
    pub(crate) fn needs_gc(&self) -> bool {
        self.has_structural_churn || self.ops_since_gc >= GC_OP_THRESHOLD
    }

    pub(crate) fn record_ops(&mut self, ops: &[Op]) {
        let non_release = ops
            .iter()
            .filter(|op| !matches!(op, Op::Release { .. }))
            .count();
        let non_release = u32::try_from(non_release).unwrap_or(u32::MAX);
        self.ops_since_gc = self.ops_since_gc.saturating_add(non_release);
        self.has_structural_churn |= ops.iter().any(is_structural_op);
    }

    pub(crate) fn clear_gc_dirty(&mut self) {
        self.ops_since_gc = 0;
        self.has_structural_churn = false;
    }

    /// Runs `f` against this guest VM and charges its elapsed time against the
    /// rolling CPU window.
    #[inline]
    pub(crate) fn run_charged<R>(&mut self, f: impl FnOnce(&Script) -> R) -> R {
        let start = Instant::now();
        let out = f(&self.script);
        let elapsed = start.elapsed();
        if self.script.fatal().is_none()
            && let Some(reason) = self.cpu.charge(elapsed)
        {
            self.script.stop(reason);
        }
        out
    }
}

/// All live guest plugin VMs hosted inside one engine.
pub(crate) struct Plugins {
    vms: FxHashMap<TenantId, PluginVm>,
    by_id: FxHashMap<String, TenantId>,
    /// Deterministic iteration order for commits, timers and broadcasts.
    order: Vec<TenantId>,
    next_tenant: TenantId,
    gc_cursor: usize,
    /// Mirrors the engine's `content_culling` setting for newly loaded plugins.
    lazy_rows: bool,
    #[cfg(feature = "js-core")]
    shared_ids: nsh_js::SharedIds,
}

impl Plugins {
    pub(crate) fn new(lazy_rows: bool) -> Self {
        Self {
            vms: FxHashMap::default(),
            by_id: FxHashMap::default(),
            order: Vec::new(),
            next_tenant: 1,
            gc_cursor: 0,
            lazy_rows,
            #[cfg(feature = "js-core")]
            shared_ids: nsh_js::SharedIds::new(),
        }
    }

    #[cfg(feature = "js-core")]
    pub(crate) fn shared_ids(&self) -> nsh_js::SharedIds {
        self.shared_ids.clone()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.vms.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.vms.len()
    }

    /// Allocates the next unused non-zero [`TenantId`].
    pub(crate) fn alloc_tenant(&mut self) -> Option<TenantId> {
        for _ in 0..u16::MAX {
            let candidate = self.next_tenant;
            self.next_tenant = self.next_tenant.wrapping_add(1);
            if self.next_tenant == SHELL_TENANT {
                self.next_tenant = 1;
            }
            if candidate != SHELL_TENANT && !self.vms.contains_key(&candidate) {
                return Some(candidate);
            }
        }
        None
    }

    /// Inserts a newly started guest VM.
    pub(crate) fn insert(&mut self, vm: PluginVm) {
        let tenant = vm.tenant;
        self.by_id.insert(vm.id.clone(), tenant);
        self.order.push(tenant);
        self.vms.insert(tenant, vm);
    }

    /// Removes a guest VM by `id`, returning it for teardown.
    pub(crate) fn remove_by_id(&mut self, id: &str) -> Option<PluginVm> {
        let tenant = self.by_id.remove(id)?;
        self.order.retain(|&t| t != tenant);
        self.vms.remove(&tenant)
    }

    /// Removes a guest VM by `tenant`, returning it for teardown.
    pub(crate) fn remove_tenant(&mut self, tenant: TenantId) -> Option<PluginVm> {
        let vm = self.vms.remove(&tenant)?;
        self.by_id.remove(&vm.id);
        self.order.retain(|&t| t != tenant);
        Some(vm)
    }

    pub(crate) fn get_mut(&mut self, tenant: TenantId) -> Option<&mut PluginVm> {
        self.vms.get_mut(&tenant)
    }

    pub(crate) fn tenants(&self) -> Vec<TenantId> {
        self.order.clone()
    }

    pub(crate) fn next_timer(&self) -> Option<Instant> {
        self.vms
            .values()
            .filter_map(|vm| vm.script.next_timer())
            .min()
    }

    pub(crate) fn wants_frame(&self) -> bool {
        self.vms.values().any(|vm| vm.script.wants_frame())
    }

    pub(crate) fn has_posted(&self) -> bool {
        self.vms.values().any(|vm| vm.script.has_posted())
    }

    pub(crate) fn has_dirty_gc(&self) -> bool {
        self.vms.values().any(PluginVm::needs_gc)
    }

    pub(crate) fn dispatch_event(
        &mut self,
        tenant: TenantId,
        id: NodeId,
        event: EventKind,
        payload: &EventPayload,
    ) {
        if let Some(vm) = self.vms.get_mut(&tenant) {
            vm.run_charged(|s| s.dispatch_event(id, event, payload));
        }
    }

    pub(crate) fn deliver(&mut self, tenant: TenantId, event: JsEvent) {
        if let Some(vm) = self.vms.get_mut(&tenant) {
            vm.run_charged(|s| s.deliver(event));
        }
    }

    pub(crate) fn bind_list(&mut self, tenant: TenantId, list: NodeId, binds: &[(u32, u32)]) {
        if let Some(vm) = self.vms.get_mut(&tenant) {
            vm.run_charged(|s| s.bind_list(list, binds));
        }
    }

    pub(crate) fn host_event(&mut self, tenant: TenantId, name: &str, data: nsh_core::JsonValue) {
        if let Some(vm) = self.vms.get_mut(&tenant) {
            vm.run_charged(|s| s.host_event(name, data));
        }
    }

    pub(crate) fn host_reply(
        &mut self,
        tenant: TenantId,
        id: u64,
        result: Result<nsh_core::JsonValue, String>,
    ) {
        if let Some(vm) = self.vms.get_mut(&tenant) {
            vm.run_charged(|s| s.host_reply(id, result));
        }
    }

    pub(crate) fn animations_finished(
        &mut self,
        tenant: TenantId,
        finished: &[nsh_core::animation::Finished],
    ) {
        if let Some(vm) = self.vms.get_mut(&tenant) {
            vm.run_charged(|s| s.animations_finished(finished));
        }
    }

    pub(crate) fn run_timers(&mut self, now: Instant) {
        for tenant in self.tenants() {
            if let Some(vm) = self.vms.get_mut(&tenant) {
                if vm.script.next_timer().is_some_and(|due| due <= now) {
                    vm.run_charged(|s| s.run_timers(now));
                }
            }
        }
    }

    pub(crate) fn run_system(&mut self) {
        for tenant in self.tenants() {
            if let Some(vm) = self.vms.get_mut(&tenant) {
                vm.run_charged(Script::run_system);
            }
        }
    }

    pub(crate) fn run_frame(&mut self) {
        for tenant in self.tenants() {
            if let Some(vm) = self.vms.get_mut(&tenant)
                && vm.script.wants_frame()
            {
                vm.run_charged(Script::run_frame);
            }
        }
    }

    pub(crate) fn run_posted(&mut self) {
        for tenant in self.tenants() {
            if let Some(vm) = self.vms.get_mut(&tenant)
                && vm.script.has_posted()
            {
                vm.run_charged(Script::run_posted);
            }
        }
    }

    pub(crate) fn broadcast_environment(&mut self, env: &Environment) {
        for tenant in self.tenants() {
            if let Some(vm) = self.vms.get_mut(&tenant) {
                vm.run_charged(|s| s.set_environment(env));
            }
        }
    }

    pub(crate) fn broadcast_displays(&mut self, displays: &[nsh_core::desktop::Display]) {
        for tenant in self.tenants() {
            if let Some(vm) = self.vms.get_mut(&tenant) {
                vm.run_charged(|s| s.set_displays(displays));
            }
        }
    }

    pub(crate) fn broadcast_tokens(&mut self, tokens: std::sync::Arc<nsh_core::element::Tokens>) {
        for tenant in self.tenants() {
            if let Some(vm) = self.vms.get_mut(&tenant) {
                let tokens = tokens.clone();
                vm.run_charged(|s| s.set_tokens(tokens));
            }
        }
    }

    pub(crate) fn broadcast_theme_tokens(&mut self, theme: nsh_core::element::ThemeTokens) {
        for tenant in self.tenants() {
            if let Some(vm) = self.vms.get_mut(&tenant) {
                let theme = theme.clone();
                vm.run_charged(|s| s.set_theme_tokens(theme));
            }
        }
    }

    pub(crate) fn set_lazy_rows(&mut self, lazy: bool) {
        self.lazy_rows = lazy;
        for vm in self.vms.values() {
            vm.script.set_lazy_rows(lazy);
        }
    }

    /// Routes one DOM/input event to either the shell VM or the owning guest plugin.
    #[inline]
    pub(crate) fn route_event<M: TextEngine>(
        &mut self,
        script: Option<&Script>,
        core: &Core<M>,
        id: NodeId,
        event: EventKind,
        payload: &EventPayload,
    ) {
        if id == nsh_core::ROOT_ID && event == EventKind::Menu && !self.is_empty() {
            let EventPayload::Map(fields) = payload else {
                return;
            };
            // A context menu closing: each VM settles the one it showed.
            if fields
                .iter()
                .any(|(k, v)| *k == "closed" && *v == PayloadValue::Bool(true))
            {
                if let Some(script) = script {
                    script.dispatch_event(id, event, payload);
                }
                for tenant in self.tenants() {
                    self.dispatch_event(tenant, id, event, payload);
                }
                return;
            }
            // A plugin's context menu item, with the id the plugin gave it.
            let item = fields.iter().find_map(|(k, v)| match v {
                PayloadValue::Number(n) if *k == "id" => Some(*n as u32),
                _ => None,
            });
            if let Some((tenant, own)) = item.and_then(|item| core.guest_menu_item(item)) {
                let fields = fields
                    .iter()
                    .map(|(k, v)| match *k {
                        "id" => (*k, PayloadValue::Number(f64::from(own))),
                        _ => (*k, v.clone()),
                    })
                    .collect();
                self.dispatch_event(tenant, id, event, &EventPayload::Map(fields));
                return;
            }
        }
        let tenant = core.node_tenant(id);
        if tenant == SHELL_TENANT {
            if let Some(script) = script {
                script.dispatch_event(id, event, payload);
            }
        } else {
            self.dispatch_event(tenant, id, event, payload);
        }
    }

    /// Routes a virtual list cell bind to the owning VM.
    #[inline]
    pub(crate) fn route_list_bind<M: TextEngine>(
        &mut self,
        script: Option<&Script>,
        core: &Core<M>,
        list: NodeId,
        binds: &[(u32, u32)],
    ) {
        let tenant = core.node_tenant(list);
        if tenant == SHELL_TENANT {
            if let Some(script) = script {
                script.bind_list(list, binds);
            }
        } else {
            self.bind_list(tenant, list, binds);
        }
    }

    /// Routes a core-emitted [`JsEvent`] (rows, async reply, component render)
    /// to its owning VM(s).
    pub(crate) fn route_js_event<M: TextEngine>(
        &mut self,
        script: Option<&Script>,
        core: &mut Core<M>,
        event: JsEvent,
    ) {
        match event {
            JsEvent::Rows(ids) => {
                if self.is_empty() {
                    if let Some(script) = script {
                        script.deliver(JsEvent::Rows(ids));
                    }
                    return;
                }
                let mut by_tenant: FxHashMap<TenantId, Vec<NodeId>> = FxHashMap::default();
                for id in ids {
                    by_tenant.entry(core.node_tenant(id)).or_default().push(id);
                }
                for (tenant, group) in by_tenant {
                    if tenant == SHELL_TENANT {
                        if let Some(script) = script {
                            script.deliver(JsEvent::Rows(group));
                        }
                    } else {
                        self.deliver(tenant, JsEvent::Rows(group));
                    }
                }
            }
            JsEvent::Reply { call, result } => {
                let tenant = core.take_call_tenant(call);
                let reply = JsEvent::Reply { call, result };
                if tenant == SHELL_TENANT {
                    if let Some(script) = script {
                        script.deliver(reply);
                    }
                } else {
                    self.deliver(tenant, reply);
                }
            }
            other => {
                let host = match &other {
                    JsEvent::Component { host, .. }
                    | JsEvent::Children { host, .. }
                    | JsEvent::Render { host, .. }
                    | JsEvent::Unrender { host, .. } => *host,
                    _ => nsh_core::ROOT_ID,
                };
                let tenant = core.node_tenant(host);
                if tenant == SHELL_TENANT {
                    if let Some(script) = script {
                        script.deliver(other);
                    }
                } else {
                    self.deliver(tenant, other);
                }
            }
        }
    }

    /// Routes completed animations to the VM(s) that started them.
    pub(crate) fn route_animations_finished<M: TextEngine>(
        &mut self,
        script: Option<&Script>,
        core: &mut Core<M>,
        finished: Vec<nsh_core::animation::Finished>,
    ) {
        if self.is_empty() {
            if let Some(script) = script {
                script.animations_finished(&finished);
            }
            return;
        }
        let mut by_tenant: FxHashMap<TenantId, Vec<nsh_core::animation::Finished>> =
            FxHashMap::default();
        for f in finished {
            let tenant = core.take_anim_tenant(f.animation);
            by_tenant.entry(tenant).or_default().push(f);
        }
        for (tenant, group) in by_tenant {
            if tenant == SHELL_TENANT {
                if let Some(script) = script {
                    script.animations_finished(&group);
                }
            } else {
                self.animations_finished(tenant, &group);
            }
        }
    }

    /// Applies every op currently queued in `ops` under `tenant`, stopping only
    /// that VM on a fatal quota or tenant-isolation error.
    pub(crate) fn apply_ops<M: TextEngine>(
        &mut self,
        core: &mut Core<M>,
        ops: &mut Vec<Op>,
        tenant: TenantId,
        shell: Option<&Script>,
    ) -> usize {
        let count = ops.len();
        if count == 0 {
            return 0;
        }
        let mut drained = ops.drain(..);
        while let Some(op) = drained.next() {
            let op_name = (tenant != SHELL_TENANT).then(|| op_debug_tag(&op));
            match core.apply_as(tenant, op) {
                Ok(()) => {}
                Err(err) if err.is_fatal() => {
                    core.discard_ops(drained.by_ref());
                    let reason = match err {
                        ApplyError::TooManyNodes(max) => {
                            format!("the app created more than {max} nodes and was stopped")
                        }
                        ApplyError::TooDeep(max) => {
                            format!("the app nested views more than {max} deep and was stopped")
                        }
                        ApplyError::TenantViolation(id) => {
                            let kind = op_name.unwrap_or("op");
                            format!(
                                "plugin attempted unauthorized {kind} on node {id} and was stopped"
                            )
                        }
                        other => {
                            format!("the app sent an invalid op ({other:?}) and was stopped")
                        }
                    };
                    if tenant == SHELL_TENANT {
                        if let Some(script) = shell {
                            script.stop(reason);
                        }
                    } else if let Some(vm) = self.get_mut(tenant) {
                        vm.script.stop(reason);
                    }
                    break;
                }
                Err(err) => log::error!("invalid op: {err:?}"),
            }
        }
        drop(drained);
        crate::release_spare_ops(ops);
        count
    }

    /// Drains and applies queued ops from every guest plugin VM under its own
    /// `TenantId` and `Posts` queue, restoring `shell_posts` afterward.
    pub(crate) fn apply_all_ops<M: TextEngine>(
        &mut self,
        core: &mut Core<M>,
        ops: &mut Vec<Op>,
        shell_posts: &nsh_core::element::Posts,
    ) -> usize {
        if self.is_empty() {
            return 0;
        }
        let mut total = 0;
        for tenant in self.tenants() {
            let posts = {
                let Some(vm) = self.get_mut(tenant) else {
                    continue;
                };
                vm.script.take_ops(ops);
                if ops.is_empty() {
                    continue;
                }
                vm.record_ops(ops);
                vm.posts.clone()
            };
            core.set_posts(posts);
            total += self.apply_ops(core, ops, tenant, None);
        }
        core.set_posts(shell_posts.clone());
        total
    }

    /// Unloads a guest plugin VM by id and cleans its entire tenant subtree.
    pub(crate) fn unload<M: TextEngine>(&mut self, id: &str, core: &mut Core<M>) {
        if let Some(vm) = self.remove_by_id(id) {
            teardown_vm(vm, core);
        }
    }

    /// Runs cycle collection on at most one dirty guest VM per turn (round-robin),
    /// returning the collected plugin id, leaving idle and non-dirty guest VMs untouched.
    pub(crate) fn collect_one_dirty_gc(&mut self) -> Option<String> {
        if self.order.is_empty() {
            return None;
        }
        let len = self.order.len();
        for offset in 0..len {
            let idx = (self.gc_cursor + offset) % len;
            let tenant = self.order[idx];
            if let Some(vm) = self.vms.get_mut(&tenant)
                && vm.needs_gc()
            {
                let id = vm.id.clone();
                vm.script.collect_garbage();
                vm.clear_gc_dirty();
                self.gc_cursor = (idx + 1) % len;
                return Some(id);
            }
        }
        None
    }

    /// Tears down every guest VM whose `fatal()` is set (OOM, CPU fuse trip,
    /// quota overrun or tenant violation), freeing its QuickJS heap and
    /// cascading removal of all its nodes, components and animations from `core`.
    /// Returns `true` when at least one plugin was reaped.
    pub(crate) fn reap_stopped<M: TextEngine>(
        &mut self,
        core: &mut Core<M>,
        sink: &dyn MountSink,
    ) -> bool {
        let stopped: Vec<TenantId> = self
            .order
            .iter()
            .copied()
            .filter(|t| {
                self.vms
                    .get(t)
                    .is_some_and(|vm| vm.script.fatal().is_some())
            })
            .collect();
        if stopped.is_empty() {
            return false;
        }
        for tenant in stopped {
            if let Some(vm) = self.remove_tenant(tenant) {
                let reason = vm.script.fatal().unwrap_or_default();
                let message = format!("[plugin:{}] {reason}", vm.id);
                sink.report(
                    &AppError::new(AppErrorKind::Stopped, &message)
                        .with_plugin(Some(vm.id.clone())),
                );
                vm.posts.close();
                drop(vm);
                core.clear_tenant(tenant);
            }
        }
        true
    }

    /// Helper for creating a guest `PluginVm` entry with its rolling CPU window.
    pub(crate) fn make_vm(
        id: String,
        tenant: TenantId,
        script: Script,
        posts: nsh_core::element::Posts,
        host: HostAttachment,
        cpu_window_budget: Option<Duration>,
    ) -> PluginVm {
        PluginVm {
            id,
            tenant,
            script,
            posts,
            _host: host,
            ops_since_gc: 0,
            has_structural_churn: true,
            cpu: CpuWindow::new(cpu_window_budget),
        }
    }
}

/// Loads (or replaces) a sandboxed guest plugin VM on the engine thread.
pub(crate) fn load_plugin<M: TextEngine + 'static>(
    state: &mut State<Rc<M>>,
    env: &PluginHostEnv<M>,
    sink: &Rc<dyn MountSink>,
    plugin: PluginConfig,
) {
    state.plugins.unload(&plugin.id, &mut state.core);
    let Some(tenant) = state.plugins.alloc_tenant() else {
        log::error!(
            "too many concurrent plugins loaded; refused `{}`",
            plugin.id
        );
        return;
    };
    state
        .core
        .set_tenant_max_nodes(tenant, plugin.limits.max_nodes);
    let posts = nsh_core::element::Posts::new();
    let wake_tx = state.net.tx.clone();
    posts.set_notify(Arc::new(move || {
        let _ = wake_tx.send(Message::Wake);
    }));
    let tx = state.net.tx.clone();
    let attachment = plugin.host.attach_tenant(tenant, tx.clone());
    #[cfg(any(feature = "js-core", all(feature = "browser", target_arch = "wasm32")))]
    let system = {
        let system_tx = tx.clone();
        let bridge = plugin.host.bridge_tenant(tenant, tx);
        script::SystemConfig {
            access: plugin.bundle.policy.access(),
            websocket: env.websocket.clone(),
            refuse_connections: env.refuse_connections.clone(),
            filesystem: env.filesystem.clone(),
            updater: None,
            app_url: env.app_url.clone(),
            data_dir: env.data_dir.clone(),
            paths: crate::bundle_system_paths(&plugin.bundle),
            wake: Arc::new(move || {
                let _ = system_tx.send(Message::Wake);
            }),
            host: Rc::new(bridge),
            lazy_rows: state.plugins.lazy_rows,
            #[cfg(feature = "js-core")]
            shared_ids: Some(state.plugins.shared_ids()),
            #[cfg(feature = "js-core")]
            guest: true,
            #[cfg(feature = "js-core")]
            plugin_id: Some(plugin.id.clone()),
        }
    };
    #[cfg(not(any(feature = "js-core", all(feature = "browser", target_arch = "wasm32"))))]
    let system = script::SystemConfig;
    let Some((script, _)) = Script::start(
        &plugin.bundle,
        plugin.limits,
        &env.ui_sets,
        posts.clone(),
        &state.environment,
        &state.displays,
        state.js_tokens.clone(),
        state.core.theme_tokens(),
        env.text_engine.clone(),
        env.stop.clone(),
        sink.clone(),
        system,
    ) else {
        posts.close();
        state.core.clear_tenant(tenant);
        return;
    };
    if let Some(reason) = script.fatal() {
        let message = format!("[plugin:{}] {reason}", plugin.id);
        sink.report(
            &AppError::new(AppErrorKind::Stopped, &message).with_plugin(Some(plugin.id.clone())),
        );
        posts.close();
        drop(script);
        state.core.clear_tenant(tenant);
        return;
    }
    script.run_system();
    let vm = Plugins::make_vm(
        plugin.id,
        tenant,
        script,
        posts,
        attachment,
        plugin.cpu_window_budget,
    );
    state.plugins.insert(vm);
}

/// Drops a removed plugin VM and cleans its entire [`Core`] tenant state.
pub(crate) fn teardown_vm<M: TextEngine>(vm: PluginVm, core: &mut Core<M>) {
    let tenant = vm.tenant;
    vm.posts.close();
    drop(vm);
    core.clear_tenant(tenant);
}

/// Short op kind label for tenant-violation diagnostics.
pub(crate) fn op_debug_tag(op: &nsh_core::Op) -> &'static str {
    use nsh_core::Op;
    match op {
        Op::Create { .. } => "Create",
        Op::CreateText { .. } => "CreateText",
        Op::Evaluate { .. } => "Evaluate",
        Op::Draw { .. } => "Draw",
        Op::SetText { .. } => "SetText",
        Op::SetStyle { .. } => "SetStyle",
        Op::SetProp { .. } => "SetProp",
        Op::Listen { .. } => "Listen",
        Op::Insert { .. } => "Insert",
        Op::Remove { .. } => "Remove",
        Op::Placeholder { .. } => "Placeholder",
        Op::Release { .. } => "Release",
        Op::AttachSlot { .. } => "AttachSlot",
        Op::DetachSlot { .. } => "DetachSlot",
        Op::Command { .. } => "Command",
        Op::Rust(_) => "Rust",
        Op::RenderCanvas { .. } => "RenderCanvas",
        Op::Animation(_) => "Animation",
    }
}

#[cfg(all(test, feature = "js-core"))]
mod tests {
    use std::sync::{Arc, Mutex};

    use nsh_core::{AttributedText, Points, Size, TextEngine, TextLayout, access::Capabilities};

    use crate::{
        App, AppError, Bundle, Engine, EngineConfig, HostChannel, MountSink, PluginConfig, Policy,
    };

    struct DummyText;
    impl TextEngine for DummyText {
        type Prepared = ();
        fn prepare(&self, _: &AttributedText) -> Self::Prepared {}
        fn measure(&self, _: &Self::Prepared, _: Option<Points>) -> Size<Points> {
            Size::new(Points(40.0), Points(16.0))
        }
        fn layout(&self, _: &Self::Prepared, _: Size<Points>) -> TextLayout {
            TextLayout(Arc::new(()))
        }
    }

    #[derive(Clone, Default)]
    struct DummySink(Arc<Mutex<Vec<AppError>>>);
    impl MountSink for DummySink {
        fn submit(&self, _: nsh_core::MountBatch) {}
        fn report(&self, err: &AppError) {
            self.0.lock().unwrap().push(err.clone());
        }
    }

    fn bundle_tsx(code: &str) -> Bundle {
        let compiled = nsh_bundler::compile_module("App.tsx", code, &[]).expect("compile");
        let mut policy = Policy::default();
        let mut declared = Capabilities::all();
        declared.host = vec!["probe".into()];
        policy.declared = Some(declared);
        Bundle::source(compiled).with_policy(policy)
    }

    #[test]
    fn guest_100_mount_slot_ops_never_touch_root() {
        let shell_tsx = r#"
import { host, mount, Scroll, Text, View } from "nsh";
const ROWS = Array.from({ length: 100 }, (_, i) => i);
function Shell() {
  host.post("probe", "shell.ready");
  return (
    <Scroll style={{ flex: 1 }}>
      {ROWS.map((i) => (
        <View testID={`row-${i}`} style={{ flexDirection: "column", width: 760 }}>
          <View style={{ height: 16 }}><Text>{`Session ${i}`}</Text></View>
          <View slotTarget={`row:${i}`} style={{ flexDirection: "row", gap: 4 }} />
        </View>
      ))}
    </Scroll>
  );
}
mount(Shell);
"#;
        let plugin_a_tsx = r##"
import { host, mountSlot, signal, Text, View } from "nsh";
const expanded = Array.from({ length: 100 }, () => signal(false));
for (let i = 0; i < 100; i++) {
  mountSlot(`row:${i}`, () => (
    <View
      testID={`pa-badge-${i}`}
      onPress={() => host.post("probe", `pa.click.${i}`)}
      style={{ width: 120, height: expanded[i]() ? 76 : 20, backgroundColor: "#224466" }}
    >
      <Text>{expanded[i]() ? `Expanded ${i}` : `Tag ${i}`}</Text>
    </View>
  ));
}
host.on("toggle", (d: any) => {
  expanded[d.row].set(!expanded[d.row]());
});
host.post("probe", "pa.ready");
"##;
        let sink = DummySink::default();
        let mut cfg = EngineConfig::new(App::Js(bundle_tsx(shell_tsx)));
        cfg.content_culling = true;
        cfg.ui_sets = vec![nsh_base::registry()];
        cfg.viewport = Size::new(Points(800.0), Points(600.0));
        cfg.environment.os = nsh_core::platform::Os::MacOs;
        let engine = Engine::start(cfg, || DummyText, sink.clone()).expect("engine");
        engine.control().set_content_culling(false);
        engine
            .control()
            .resize(Size::new(Points(800.0), Points(600.0)));
        engine.control().set_displays(vec![]);
        let (tx0, rx0) = std::sync::mpsc::channel();
        engine.control().snapshot_stats(move |s| {
            let _ = tx0.send(s);
        });
        let _ = rx0
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("stats0");

        let plugin_b_tsx = r##"
import { host, mountSlot, Text, View } from "nsh";
mountSlot("row:42", () => (
  <View testID="pb-badge-42" onPress={() => host.post("probe", "pb.click.42")} style={{ width: 80, height: 20 }}>
    <Text>Git</Text>
  </View>
));
host.post("probe", "pb.ready");
"##;
        engine.control().load_plugin(
            PluginConfig::new("plugin-a", bundle_tsx(plugin_a_tsx)).with_host(HostChannel::new()),
        );
        engine.control().load_plugin(
            PluginConfig::new("plugin-b", bundle_tsx(plugin_b_tsx)).with_host(HostChannel::new()),
        );
        let mut env2 = nsh_core::platform::Environment::default();
        env2.os = nsh_core::platform::Os::MacOs;
        env2.window = Size::new(Points(800.0), Points(600.0));
        engine.control().set_environment(env2);
        let (tx, rx) = std::sync::mpsc::channel();
        engine.control().snapshot_stats(move |s| {
            let _ = tx.send(s);
        });
        let stats = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("stats");
        let errors = sink.0.lock().unwrap().clone();
        assert!(errors.is_empty(), "errors: {errors:?}, stats={stats:?}");
        assert_eq!(stats.plugins, 2, "stats={stats:?}");
    }

    #[test]
    fn guest_layout_animation_animates_its_views_and_keeps_it_running() {
        let shell_tsx = r#"
import { mount, View } from "nsh";
mount(() => <View slotTarget="slot" style={{ width: 200, height: 100 }} />);
"#;
        let plugin_tsx = r#"
import { host, LayoutAnimation, mountSlot, signal, View } from "nsh";
const open = signal(false);
mountSlot("slot", () => <View style={{ width: 20, height: open() ? 60 : 20 }} />);
host.on("toggle", () => {
  LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
  open.set(!open());
});
"#;
        let sink = DummySink::default();
        let mut cfg = EngineConfig::new(App::Js(bundle_tsx(shell_tsx)));
        cfg.viewport = Size::new(Points(200.0), Points(100.0));
        cfg.environment.os = nsh_core::platform::Os::MacOs;
        let engine = Engine::start(cfg, || DummyText, sink.clone()).expect("engine");
        let stats_of = || {
            let (tx, rx) = std::sync::mpsc::channel();
            engine.control().snapshot_stats(move |s| {
                let _ = tx.send(s);
            });
            rx.recv_timeout(std::time::Duration::from_secs(5))
                .expect("stats")
        };
        let ch = HostChannel::new();
        engine.control().load_plugin(
            PluginConfig::new("animates", bundle_tsx(plugin_tsx)).with_host(ch.clone()),
        );
        assert_eq!(stats_of().plugins, 1);
        ch.send("toggle", serde_json::Value::Null).unwrap();
        let stats = stats_of();
        let errs = sink.0.lock().unwrap().clone();
        assert!(errs.is_empty(), "errors: {errs:?}");
        assert_eq!(stats.plugins, 1);
    }

    #[derive(Clone, Default)]
    struct RecordingSink {
        items: Arc<Mutex<Vec<nsh_core::MountItem>>>,
        errors: Arc<Mutex<Vec<AppError>>>,
    }
    impl MountSink for RecordingSink {
        fn submit(&self, batch: nsh_core::MountBatch) {
            self.items.lock().unwrap().extend(batch.items);
        }
        fn report(&self, err: &AppError) {
            self.errors.lock().unwrap().push(err.clone());
        }
    }

    /// A plugin's context menu shows; its chosen item runs in the plugin,
    /// not in the shell, whose menu ids are its own; its promise settles
    /// when it closes.
    #[test]
    fn guest_context_menu_runs_its_items() {
        let shell_tsx = r#"
import { host, mount, View } from "nsh";
import { Menu } from "nsh/desktop";
mount(() => <View slotTarget="slot" style={{ width: 200, height: 100 }} />);
host.on("menu", () => {
  Menu.popup([{ label: "Shell", onSelect: () => host.post("probe", "shell") }]);
});
"#;
        let plugin_tsx = r#"
import { host, mountSlot, View } from "nsh";
import { Menu } from "nsh/desktop";
mountSlot("slot", () => <View style={{ width: 20, height: 20 }} />);
host.on("menu", () => {
  Menu.popup([{ label: "Open", onSelect: () => host.post("probe", "open") }], { x: 4, y: 4 })
    .then(() => host.post("probe", "closed"));
});
"#;
        let sink = RecordingSink::default();
        let posts = Arc::new(Mutex::new(Vec::<String>::new()));
        let shell_channel = HostChannel::new();
        let heard = posts.clone();
        shell_channel.handle("probe", move |call| {
            heard.lock().unwrap().push(format!("shell:{}", call.data()));
        });
        let mut cfg = EngineConfig::new(App::Js(bundle_tsx(shell_tsx)));
        cfg.viewport = Size::new(Points(200.0), Points(100.0));
        cfg.environment.os = nsh_core::platform::Os::MacOs;
        cfg.host = shell_channel.clone();
        let engine = Engine::start(cfg, || DummyText, sink.clone()).expect("engine");
        let settle = || {
            let (tx, rx) = std::sync::mpsc::channel();
            engine.control().flush(move |_| {
                let _ = tx.send(());
            });
            rx.recv_timeout(std::time::Duration::from_secs(5))
                .expect("flush");
        };
        let ch = HostChannel::new();
        let heard = posts.clone();
        ch.handle("probe", move |call| {
            heard
                .lock()
                .unwrap()
                .push(format!("plugin:{}", call.data()));
        });
        engine
            .control()
            .load_plugin(PluginConfig::new("menus", bundle_tsx(plugin_tsx)).with_host(ch.clone()));
        settle();
        // The shell's menu first: its item's id is the shell's.
        shell_channel.send("menu", serde_json::Value::Null).unwrap();
        settle();
        ch.send("menu", serde_json::Value::Null).unwrap();
        settle();
        let popups: Vec<u32> = sink
            .items
            .lock()
            .unwrap()
            .iter()
            .filter_map(|item| match item {
                nsh_core::MountItem::Command {
                    id: nsh_core::ROOT_ID,
                    command: nsh_core::Command::Desktop(desktop),
                } => match &**desktop {
                    nsh_core::desktop::DesktopCommand::PopupMenu { items, .. } => {
                        match items.first() {
                            Some(nsh_core::desktop::MenuItem::Action(a)) => Some(a.id),
                            _ => None,
                        }
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(popups.len(), 2, "{popups:?}");
        assert_ne!(popups[0], popups[1], "{popups:?}");
        let menu = |fields: Vec<(&'static str, nsh_core::PayloadValue)>| {
            engine.control().dispatch_event(
                nsh_core::ROOT_ID,
                nsh_core::EventKind::Menu,
                nsh_core::EventPayload::Map(fields),
            );
        };
        menu(vec![(
            "id",
            nsh_core::PayloadValue::Number(f64::from(popups[1])),
        )]);
        menu(vec![
            ("id", nsh_core::PayloadValue::Number(0.0)),
            ("closed", nsh_core::PayloadValue::Bool(true)),
        ]);
        settle();
        let errs = sink.errors.lock().unwrap().clone();
        assert!(errs.is_empty(), "errors: {errs:?}");
        assert_eq!(
            *posts.lock().unwrap(),
            ["plugin:\"open\"", "plugin:\"closed\""]
        );
    }

    /// What only the shell does to the whole window (a shortcut of the
    /// app's, here) throws where a plugin asks for it, with its stack; the
    /// plugin keeps running.
    #[test]
    fn guest_writing_the_root_throws_and_keeps_it_running() {
        let shell_tsx = r#"
import { mount, View } from "nsh";
mount(() => <View slotTarget="slot" style={{ width: 200, height: 100 }} />);
"#;
        let plugin_tsx = r#"
import { host, keymap, mountSlot, View } from "nsh";
mountSlot("slot", () => <View style={{ width: 20, height: 20 }} />);
function bindSearch() {
  keymap.bind("mod+k", () => {});
}
host.on("bind", () => bindSearch());
"#;
        let sink = DummySink::default();
        let mut cfg = EngineConfig::new(App::Js(bundle_tsx(shell_tsx)));
        cfg.viewport = Size::new(Points(200.0), Points(100.0));
        cfg.environment.os = nsh_core::platform::Os::MacOs;
        let engine = Engine::start(cfg, || DummyText, sink.clone()).expect("engine");
        let stats_of = || {
            let (tx, rx) = std::sync::mpsc::channel();
            engine.control().snapshot_stats(move |s| {
                let _ = tx.send(s);
            });
            rx.recv_timeout(std::time::Duration::from_secs(5))
                .expect("stats")
        };
        let ch = HostChannel::new();
        engine
            .control()
            .load_plugin(PluginConfig::new("binds", bundle_tsx(plugin_tsx)).with_host(ch.clone()));
        assert_eq!(stats_of().plugins, 1, "{:?}", sink.0.lock().unwrap());
        ch.send("bind", serde_json::Value::Null).unwrap();
        let stats = stats_of();
        let errs = sink.0.lock().unwrap().clone();
        assert_eq!(errs.len(), 1, "errors: {errs:?} stats: {stats:?}");
        assert_eq!(errs[0].kind, crate::AppErrorKind::Uncaught, "{errs:?}");
        let stack = errs[0].stack.as_deref().unwrap_or_default();
        assert!(stack.contains("bindSearch"), "{errs:?}");
        assert_eq!(stats.plugins, 1);
    }

    #[test]
    fn guest_errors_tag_plugin_id_and_cleanup_failed_load() {
        let shell_tsx = r#"
import { mount, View } from "nsh";
mount(() => <View slotTarget="slot" style={{ width: 100, height: 40 }} />);
"#;
        let sink = DummySink::default();
        let mut cfg = EngineConfig::new(App::Js(bundle_tsx(shell_tsx)));
        cfg.viewport = Size::new(Points(200.0), Points(100.0));
        cfg.environment.os = nsh_core::platform::Os::MacOs;
        let engine = Engine::start(cfg, || DummyText, sink.clone()).expect("engine");
        let stats_of = || {
            let (tx, rx) = std::sync::mpsc::channel();
            engine.control().snapshot_stats(move |s| {
                let _ = tx.send(s);
            });
            rx.recv_timeout(std::time::Duration::from_secs(5))
                .expect("stats")
        };
        let baseline = stats_of();
        assert_eq!(baseline.plugins, 0);

        // 1. Top-level throw: reports Load tagged with plugin id and tears down partial slot nodes.
        let bad_load = r#"
import { mountSlot, View } from "nsh";
mountSlot("slot", () => <View style={{ width: 20, height: 20 }} />);
throw new Error("top-boom");
"#;
        engine
            .control()
            .load_plugin(PluginConfig::new("broken-load", bundle_tsx(bad_load)));
        let after_bad = stats_of();
        assert_eq!(after_bad.plugins, 0);
        assert_eq!(after_bad.nodes, baseline.nodes);
        {
            let errs = sink.0.lock().unwrap();
            assert_eq!(errs.len(), 1, "errors: {errs:?}");
            assert_eq!(errs[0].kind, crate::AppErrorKind::Load);
            assert_eq!(errs[0].plugin.as_deref(), Some("broken-load"));
        }

        // 2. Top-level infinite loop: reports a single Stopped error tagged with plugin id.
        sink.0.lock().unwrap().clear();
        let looper = r#"for (;;) {}"#;
        let mut looper_cfg = PluginConfig::new("looper", bundle_tsx(looper));
        looper_cfg.limits.load_budget = std::time::Duration::from_millis(40);
        engine.control().load_plugin(looper_cfg);
        let after_loop = stats_of();
        assert_eq!(after_loop.plugins, 0);
        {
            let errs = sink.0.lock().unwrap();
            assert_eq!(errs.len(), 1, "errors: {errs:?}");
            assert_eq!(errs[0].kind, crate::AppErrorKind::Stopped);
            assert_eq!(errs[0].plugin.as_deref(), Some("looper"));
        }

        // 3. Uncaught callback exception: reports Uncaught tagged with plugin id; plugin stays alive.
        sink.0.lock().unwrap().clear();
        let uncaught = r#"
import { host } from "nsh";
host.on("poke", () => { throw new Error("cb-boom"); });
"#;
        let ch = HostChannel::new();
        engine
            .control()
            .load_plugin(PluginConfig::new("uncaught", bundle_tsx(uncaught)).with_host(ch.clone()));
        assert_eq!(stats_of().plugins, 1);
        ch.send("poke", serde_json::Value::Null).unwrap();
        let after_uncaught = stats_of();
        assert_eq!(after_uncaught.plugins, 1);
        {
            let errs = sink.0.lock().unwrap();
            assert_eq!(errs.len(), 1, "errors: {errs:?}");
            assert_eq!(errs[0].kind, crate::AppErrorKind::Uncaught);
            assert_eq!(errs[0].plugin.as_deref(), Some("uncaught"));
        }

        // 4. Full rolldown bundle (with `_$setup` prologue) in a guest VM never touches ROOT_ID.
        sink.0.lock().unwrap().clear();
        let dir = std::env::temp_dir().join(format!(
            "nsh-guest-bundle-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("nsh.json"), "{}").unwrap();
        std::fs::write(
            dir.join("App.tsx"),
            r#"import { mountSlot, View } from "nsh";
mountSlot("slot", () => <View style={{ width: 10, height: 10 }} />);
"#,
        )
        .unwrap();
        let bundled = nsh_bundler::bundle(&nsh_bundler::BundleOptions {
            entry: dir.join("App.tsx"),
            ..Default::default()
        })
        .expect("rolldown bundle");
        let _ = std::fs::remove_dir_all(&dir);
        engine.control().load_plugin(PluginConfig::new(
            "bundled",
            Bundle::source(bundled.code),
        ));
        let after_bundled = stats_of();
        let errs = sink.0.lock().unwrap().clone();
        assert!(errs.is_empty(), "bundled guest errors: {errs:?}");
        assert_eq!(after_bundled.plugins, 2);
    }

    #[test]
    fn mount_slot_holder_fills_flex_containers_and_keeps_intrinsic_sizes() {
        let shell_src = r#"
import { mount, View } from "nsh";
mount(() => (
  <View style={{ width: 260, height: 600 }}>
    <View slotTarget="top" />
    <View slotTarget="fill" style={{ flex: 1, minHeight: 0 }} />
  </View>
));
"#;
        let plugin_src = r#"
import { host, mountSlot, Scroll, View } from "nsh";
mountSlot("top", () => (
  <View style={{ height: 48 }} onLayout={(e) => host.post("layout", { slot: "top", ...e })} />
));
mountSlot("fill", () => (
  <View style={{ flex: 1, minHeight: 0 }}>
    <Scroll
      style={{ flex: 1 }}
      onLayout={(e) => host.post("layout", { slot: "fill", ...e })}
    >
      <View style={{ height: 2000 }} />
    </Scroll>
  </View>
));
"#;
        let layouts: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
        let ch = HostChannel::new();
        {
            let layouts = layouts.clone();
            ch.handle("layout", move |call| {
                layouts.lock().unwrap().push(call.data().clone());
                call.resolve(serde_json::Value::Null);
            });
        }
        let sink = DummySink::default();
        let mut cfg = EngineConfig::new(App::Js(bundle_tsx(shell_src)));
        cfg.viewport = Size::new(Points(260.0), Points(600.0));
        cfg.environment.os = nsh_core::platform::Os::MacOs;
        let engine = Engine::start(cfg, || DummyText, sink.clone()).expect("start engine");
        let mut policy = Policy::default();
        let mut declared = nsh_core::access::Capabilities::all();
        declared.host = vec!["layout".to_owned()];
        policy.declared = Some(declared);
        engine.control().load_plugin(
            PluginConfig::new("layout-plugin", bundle_tsx(plugin_src).with_policy(policy))
                .with_host(ch),
        );

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let (tx, rx) = std::sync::mpsc::channel();
            engine.control().snapshot_stats(move |s| {
                let _ = tx.send(s);
            });
            let _ = rx.recv_timeout(std::time::Duration::from_secs(2));
            let seen = layouts.lock().unwrap();
            let top = seen.iter().rfind(|v| v["slot"] == "top").cloned();
            let fill = seen.iter().rfind(|v| v["slot"] == "fill").cloned();
            if let (Some(top), Some(fill)) = (top, fill) {
                assert_eq!(top["bounds"]["size"]["height"].as_f64(), Some(48.0), "top={top}");
                assert_eq!(fill["bounds"]["size"]["width"].as_f64(), Some(260.0), "fill={fill}");
                assert_eq!(fill["bounds"]["size"]["height"].as_f64(), Some(552.0), "fill={fill}");
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "did not receive both slot layouts; seen={:?}, errors={:?}",
                *seen,
                *sink.0.lock().unwrap(),
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    #[test]
    fn is_structural_op_tracks_tree_mutations_and_ignores_release() {
        use super::is_structural_op;
        use nsh_core::Op;

        assert!(is_structural_op(&Op::Create {
            id: 1,
            tag: nsh_core::Tag::View,
        }));
        assert!(is_structural_op(&Op::Remove {
            parent: 1,
            child: 2,
        }));
        assert!(is_structural_op(&Op::Insert {
            parent: 1,
            child: 2,
            before: None,
        }));
        assert!(!is_structural_op(&Op::Release { id: 1 }));
        assert!(!is_structural_op(&Op::SetText {
            id: 1,
            text: String::new(),
        }));
        assert!(!is_structural_op(&Op::SetStyle {
            id: 1,
            style: Box::default(),
            patch: nsh_core::StylePatch::default(),
        }));
    }

    #[test]
    fn collect_one_dirty_gc_collects_at_most_one_vm_round_robin() {
        use super::{GC_OP_THRESHOLD, Plugins, Script, TenantId, script};
        use nsh_core::{Core, Op};

        let (tx, _rx) = crossbeam_channel::unbounded();
        let host = HostChannel::new();
        let env = crate::Environment::default();
        let displays: Arc<[nsh_core::desktop::Display]> = Arc::from(Vec::new());
        let core = Core::new(
            std::rc::Rc::new(DummyText),
            Size::new(Points(400.0), Points(300.0)),
        );
        let mut plugins = Plugins::new(false);
        let shared_ids = plugins.shared_ids();
        let make_vm = |id: &str, tenant: TenantId| {
            let bundle = bundle_tsx("");
            let cfg = PluginConfig::new(id, bundle.clone());
            let posts = nsh_core::element::Posts::new();
            let attachment = host.attach_tenant(tenant, tx.clone());
            let system = script::SystemConfig {
                access: Default::default(),
                websocket: None,
                refuse_connections: None,
                filesystem: None,
                updater: None,
                app_url: Default::default(),
                data_dir: None,
                paths: crate::bundle_system_paths(&bundle),
                wake: Arc::new(|| {}),
                host: std::rc::Rc::new(host.bridge_tenant(tenant, tx.clone())),
                lazy_rows: false,
                shared_ids: Some(shared_ids.clone()),
                guest: true,
                plugin_id: Some(id.to_owned()),
            };
            let (script, _) = Script::start(
                &bundle,
                cfg.limits,
                &[],
                posts.clone(),
                &env,
                &displays,
                core.tokens().clone(),
                core.theme_tokens(),
                std::rc::Rc::new(DummyText),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                std::rc::Rc::new(DummySink::default()),
                system,
            )
            .expect("script start");
            Plugins::make_vm(id.to_owned(), tenant, script, posts, attachment, None)
        };

        plugins.insert(make_vm("vm-a", 1));
        plugins.insert(make_vm("vm-b", 2));
        assert!(plugins.has_dirty_gc());

        assert_eq!(plugins.collect_one_dirty_gc().as_deref(), Some("vm-a"));
        assert!(!plugins.get_mut(1).unwrap().needs_gc());
        assert!(plugins.get_mut(2).unwrap().needs_gc());
        assert!(plugins.has_dirty_gc());

        assert_eq!(plugins.collect_one_dirty_gc().as_deref(), Some("vm-b"));
        assert!(!plugins.get_mut(2).unwrap().needs_gc());
        assert!(!plugins.has_dirty_gc());
        assert_eq!(plugins.collect_one_dirty_gc(), None);

        // Op::Release alone must never re-dirty a collected VM.
        plugins
            .get_mut(1)
            .unwrap()
            .record_ops(&[Op::Release { id: 1 }]);
        assert!(!plugins.get_mut(1).unwrap().needs_gc());

        // Non-structural ops accumulate up to GC_OP_THRESHOLD before dirtying.
        let mut updates: Vec<Op> = (0..(GC_OP_THRESHOLD - 1))
            .map(|i| Op::SetText {
                id: i,
                text: String::new(),
            })
            .collect();
        plugins.get_mut(1).unwrap().record_ops(&updates);
        assert!(!plugins.get_mut(1).unwrap().needs_gc());
        updates.truncate(1);
        plugins.get_mut(1).unwrap().record_ops(&updates);
        assert!(plugins.get_mut(1).unwrap().needs_gc());
    }
}
