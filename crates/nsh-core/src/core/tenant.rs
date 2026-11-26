//! Multi-tenant isolation and cross-tenant slot mounting on a single [`Core`].
//!
//! Each JS VM on the engine thread owns a distinct [`TenantId`]
//! ([`SHELL_TENANT`] = `0` for the primary shell VM; non-zero ids for plugin
//! VMs). All tenants share one layout tree, overlay stack, focus chain and
//! accessibility tree, while node ownership, quotas, cross-tenant slot
//! attachments and cascade teardown are enforced here.

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use super::{ApplyError, Core, NO_KEY};
use crate::{
    Command, NodeId, Op, ROOT_ID, TextEngine,
    animation::{AnimationId, AnimationOp, NodeId as AnimNodeId},
    desktop::{DesktopCommand, MenuItem},
    element::RustCall,
    mount::MountItem,
};

/// Identifies which JS VM owns a node, run, call or animation in a [`Core`].
pub type TenantId = u16;

/// The primary surface VM (shell or standalone app).
pub const SHELL_TENANT: TenantId = 0;

/// Per-[`Core`] multi-tenant ownership, quota and slot attachment tables.
#[derive(Default)]
pub(super) struct TenantState {
    /// Tenant currently applying ops via [`Core::apply_as`].
    pub(super) current: TenantId,
    /// Per-tenant node + run ceiling (`SHELL_TENANT` uses `Core::max_nodes`).
    quotas: FxHashMap<TenantId, usize>,
    /// Live node + run count for each non-shell tenant.
    counts: FxHashMap<TenantId, usize>,
    /// Named slot target -> host node id (the container accepting plugin mounts).
    slot_targets: FxHashMap<Arc<str>, NodeId>,
    /// Reverse map: host node id -> its active slot target name.
    node_slot_target: FxHashMap<NodeId, Arc<str>>,
    /// Slot target name -> mounted wrapper nodes `(tenant, child_id)` in order.
    slot_mounts: FxHashMap<Arc<str>, Vec<(TenantId, NodeId)>>,
    /// Reverse map: mounted wrapper node id -> `(tenant, slot_name)`.
    mounted_slots: FxHashMap<NodeId, (TenantId, Arc<str>)>,
    /// Pending async/measure/rust call id -> owning tenant.
    call_owners: FxHashMap<u32, TenantId>,
    /// Pending started animation / read request id -> owning tenant.
    anim_owners: FxHashMap<AnimationId, TenantId>,
    /// Live animation graph node ids created by each non-shell tenant.
    anim_nodes: FxHashMap<TenantId, FxHashSet<AnimNodeId>>,
    /// Live animation composite keys created by each non-shell tenant.
    anim_composites: FxHashMap<TenantId, FxHashSet<u32>>,
    /// The items of the plugin's context menu showing: their id at the
    /// platform, their tenant and the id the plugin gave them.
    guest_menu: Vec<(u32, TenantId, u32)>,
    /// The next item of a plugin's context menu, from [`GUEST_MENU_FIRST_ID`].
    next_guest_menu: u32,
}

/// The first id of a plugin's context menu item at the platform: above the
/// ids JS gives (counted from 1 in each VM, which the plugins' would meet),
/// below the views' (`element::menus::FIRST_ID`). The root's `Menu` event
/// of one goes to its plugin with its own id ([`Core::guest_menu_item`]).
const GUEST_MENU_FIRST_ID: u32 = 1 << 30;

impl TenantState {
    /// Increments the live count for `tenant` (no-op for [`SHELL_TENANT`]).
    #[inline]
    pub(super) fn inc(&mut self, tenant: TenantId) {
        if tenant != SHELL_TENANT {
            *self.counts.entry(tenant).or_insert(0) += 1;
        }
    }

    /// Decrements the live count for `tenant` (no-op for [`SHELL_TENANT`]).
    #[inline]
    pub(super) fn dec(&mut self, tenant: TenantId) {
        if tenant != SHELL_TENANT
            && let Some(count) = self.counts.get_mut(&tenant)
        {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.counts.remove(&tenant);
            }
        }
    }

    /// Checks whether `tenant` can allocate one more node or text run.
    #[inline]
    pub(super) fn check_quota(&self, tenant: TenantId) -> Result<(), ApplyError> {
        if tenant == SHELL_TENANT {
            return Ok(());
        }
        if let Some(&limit) = self.quotas.get(&tenant) {
            let live = self.counts.get(&tenant).copied().unwrap_or(0);
            if live >= limit {
                return Err(ApplyError::TooManyNodes(limit));
            }
        }
        Ok(())
    }
}

/// Whether a plugin VM may send `command` to the root: a layout animation
/// there animates the plugin's own views (`layout_scope.rs`), a context
/// menu's items report to the plugin ([`Core::guest_menu_item`]), and
/// `ClearJsResponder` (`touch.rs`) is a benign global reset.
fn guest_root_command(command: &Command) -> bool {
    match command {
        Command::ClearJsResponder | Command::LayoutAnimation(_) => true,
        Command::Desktop(desktop) => matches!(**desktop, DesktopCommand::PopupMenu { .. }),
        _ => false,
    }
}

/// Whether `op` writes to the root, what only the shell does (the app's
/// shortcuts, menus, theme, windows): [`Core::apply_as`] stops a plugin VM
/// for it, and the VMs refuse it where a plugin asks, with its stack.
pub fn guest_writes_root(op: &Op) -> bool {
    let root = |id: &NodeId| *id == ROOT_ID;
    match op {
        Op::Create { id, .. }
        | Op::CreateText { id, .. }
        | Op::Evaluate { id, .. }
        | Op::Draw { id, .. }
        | Op::SetText { id, .. }
        | Op::SetStyle { id, .. }
        | Op::SetProp { id, .. }
        | Op::Listen { id, .. }
        | Op::Placeholder { id }
        | Op::Release { id }
        | Op::AttachSlot { child: id, .. }
        | Op::DetachSlot { child: id } => root(id),
        Op::Insert {
            parent,
            child,
            before,
        } => root(parent) || root(child) || before.as_ref().is_some_and(root),
        Op::Remove { parent, child } => root(parent) || root(child),
        Op::Command { id, command } => root(id) && !guest_root_command(command),
        Op::Rust(rust_op) => matches!(&rust_op.run, RustCall::Method(host, _) if root(host)),
        Op::RenderCanvas { .. } | Op::Animation(_) => false,
    }
}

impl<M: TextEngine> Core<M> {
    /// The context menu `items` of plugin `tenant`, with ids of their own
    /// at the platform; they replace the plugin context menu that showed
    /// (one shows at a time).
    pub(super) fn guest_menu_ids(&mut self, tenant: TenantId, items: &[MenuItem]) -> Vec<MenuItem> {
        self.tenants.guest_menu.clear();
        self.renumber_guest_menu(tenant, items)
    }

    fn renumber_guest_menu(&mut self, tenant: TenantId, items: &[MenuItem]) -> Vec<MenuItem> {
        items
            .iter()
            .map(|item| match item {
                MenuItem::Action(action) if action.id != 0 => {
                    let t = &mut self.tenants;
                    let id = GUEST_MENU_FIRST_ID + t.next_guest_menu;
                    t.next_guest_menu = (t.next_guest_menu + 1) % GUEST_MENU_FIRST_ID;
                    t.guest_menu.push((id, tenant, action.id));
                    let mut action = action.clone();
                    action.id = id;
                    MenuItem::Action(action)
                }
                MenuItem::Submenu { items, .. } => {
                    let renumbered = self.renumber_guest_menu(tenant, items);
                    let mut item = item.clone();
                    if let MenuItem::Submenu { items, .. } = &mut item {
                        *items = renumbered;
                    }
                    item
                }
                other => other.clone(),
            })
            .collect()
    }

    /// The plugin and its own id of the context menu item that reports as
    /// `id` at the platform; `None` for one of the shell's.
    pub fn guest_menu_item(&self, id: u32) -> Option<(TenantId, u32)> {
        self.tenants
            .guest_menu
            .iter()
            .find(|m| m.0 == id)
            .map(|&(_, tenant, own)| (tenant, own))
    }

    /// Sets the node + text-run quota for guest tenant `tenant`.
    pub fn set_tenant_max_nodes(&mut self, tenant: TenantId, max_nodes: usize) {
        if tenant == SHELL_TENANT {
            self.set_max_nodes(max_nodes);
        } else {
            self.tenants.quotas.insert(tenant, max_nodes);
        }
    }

    /// Live nodes and raw text runs currently owned by `tenant`.
    pub fn tenant_node_count(&self, tenant: TenantId) -> usize {
        if tenant == SHELL_TENANT {
            let guest_total: usize = self.tenants.counts.values().sum();
            (self.nodes.len() + self.runs.len()).saturating_sub(guest_total)
        } else {
            self.tenants.counts.get(&tenant).copied().unwrap_or(0)
        }
    }

    /// Which tenant owns `id` (node or folded text run); defaults to
    /// [`SHELL_TENANT`] when `id` is unknown or belongs to the shell.
    pub fn node_tenant(&self, id: NodeId) -> TenantId {
        if let Ok(key) = self.key(id) {
            return self.nodes[key as usize].tenant;
        }
        if let Some(run) = self.runs.get(&id) {
            return run.tenant;
        }
        SHELL_TENANT
    }

    /// Consumes and returns the tenant that started `call` (for `JsEvent::Reply`).
    pub fn take_call_tenant(&mut self, call: u32) -> TenantId {
        self.tenants
            .call_owners
            .remove(&call)
            .unwrap_or(SHELL_TENANT)
    }

    /// Consumes and returns the tenant that started `animation` (for
    /// `AnimationsFinished`).
    pub fn take_anim_tenant(&mut self, animation: AnimationId) -> TenantId {
        self.tenants
            .anim_owners
            .remove(&animation)
            .unwrap_or(SHELL_TENANT)
    }

    /// Applies one op produced by `tenant`, enforcing tenant isolation and
    /// recording call/animation ownership.
    pub fn apply_as(&mut self, tenant: TenantId, op: Op) -> Result<(), ApplyError> {
        if tenant != SHELL_TENANT {
            self.check_tenant_op(tenant, &op)?;
        }
        self.record_op_ownership(tenant, &op);
        let prev = self.tenants.current;
        self.tenants.current = tenant;
        let result = self.apply(op);
        self.tenants.current = prev;
        result
    }

    /// Validates that guest `tenant` only touches its own nodes and never
    /// mutates `ROOT_ID` or another tenant's subtree.
    fn check_tenant_op(&self, tenant: TenantId, op: &Op) -> Result<(), ApplyError> {
        let check_owned = |id: NodeId| -> Result<(), ApplyError> {
            if id == ROOT_ID {
                return Err(ApplyError::TenantViolation(id));
            }
            if let Ok(key) = self.key(id) {
                if self.nodes[key as usize].tenant != tenant {
                    return Err(ApplyError::TenantViolation(id));
                }
            } else if let Some(run) = self.runs.get(&id)
                && run.tenant != tenant
            {
                return Err(ApplyError::TenantViolation(id));
            }
            Ok(())
        };

        match op {
            Op::Create { id, .. } | Op::CreateText { id, .. } => {
                if *id == ROOT_ID {
                    return Err(ApplyError::TenantViolation(*id));
                }
            }
            Op::Evaluate { id, .. }
            | Op::Draw { id, .. }
            | Op::SetText { id, .. }
            | Op::SetStyle { id, .. }
            | Op::SetProp { id, .. }
            | Op::Listen { id, .. }
            | Op::Placeholder { id }
            | Op::Release { id }
            | Op::AttachSlot { child: id, .. }
            | Op::DetachSlot { child: id } => check_owned(*id)?,
            Op::Insert {
                parent,
                child,
                before,
            } => {
                check_owned(*parent)?;
                check_owned(*child)?;
                if let Some(b) = before {
                    check_owned(*b)?;
                }
            }
            Op::Remove { parent, child } => {
                check_owned(*parent)?;
                check_owned(*child)?;
            }
            Op::Command { id, command } => {
                if !(*id == ROOT_ID && guest_root_command(command)) {
                    check_owned(*id)?;
                }
            }
            Op::Rust(rust_op) => {
                if let RustCall::Method(host, _) = &rust_op.run {
                    check_owned(*host)?;
                }
            }
            Op::RenderCanvas { .. } | Op::Animation(_) => {}
        }
        Ok(())
    }

    /// Records call and animation ownership so async replies and animation
    /// completions route back to the originating tenant and teardown frees
    /// leaked animation graph nodes.
    fn record_op_ownership(&mut self, tenant: TenantId, op: &Op) {
        if tenant == SHELL_TENANT {
            return;
        }
        match op {
            Op::Evaluate { call, .. } | Op::RenderCanvas { call, .. } if *call != 0 => {
                self.tenants.call_owners.insert(*call, tenant);
            }
            Op::Rust(rust_op) if rust_op.call != 0 => {
                self.tenants.call_owners.insert(rust_op.call, tenant);
            }
            Op::Command {
                command: Command::Measure { call } | Command::MeasureScrollPosition { call },
                ..
            } if *call != 0 => {
                self.tenants.call_owners.insert(*call, tenant);
            }
            Op::Animation(anim) => match anim {
                AnimationOp::CreateValue { node, .. }
                | AnimationOp::CreateInterpolate { node, .. }
                | AnimationOp::CreateDiffClamp { node, .. }
                | AnimationOp::CreateMath { node, .. } => {
                    self.tenants
                        .anim_nodes
                        .entry(tenant)
                        .or_default()
                        .insert(*node);
                }
                AnimationOp::Release { node } => {
                    if let Some(set) = self.tenants.anim_nodes.get_mut(&tenant) {
                        set.remove(node);
                    }
                }
                AnimationOp::Start { animation, .. } => {
                    self.tenants.anim_owners.insert(*animation, tenant);
                }
                AnimationOp::StartComposite { animation, key, .. } => {
                    self.tenants.anim_owners.insert(*animation, tenant);
                    self.tenants
                        .anim_composites
                        .entry(tenant)
                        .or_default()
                        .insert(*key);
                }
                AnimationOp::ReleaseComposite { key } => {
                    if let Some(set) = self.tenants.anim_composites.get_mut(&tenant) {
                        set.remove(key);
                    }
                }
                AnimationOp::Read { request, .. } => {
                    self.tenants.anim_owners.insert(*request, tenant);
                }
                _ => {}
            },
            _ => {}
        }
    }

    /// Registers or clears `id` as the named slot target `name`.
    pub(super) fn set_slot_target(&mut self, id: NodeId, name: Option<Box<str>>) {
        let owner = self.node_tenant(id);
        // Remove previous target name on this node if it changed.
        if let Some(old_name) = self.tenants.node_slot_target.remove(&id) {
            if name.as_deref() == Some(old_name.as_ref()) {
                self.tenants.node_slot_target.insert(id, old_name);
                return;
            }
            if self.tenants.slot_targets.get(&old_name) == Some(&id) {
                self.tenants.slot_targets.remove(&old_name);
                self.detach_slot_mounts(&old_name);
            }
        }
        let Some(name) = name.filter(|s| !s.is_empty()) else {
            return;
        };
        let name: Arc<str> = Arc::from(name.as_ref());
        // Prevent a guest tenant from hijacking another tenant's live slot target.
        if let Some(&existing) = self.tenants.slot_targets.get(&name)
            && existing != id
        {
            let existing_owner = self.node_tenant(existing);
            if owner != SHELL_TENANT && existing_owner != owner {
                log::warn!(
                    "tenant {owner} attempted to hijack slotTarget `{name}` owned by tenant {existing_owner}"
                );
                return;
            }
            self.tenants.node_slot_target.remove(&existing);
            self.detach_slot_mounts(&name);
        }
        self.tenants.slot_targets.insert(name.clone(), id);
        self.tenants.node_slot_target.insert(id, name.clone());
        // Attach any plugin mounts waiting for this target name.
        self.attach_pending_slot_mounts(&name, id);
    }

    /// Attaches wrapper node `child` (owned by `tenant`) into named slot `name`.
    pub(super) fn attach_slot(
        &mut self,
        tenant: TenantId,
        name: Arc<str>,
        child: NodeId,
    ) -> Result<(), ApplyError> {
        let child_key = self.key(child)?;
        if self.nodes[child_key as usize].tenant != tenant {
            return Err(ApplyError::TenantViolation(child));
        }
        // Detach from any previous slot mount.
        if self.tenants.mounted_slots.contains_key(&child) {
            self.detach_slot(child);
        }
        self.tenants
            .slot_mounts
            .entry(name.clone())
            .or_default()
            .push((tenant, child));
        self.tenants
            .mounted_slots
            .insert(child, (tenant, name.clone()));
        if let Some(&target_id) = self.tenants.slot_targets.get(&name) {
            self.link_slot_child(target_id, child);
        }
        Ok(())
    }

    /// Unregisters and detaches slot wrapper node `child`.
    pub(super) fn detach_slot(&mut self, child: NodeId) {
        if let Some((_, name)) = self.tenants.mounted_slots.remove(&child)
            && let Some(list) = self.tenants.slot_mounts.get_mut(&name)
        {
            list.retain(|&(_, id)| id != child);
            if list.is_empty() {
                self.tenants.slot_mounts.remove(&name);
            }
        }
        if let Ok(key) = self.key(child)
            && self.nodes[key as usize].parent != NO_KEY
        {
            self.detach(key);
        }
    }

    /// Called right before `id` is removed from `Core::nodes` in `release`.
    /// Detaches any foreign slot mounts currently hosted inside `id` so they
    /// stay valid for re-attachment if a replacement target node mounts.
    pub(super) fn before_node_released(&mut self, id: NodeId) {
        if let Some(name) = self.tenants.node_slot_target.remove(&id)
            && self.tenants.slot_targets.get(&name) == Some(&id)
        {
            self.tenants.slot_targets.remove(&name);
            self.detach_slot_mounts(&name);
        }
        if let Some((_, name)) = self.tenants.mounted_slots.remove(&id)
            && let Some(list) = self.tenants.slot_mounts.get_mut(&name)
        {
            list.retain(|&(_, c)| c != id);
            if list.is_empty() {
                self.tenants.slot_mounts.remove(&name);
            }
        }
    }

    /// Links `child` into `target_id` if both are alive and acyclic.
    fn link_slot_child(&mut self, target_id: NodeId, child: NodeId) {
        let (Ok(target_key), Ok(child_key)) = (self.key(target_id), self.key(child)) else {
            return;
        };
        if self.is_ancestor(child_key, target_key) {
            log::warn!("slot mount cycle ignored: target={target_id} child={child}");
            return;
        }
        if self.nodes[child_key as usize].parent != NO_KEY {
            self.detach(child_key);
        }
        let _ = self.insert(target_id, child, None);
    }

    /// Attaches all registered mounts for `name` into `target_id`.
    fn attach_pending_slot_mounts(&mut self, name: &Arc<str>, target_id: NodeId) {
        let children: Vec<NodeId> = self
            .tenants
            .slot_mounts
            .get(name)
            .map(|list| list.iter().map(|&(_, id)| id).collect())
            .unwrap_or_default();
        for child in children {
            self.link_slot_child(target_id, child);
        }
    }

    /// Detaches all slot wrapper nodes currently mounted under `name` without
    /// dropping their registrations.
    fn detach_slot_mounts(&mut self, name: &Arc<str>) {
        let children: Vec<NodeId> = self
            .tenants
            .slot_mounts
            .get(name)
            .map(|list| list.iter().map(|&(_, id)| id).collect())
            .unwrap_or_default();
        for child in children {
            if let Ok(key) = self.key(child)
                && self.nodes[key as usize].parent != NO_KEY
            {
                self.detach(key);
            }
        }
    }

    /// Tears down every node, text run, Rust component instance, slot mount,
    /// animation node/composite and pending call belonging to guest `tenant`.
    ///
    /// Leaves the shell and every other tenant's subtree untouched.
    pub fn clear_tenant(&mut self, tenant: TenantId) {
        if tenant == SHELL_TENANT {
            return;
        }
        let _entered = self.components.runtime.enter();
        let prev = self.tenants.current;
        self.tenants.current = SHELL_TENANT;

        // 1. Detach foreign mounts hosted inside slot targets owned by `tenant`.
        let owned_targets: Vec<NodeId> = self
            .tenants
            .node_slot_target
            .keys()
            .copied()
            .filter(|&id| self.node_tenant(id) == tenant)
            .collect();
        for target_id in owned_targets {
            self.before_node_released(target_id);
        }

        // 2. Detach and unregister all slot wrappers owned by `tenant`.
        let owned_mounts: Vec<NodeId> = self
            .tenants
            .mounted_slots
            .iter()
            .filter_map(|(&child, &(t, _))| (t == tenant).then_some(child))
            .collect();
        for child in owned_mounts {
            self.detach_slot(child);
        }

        // 3. Release all Rust component instances whose host belongs to `tenant`
        //    (cleans up overlays, regions, virtual lists, Escape/press watchers,
        //    and cancels component async tasks).
        let component_hosts: Vec<NodeId> = self
            .components
            .instances
            .keys()
            .copied()
            .filter(|&host| self.node_tenant(host) == tenant)
            .collect();
        for host in component_hosts {
            self.release_instance(host);
        }

        // 4. Release every remaining node and folded text run owned by `tenant`.
        let node_ids: Vec<NodeId> = self
            .nodes
            .iter()
            .filter_map(|(_, n)| (n.tenant == tenant).then_some(n.id))
            .collect();
        for id in node_ids {
            let _ = self.apply(Op::Release { id });
        }
        let run_ids: Vec<NodeId> = self
            .runs
            .iter()
            .filter_map(|(&id, r)| (r.tenant == tenant).then_some(id))
            .collect();
        for id in run_ids {
            let _ = self.apply(Op::Release { id });
        }

        // 5. Release any native animation graph nodes and composites owned by `tenant`.
        if let Some(composites) = self.tenants.anim_composites.remove(&tenant) {
            for key in composites {
                let op = AnimationOp::ReleaseComposite { key };
                self.reach.apply(&op);
                self.mount.push(MountItem::Animation(op));
            }
        }
        if let Some(anim_nodes) = self.tenants.anim_nodes.remove(&tenant) {
            for node in anim_nodes {
                let op = AnimationOp::Release { node };
                self.reach.apply(&op);
                self.mount.push(MountItem::Animation(op));
            }
        }

        // 6. Forget pending call and animation reply routing and quotas.
        self.tenants.call_owners.retain(|_, &mut t| t != tenant);
        self.tenants.anim_owners.retain(|_, &mut t| t != tenant);
        self.tenants.counts.remove(&tenant);
        self.tenants.quotas.remove(&tenant);

        self.tenants.current = prev;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ApproxTextEngine, EventKind, Prop, Tag,
        geometry::{Points, Size},
    };

    fn make_core() -> Core<ApproxTextEngine> {
        Core::new(ApproxTextEngine, Size::new(Points(800.0), Points(600.0)))
    }

    #[test]
    fn guest_cannot_mutate_root_or_other_tenant() {
        let mut core = make_core();
        core.apply_as(
            SHELL_TENANT,
            Op::Create {
                id: 2,
                tag: Tag::View,
            },
        )
        .unwrap();
        core.apply_as(
            SHELL_TENANT,
            Op::Insert {
                parent: ROOT_ID,
                child: 2,
                before: None,
            },
        )
        .unwrap();

        let err = core
            .apply_as(
                1,
                Op::Insert {
                    parent: ROOT_ID,
                    child: 4,
                    before: None,
                },
            )
            .unwrap_err();
        assert_eq!(err, ApplyError::TenantViolation(ROOT_ID));
        assert!(err.is_fatal());

        let err = core
            .apply_as(
                1,
                Op::Listen {
                    id: 2,
                    event: EventKind::Press,
                    enabled: true,
                },
            )
            .unwrap_err();
        assert_eq!(err, ApplyError::TenantViolation(2));
    }

    #[test]
    fn cross_tenant_slot_attaches_in_either_order_and_clears_cleanly() {
        let mut core = make_core();
        // Plugin 1 mounts into "row:1" BEFORE shell creates the target.
        core.apply_as(
            1,
            Op::Create {
                id: 4,
                tag: Tag::View,
            },
        )
        .unwrap();
        core.apply_as(
            1,
            Op::AttachSlot {
                name: Arc::from("row:1"),
                child: 4,
            },
        )
        .unwrap();

        // Shell creates container node 2 with slotTarget = "row:1".
        core.apply_as(
            SHELL_TENANT,
            Op::Create {
                id: 2,
                tag: Tag::View,
            },
        )
        .unwrap();
        core.apply_as(
            SHELL_TENANT,
            Op::Insert {
                parent: ROOT_ID,
                child: 2,
                before: None,
            },
        )
        .unwrap();
        core.apply_as(
            SHELL_TENANT,
            Op::SetProp {
                id: 2,
                prop: Prop::SlotTarget(Some("row:1".into())),
            },
        )
        .unwrap();

        // Plugin 2 mounts into the SAME slot AFTER shell created it.
        core.apply_as(
            2,
            Op::Create {
                id: 6,
                tag: Tag::View,
            },
        )
        .unwrap();
        core.apply_as(
            2,
            Op::AttachSlot {
                name: Arc::from("row:1"),
                child: 6,
            },
        )
        .unwrap();

        let k2 = core.key(2).unwrap();
        assert_eq!(core.nodes[k2 as usize].children.len(), 2);
        assert_eq!(core.tenant_node_count(1), 1);
        assert_eq!(core.tenant_node_count(2), 1);

        // Clearing plugin 1 leaves shell and plugin 2 intact.
        core.clear_tenant(1);
        assert_eq!(core.tenant_node_count(1), 0);
        assert_eq!(core.tenant_node_count(2), 1);
        assert_eq!(core.nodes[k2 as usize].children.len(), 1);
        assert!(core.key(4).is_err());
        assert!(core.key(6).is_ok());

        // Clearing plugin 2 leaves shell target empty and intact.
        core.clear_tenant(2);
        assert_eq!(core.tenant_node_count(2), 0);
        assert_eq!(core.nodes[k2 as usize].children.len(), 0);
        assert!(core.key(2).is_ok());
    }

    #[test]
    fn guest_quota_is_enforced_per_tenant() {
        let mut core = make_core();
        core.set_tenant_max_nodes(1, 2);
        core.apply_as(
            1,
            Op::Create {
                id: 2,
                tag: Tag::View,
            },
        )
        .unwrap();
        core.apply_as(
            1,
            Op::Create {
                id: 4,
                tag: Tag::View,
            },
        )
        .unwrap();
        let err = core
            .apply_as(
                1,
                Op::Create {
                    id: 6,
                    tag: Tag::View,
                },
            )
            .unwrap_err();
        assert_eq!(err, ApplyError::TooManyNodes(2));

        // Shell and tenant 2 are unaffected.
        core.apply_as(
            2,
            Op::Create {
                id: 8,
                tag: Tag::View,
            },
        )
        .unwrap();
        assert_eq!(core.tenant_node_count(2), 1);
    }
}
