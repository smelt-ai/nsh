//! Layout animations of one node's change (`Command::LayoutAnimation` sent
//! to a node rather than the root, Rust's `ElementRef::animate_layout`): a
//! region that grows or shrinks (`nsh/base`'s reveals) animates the frames
//! that its change moves, and the rest of the commit (a list that updates,
//! live data) changes at once, as it would without the animation.
//!
//! The platforms animate the frames that follow a `MountItem::LayoutAnimation`
//! in the batch, until the next one (`MountItem::LayoutAnimation`'s docs),
//! so the scope is mounted data without a platform knowing about it: before
//! the batch is handed out, the frames are split into runs, each led by the
//! layout animation it takes (an empty one where it takes none).
//!
//! The frames a node's change moves: the node's own and those of its
//! subtree; those of the children of its ancestors (its siblings and theirs,
//! which a change of its size pushes along); and those of the children of a
//! view moved that way that changed size too (a `flex-1` sibling's content).
//! Frames are relative to the parent, so a view that only moved does not
//! move its children. A view that gained or lost a child in that commit
//! does not carry the change into its content: its children moved because
//! of that (a list that updates under a header that collapses), not
//! because of the scoped node. Insertions and removals are not scoped: a scoped
//! animation animates frames only (its `update`). A commit that also
//! animates its whole layout (a layout animation at the root) animates
//! every frame with that one.
//!
//! A view that follows another one (`follow.rs`) moving to another element
//! scopes its own frame alone (and its subtree's): what changes around it
//! in that commit is not its doing.
//!
//! A plugin's layout animation at the root (`LayoutAnimation.configureNext`
//! in a guest VM, `tenant.rs`) animates the plugin's own views, all of
//! their changes: frames, insertions and removals, as the shell's animates
//! every view. The other tenants' views change as they would without it.
//! It leads the batch as the shell's would, so lists move their rows
//! under it (`list_motion.rs`); then each animated item takes its owner's
//! animation, or none. The shell's replaces the plugins' in its batch.

use super::{Core, Key, NO_KEY, tenant::TenantId};
use crate::{LayoutAnimation, LayoutTransition, MountItem, NodeId, text::TextEngine};

/// A node whose change animates the frames it moves, with `transition`.
struct Scope {
    key: Key,
    /// Its ancestors, nearest first; none when it animates its own frame
    /// alone.
    ancestors: Vec<Key>,
    transition: LayoutTransition,
}

impl<M: TextEngine> Core<M> {
    /// Node `id`'s change animates the frames it moves in the next commit
    /// with `transition`.
    pub(super) fn scope_layout_animation(&mut self, id: NodeId, transition: LayoutTransition) {
        self.scoped_layout.push((id, transition, false));
    }

    /// Node `id`'s own frame (and its subtree's) animates in the next
    /// commit with `transition`, and nothing it moves.
    pub(super) fn scope_frame_animation(&mut self, id: NodeId, transition: LayoutTransition) {
        self.scoped_layout.push((id, transition, true));
    }

    /// The frame of `key` changes in this commit, and its size with it when
    /// `resized`: noted while a scoped layout animation waits.
    #[inline]
    pub(super) fn note_resized(&mut self, key: Key, resized: bool) {
        if resized && !self.scoped_layout.is_empty() {
            self.resized.push(key);
        }
    }

    /// Node `key` lost a child in this commit: noted while a scoped layout
    /// animation waits.
    #[inline]
    pub(super) fn note_restructured(&mut self, key: Key) {
        if !self.scoped_layout.is_empty() {
            self.restructured.push(key);
        }
    }

    /// The plugins' layout animations of the next commit join this
    /// batch's, unless the shell's leads it: what leads the batch, all of
    /// their phases (each item takes its owner's in
    /// [`Core::split_scoped_frames`]).
    pub(super) fn lead_guest_layout(&mut self) -> Option<LayoutAnimation> {
        if self.shell_layout_batch {
            self.guest_layout.clear();
            return None;
        }
        for (tenant, animation) in self.guest_layout.drain(..) {
            match self.guest_layout_batch.iter_mut().find(|g| g.0 == tenant) {
                Some(g) => g.1 = animation,
                None => self.guest_layout_batch.push((tenant, animation)),
            }
        }
        let lead = self
            .guest_layout_batch
            .iter()
            .fold(LayoutAnimation::default(), |a, &(_, g)| LayoutAnimation {
                create: a.create.or(g.create),
                update: a.update.or(g.update),
                delete: a.delete.or(g.delete),
            });
        Some(lead)
    }

    /// Leads each run of animated items in the batch with the layout
    /// animation it takes: a plugin's for its own views, a scope's for the
    /// frames its node moves, an empty one for the others. Nothing while
    /// neither waits, or when the shell animates all of the batch's layout.
    pub(super) fn split_scoped_frames(&mut self) {
        let guests = std::mem::take(&mut self.guest_layout_batch);
        let mut deleted = std::mem::take(&mut self.guest_deleted);
        self.shell_layout_batch = false;
        if self.scoped_layout.is_empty() && guests.is_empty() {
            deleted.clear();
            self.guest_deleted = deleted;
            return;
        }
        let requested = std::mem::take(&mut self.scoped_layout);
        let mut resized = std::mem::take(&mut self.resized);
        let mut restructured = std::mem::take(&mut self.restructured);
        if guests.is_empty()
            && self
                .mount
                .iter()
                .any(|m| matches!(m, MountItem::LayoutAnimation(_)))
        {
            resized.clear();
            self.resized = resized;
            restructured.clear();
            self.restructured = restructured;
            return;
        }
        let scopes: Vec<Scope> = requested
            .into_iter()
            .filter_map(|(id, transition, own)| {
                let key = self.key(id).ok()?;
                let mut ancestors = Vec::new();
                let mut k = self.nodes[key as usize].parent;
                while !own && k != NO_KEY {
                    ancestors.push(k);
                    k = self.nodes[k as usize].parent;
                }
                Some(Scope {
                    key,
                    ancestors,
                    transition,
                })
            })
            .collect();
        // A view whose children changed does not carry a size change into
        // them: those that gained or lost a child are dropped from `resized`.
        for item in &self.mount {
            if let MountItem::Insert { parent, .. } = item
                && let Ok(key) = self.key(*parent)
            {
                restructured.push(key);
            }
        }
        restructured.sort_unstable();
        restructured.dedup();
        resized.retain(|k| restructured.binary_search(k).is_err());
        resized.sort_unstable();
        resized.dedup();
        if !scopes.is_empty() || !guests.is_empty() {
            let items = std::mem::take(&mut self.mount);
            let mut out = Vec::with_capacity(items.len() + 2);
            // What the batch animates the next item with, before the
            // split: the plugins' lead, or none where a list's rows change
            // at once (`list_motion.rs`).
            let mut lead = LayoutAnimation::default();
            // What the platform animates the next item with.
            let mut current = LayoutAnimation::default();
            for item in items {
                let wanted = match item {
                    MountItem::LayoutAnimation(a) => {
                        lead = a;
                        continue;
                    }
                    MountItem::Frame { id, .. } => {
                        let guest = self.guest_layout_of(&guests, &deleted, id, lead);
                        Some(guest.unwrap_or_else(|| LayoutAnimation {
                            update: self.key(id).ok().and_then(|key| {
                                scopes
                                    .iter()
                                    .find(|s| self.moves(s, key, &resized))
                                    .map(|s| s.transition)
                            }),
                            ..LayoutAnimation::default()
                        }))
                    }
                    MountItem::Insert { child: id, .. }
                    | MountItem::Remove { id }
                    | MountItem::Delete { id }
                        if !guests.is_empty() =>
                    {
                        Some(
                            self.guest_layout_of(&guests, &deleted, id, lead)
                                .unwrap_or_default(),
                        )
                    }
                    _ => None,
                };
                if let Some(wanted) = wanted
                    && wanted != current
                {
                    out.push(MountItem::LayoutAnimation(wanted));
                    current = wanted;
                }
                out.push(item);
            }
            self.mount = out;
        }
        deleted.clear();
        self.guest_deleted = deleted;
        resized.clear();
        self.resized = resized;
        restructured.clear();
        self.restructured = restructured;
    }

    /// The layout animation of view `id` when it is a plugin's that
    /// animates its own (none where the batch's `lead` changes it at once);
    /// `None` when it is not.
    fn guest_layout_of(
        &self,
        guests: &[(TenantId, LayoutAnimation)],
        deleted: &[(NodeId, TenantId)],
        id: NodeId,
        lead: LayoutAnimation,
    ) -> Option<LayoutAnimation> {
        if guests.is_empty() {
            return None;
        }
        let tenant = deleted
            .iter()
            .find(|d| d.0 == id)
            .map_or_else(|| self.node_tenant(id), |d| d.1);
        let &(_, animation) = guests.iter().find(|g| g.0 == tenant)?;
        Some(if lead == LayoutAnimation::default() {
            lead
        } else {
            animation
        })
    }

    /// Whether `scope`'s change moves the frame of `key`: `key` is in its
    /// subtree, or reached from one of its ancestors' children through
    /// views that changed size and kept their children.
    fn moves(&self, scope: &Scope, key: Key, resized: &[Key]) -> bool {
        let mut k = key;
        // Every view between `key` and `k` changed size.
        let mut carried = true;
        loop {
            if k == scope.key {
                return true;
            }
            let parent = self.nodes[k as usize].parent;
            if parent == NO_KEY {
                return false;
            }
            if carried && scope.ancestors.contains(&parent) {
                return true;
            }
            carried = carried && resized.binary_search(&parent).is_ok();
            k = parent;
        }
    }
}
