//! Guest plugin VMs on one surface: views a plugin mounts into the shell's
//! slots (`mountSlot` / `slotTarget`) lay out, stack, take focus, compose
//! text and show in the accessibility tree as one window, and a popover a
//! plugin opens leaves the clip of the slot it is in.
//!
//! Isolation (hangs, memory, prototype hardening) is covered where it is
//! enforced: `nsh-engine`'s plugin tests and `nsh-core`'s tenant quotas.

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use gpui::{Entity, Focusable, Modifiers, TestAppContext, VisualTestContext, point, px, size};
use nsh_core::access::Capabilities;
use nsh_engine::{HostChannel, PluginConfig, Policy};

use crate::{App, AppOptions, AppView, Bundle, view::NshView};

type Posts = Arc<Mutex<Vec<(String, Instant)>>>;

fn inline_options() -> AppOptions {
    AppOptions {
        inline_root: true,
        ..AppOptions::default()
    }
}

fn bundle_tsx(code: &str, harden: bool, host_names: &[&str]) -> Bundle {
    let sets = [nsh_base::registry().into()];
    let compiled = nsh_bundler::compile_module("App.tsx", code, &sets).expect("compile TSX");
    let mut policy = if harden {
        Policy::default()
    } else {
        Policy::development()
    };
    let mut declared = Capabilities::all();
    declared.host = host_names.iter().map(|s| (*s).to_owned()).collect();
    policy.declared = Some(declared);
    Bundle::source(compiled).with_policy(policy)
}

fn attach_probe(host: &HostChannel, posts: &Posts) {
    let posts = posts.clone();
    host.handle("probe", move |call| {
        let msg = call.data().as_str().unwrap_or_default().to_owned();
        posts.lock().unwrap().push((msg, Instant::now()));
        call.resolve(serde_json::Value::Null);
    });
}

fn turn(cx: &mut VisualTestContext) {
    cx.run_until_parked();
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    // 轮询例外：引擎在独立线程上运行，离屏测试驱动没有平台通知事件可等；周期 4ms，上限由各调用方控制。
    std::thread::sleep(Duration::from_millis(4));
}

fn settle(cx: &mut VisualTestContext, dur: Duration) {
    let end = Instant::now() + dur;
    while Instant::now() < end {
        turn(cx);
    }
}

fn wait_probe(
    cx: &mut VisualTestContext,
    posts: &Posts,
    what: &str,
    since: Instant,
    timeout_secs: f64,
) -> Option<f64> {
    while since.elapsed().as_secs_f64() < timeout_secs {
        if let Some((_, at)) = posts
            .lock()
            .unwrap()
            .iter()
            .find(|(name, at)| name == what && *at >= since)
        {
            return Some(at.duration_since(since).as_secs_f64() * 1e3);
        }
        turn(cx);
    }
    None
}

fn find_by_test_id(node: &NshView, target: &str) -> Option<NshView> {
    if node.state().props.test_id.as_deref() == Some(target) {
        return Some(node.clone());
    }
    for child in node.child_views() {
        if let Some(found) = find_by_test_id(&child, target) {
            return Some(found);
        }
    }
    None
}

fn view_frame(
    view: &Entity<AppView>,
    cx: &mut VisualTestContext,
    test_id: &str,
) -> Option<(f32, f32, f32, f32)> {
    view.read_with(cx, |v, _| {
        let node = find_by_test_id(&v.surface().root, test_id)?;
        let f = node.state().frame;
        Some((f.origin.x.0, f.origin.y.0, f.size.width.0, f.size.height.0))
    })
}

fn view_window_center(
    view: &Entity<AppView>,
    cx: &mut VisualTestContext,
    test_id: &str,
) -> Option<(f32, f32)> {
    view.read_with(cx, |v, _| {
        let node = find_by_test_id(&v.surface().root, test_id)?;
        let (w, h) = {
            let s = node.state();
            (s.frame.size.width.0, s.frame.size.height.0)
        };
        let mut x = w * 0.5;
        let mut y = h * 0.5;
        let mut cur = node;
        while let Some(parent) = cur.parent_view() {
            let (ox, oy) = {
                let cs = cur.state();
                parent.state().child_origin(cs.frame, cs.shift)
            };
            x += ox;
            y += oy;
            cur = parent;
        }
        let (sx, sy) = v.surface().origin.get();
        Some((x + sx, y + sy))
    })
}

fn wait_until<T>(
    cx: &mut VisualTestContext,
    timeout: Duration,
    what: &str,
    mut check: impl FnMut(&mut VisualTestContext) -> Option<T>,
) -> T {
    let start = Instant::now();
    loop {
        if let Some(val) = check(cx) {
            return val;
        }
        assert!(start.elapsed() < timeout, "timed out waiting for: {what}");
        turn(cx);
    }
}

fn focused_field_text(view: &Entity<AppView>, cx: &mut VisualTestContext) -> Option<String> {
    view.read_with(cx, |v, _| {
        crate::text_input::focused(v.surface())
            .and_then(|field| crate::text_input::accessible_value(&field))
            .map(|fv| fv.text)
    })
}

fn send_tab(view: &Entity<AppView>, cx: &mut VisualTestContext, shift: bool) {
    view.update(cx, |v, _| {
        let modifiers = crate::input::Modifiers {
            shift,
            ..crate::input::Modifiers::default()
        };
        let event = crate::input::KeyEvent {
            vk: crate::keys::VK_TAB,
            down: true,
            repeat: false,
            modifiers,
        };
        let surface = v.surface().clone();
        assert!(crate::input::key(&surface, event));
    });
}

fn build_a11y_tree(view: &Entity<AppView>, cx: &mut VisualTestContext) -> (usize, Vec<String>) {
    view.read_with(cx, |v, _| {
        let s = v.surface();
        let built = crate::access::build(
            s.key(),
            &s.root,
            s.origin.get(),
            s.scale(),
            &crate::text_input::accessible_value,
        );
        built.summary()
    })
}

pub(crate) fn guest_plugins(cx: &mut TestAppContext) {
    cx.executor().allow_parking();
    crate::set_ui_sets(vec![nsh_base::registry()]);

    a_hundred_plugin_rows_resize_and_share_their_slots(cx);
    a_plugin_popover_leaves_its_slot_clip_and_joins_focus_ime_and_a11y(cx);
}

// -----------------------------------------------------------------------------
// 100 dynamic-height rows mounted into slots; two plugins in the same slot
// -----------------------------------------------------------------------------
fn a_hundred_plugin_rows_resize_and_share_their_slots(cx: &mut TestAppContext) {
    let posts = Posts::default();
    let (view, cx) = cx.add_window_view(|window, cx| AppView::new(inline_options(), window, cx));
    cx.simulate_resize(size(px(800.0), px(600.0)));

    let shell_tsx = r#"
import { host, mount, Scroll, Text, View } from "nsh";

const ROWS = Array.from({ length: 100 }, (_, i) => i);

function Shell() {
  host.post("probe", "shell.ready");
  return (
    <Scroll style={{ flex: 1 }}>
      {ROWS.map((i) => (
        <View
          testID={`row-${i}`}
          style={{ flexDirection: "column", width: 760, paddingTop: 2, paddingBottom: 2 }}
        >
          <View style={{ height: 16 }}>
            <Text>{`Session ${i}`}</Text>
          </View>
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

host.on("toggle", (data: any) => {
  const idx = data.row as number;
  expanded[idx].set(!expanded[idx]());
  host.post("probe", `pa.toggled.${idx}`);
});

host.post("probe", "pa.ready");
"##;

    let plugin_b_tsx = r##"
import { host, mountSlot, Text, View } from "nsh";

mountSlot("row:42", () => (
  <View
    testID="pb-badge-42"
    onPress={() => host.post("probe", "pb.click.42")}
    style={{ width: 80, height: 20, backgroundColor: "#664422" }}
  >
    <Text>Git</Text>
  </View>
));
host.post("probe", "pb.ready");
"##;

    let shell_host = view.read_with(cx, |v, _| v.host());
    attach_probe(&shell_host, &posts);
    // Taken before the load: the engine thread may post before this thread
    // gets back from it.
    let t_load = Instant::now();
    view.update(cx, |v, _| {
        v.on_app_error(|e| eprintln!("plugin rows: {e}"));
        v.load(App::Js(bundle_tsx(shell_tsx, true, &["probe"])));
        // Disable scroll content culling so all 100 row slots mount at once
        // and we can assert exact layout frames for row-0 through row-99.
        v.surface().control(|c| c.set_content_culling(false));
    });
    assert!(
        wait_probe(cx, &posts, "shell.ready", t_load, 5.0).is_some(),
        "shell ready"
    );

    let host_a = HostChannel::new();
    attach_probe(&host_a, &posts);
    let host_b = HostChannel::new();
    attach_probe(&host_b, &posts);

    let t0 = Instant::now();
    view.update(cx, |v, _| {
        v.load_plugin(
            PluginConfig::new("plugin-a", bundle_tsx(plugin_a_tsx, true, &["probe"]))
                .with_host(host_a.clone()),
        );
        v.load_plugin(
            PluginConfig::new("plugin-b", bundle_tsx(plugin_b_tsx, true, &["probe"]))
                .with_host(host_b.clone()),
        );
    });
    assert!(
        wait_probe(cx, &posts, "pa.ready", t0, 5.0).is_some(),
        "plugin-a ready"
    );
    assert!(
        wait_probe(cx, &posts, "pb.ready", t0, 5.0).is_some(),
        "plugin-b ready"
    );

    // Verify all 100 rows mounted and have expected initial height (16 + 20 + 4 = 40px).
    let (r42_before, r43_before, r99_before) = wait_until(
        cx,
        Duration::from_secs(5),
        "all 100 row slots laid out",
        |cx| {
            let f0 = view_frame(&view, cx, "row-0")?;
            let f42 = view_frame(&view, cx, "row-42")?;
            let f43 = view_frame(&view, cx, "row-43")?;
            let f99 = view_frame(&view, cx, "row-99")?;
            let _pb = view_frame(&view, cx, "pb-badge-42")?;
            if (f0.3 - 40.0).abs() < 0.5 && (f42.3 - 40.0).abs() < 0.5 && (f99.3 - 40.0).abs() < 0.5
            {
                Some((f42, f43, f99))
            } else {
                None
            }
        },
    );

    // Toggle row 42 from 20px -> 76px (+56px) and confirm row 42 and all subsequent rows shift atomically.
    let t_toggle = Instant::now();
    host_a
        .send("toggle", serde_json::json!({ "row": 42 }))
        .unwrap();
    assert!(
        wait_probe(cx, &posts, "pa.toggled.42", t_toggle, 5.0).is_some(),
        "row 42 toggled"
    );
    let (r42_after, r43_after, r99_after) = wait_until(
        cx,
        Duration::from_secs(5),
        "row 42 expanded by +56px",
        |cx| {
            let f42 = view_frame(&view, cx, "row-42")?;
            let f43 = view_frame(&view, cx, "row-43")?;
            let f99 = view_frame(&view, cx, "row-99")?;
            if (f42.3 - (r42_before.3 + 56.0)).abs() < 0.5 {
                Some((f42, f43, f99))
            } else {
                None
            }
        },
    );
    assert!(
        ((r43_after.1 - r43_before.1) - 56.0).abs() < 0.5,
        "row 43 y shift mismatch: before={} after={}",
        r43_before.1,
        r43_after.1
    );
    assert!(
        ((r99_after.1 - r99_before.1) - 56.0).abs() < 0.5,
        "row 99 y shift mismatch: before={} after={}",
        r99_before.1,
        r99_after.1
    );

    // Click row 0's plugin-a badge (visible in viewport) and verify click routing.
    let (bx, by) = view_window_center(&view, cx, "pa-badge-0").expect("pa-badge-0 center");
    let t_click = Instant::now();
    cx.simulate_click(point(px(bx), px(by)), Modifiers::default());
    assert!(
        wait_probe(cx, &posts, "pa.click.0", t_click, 3.0).is_some(),
        "click routed to plugin-a in row 0"
    );

    // Unload plugin-b and confirm plugin-a's badge in the SAME slot ("row:42") remains intact!
    view.update(cx, |v, _| v.unload_plugin("plugin-b"));
    wait_until(cx, Duration::from_secs(5), "plugin-b removed", |cx| {
        view_frame(&view, cx, "pb-badge-42").is_none().then_some(())
    });
    assert!(
        view_frame(&view, cx, "pa-badge-42").is_some(),
        "plugin-a badge in shared slot survived plugin-b unload"
    );

    println!(
        "100 dynamic-height row slots mounted; row-42 h {:.0}->{:.0}px, row-43 dy={:+.0}px, row-99 dy={:+.0}px, multi-plugin slot coexistence verified",
        r42_before.3,
        r42_after.3,
        r43_after.1 - r43_before.1,
        r99_after.1 - r99_before.1
    );
    view.update(cx, |v, _| v.stop());
    cx.update(|window, _| window.remove_window());
    cx.run_until_parked();
}

// -----------------------------------------------------------------------------
// A plugin Popover leaving overflow: hidden; Tab focus, IME and a11y across VMs
// -----------------------------------------------------------------------------
fn a_plugin_popover_leaves_its_slot_clip_and_joins_focus_ime_and_a11y(cx: &mut TestAppContext) {
    let posts = Posts::default();
    let (view, cx) = cx.add_window_view(|window, cx| AppView::new(inline_options(), window, cx));
    cx.simulate_resize(size(px(800.0), px(600.0)));

    let shell_tsx = r##"
import { host, mount, Text, View } from "nsh";
import { Input } from "nsh/base";

function Shell() {
  host.post("probe", "shell.up");
  return (
    <View style={{ flex: 1, padding: 12, gap: 12 }}>
      <Text>ShellHeader</Text>
      <Input autoFocus value="shell-before" style={{ width: 220, height: 30 }} />
      {/* Tiny clipped 40x40 container: plugin Popover MUST escape this clip */}
      <View
        slotTarget="pop-slot"
        style={{ width: 40, height: 40, overflow: "hidden", backgroundColor: "#112233" }}
      />
      <Input value="shell-after" style={{ width: 220, height: 30 }} />
      <View
        testID="shell-outside"
        onPress={() => host.post("probe", "shell.outside.click")}
        style={{ position: "absolute", left: 500, top: 400, width: 200, height: 120, backgroundColor: "#223322" }}
      >
        <Text>OutsideShell</Text>
      </View>
    </View>
  );
}
mount(Shell);
"##;

    let plugin_tsx = r##"
import { host, mountSlot, signal, Text, View } from "nsh";
import { Input, Popover } from "nsh/base";

mountSlot("pop-slot", () => {
  const open = signal(true);
  const text = signal("plugin-input");
  const imeLog: string[] = [];
  return (
    <Popover
      open={open()}
      onOpenChange={(e: any) => {
        open.set(e.open);
        host.post("probe", `pop.open.${e.open}.${e.reason}`);
      }}
      trigger={
        <View testID="pop-trigger" style={{ width: 36, height: 36, backgroundColor: "#445566" }}>
          <Text>T</Text>
        </View>
      }
      popupStyle={{ width: 240, padding: 8, backgroundColor: "#333344" }}
    >
      <View style={{ gap: 6 }}>
        <View
          testID="pop-action"
          onPress={() => host.post("probe", "pop.action.click")}
          style={{ width: 200, height: 40, backgroundColor: "#556688" }}
        >
          <Text>PluginAction</Text>
        </View>
        <Input
          value={text()}
          onChange={({ value }) => {
            text.set(value);
            host.post("probe", `pop.change.${value}`);
          }}
          onCompositionChange={(e) => {
            imeLog.push(e.composing ? `c:${e.text}` : "end");
            host.post("probe", `pop.ime.${imeLog.join("|")}`);
          }}
          style={{ width: 200, height: 30 }}
        />
      </View>
    </Popover>
  );
});
host.post("probe", "pop.ready");
"##;

    let shell_host = view.read_with(cx, |v, _| v.host());
    attach_probe(&shell_host, &posts);
    let t_load = Instant::now();
    view.update(cx, |v, _| {
        v.on_app_error(|e| eprintln!("plugin popover: {e}"));
        v.load(App::Js(bundle_tsx(shell_tsx, true, &["probe"])));
    });
    cx.update(|window, cx| window.focus(&view.focus_handle(cx), cx));
    assert!(wait_probe(cx, &posts, "shell.up", t_load, 5.0).is_some());

    let plugin_host = HostChannel::new();
    attach_probe(&plugin_host, &posts);
    let t0 = Instant::now();
    view.update(cx, |v, _| {
        v.load_plugin(
            PluginConfig::new("pop-plugin", bundle_tsx(plugin_tsx, true, &["probe"]))
                .with_host(plugin_host.clone()),
        );
    });
    assert!(wait_probe(cx, &posts, "pop.ready", t0, 5.0).is_some());

    // Wait for initial focus on shell-before and for Popover action button to lay out outside the 40x40 box.
    wait_until(cx, Duration::from_secs(5), "shell-before focused", |cx| {
        (focused_field_text(&view, cx).as_deref() == Some("shell-before")).then_some(())
    });
    let (ax, ay) = wait_until(cx, Duration::from_secs(5), "pop-action laid out", |cx| {
        view_window_center(&view, cx, "pop-action")
    });
    // Verify the button center lies OUTSIDE the 40x40 clipped slot (which sits near x=[12..52], y=[54..94]).
    let (tx, ty) = view_window_center(&view, cx, "pop-trigger").expect("trigger center");
    assert!(
        ax > tx + 40.0 || ay > ty + 30.0,
        "popover action ({ax:.1}, {ay:.1}) did not escape clipped trigger ({tx:.1}, {ty:.1})"
    );

    // Unified AccessKit tree must include BOTH shell labels and plugin Popover labels!
    let (a11y_count, a11y_labels) = build_a11y_tree(&view, cx);
    assert!(
        a11y_labels.iter().any(|l| l.contains("ShellHeader")),
        "AccessKit missing ShellHeader: {a11y_labels:?}"
    );
    assert!(
        a11y_labels.iter().any(|l| l.contains("PluginAction")),
        "AccessKit missing PluginAction from guest Popover: {a11y_labels:?}"
    );

    // Unified Tab order: shell-before -> plugin trigger -> (optional pop-action) -> plugin Popover Input -> shell-after!
    send_tab(&view, cx, false); // -> Popover trigger
    wait_until(cx, Duration::from_secs(3), "focus on pop trigger", |cx| {
        view.read_with(cx, |v, _| {
            crate::key_focus::focused(v.surface())
                .is_some_and(|n| n.kind() != nsh_core::ViewKind::Input)
                && crate::text_input::focused(v.surface()).is_none()
        })
        .then_some(())
    });
    // Advance Tab into the Popover's Input (may pass through pressable `pop-action` first).
    let mut tabs_to_plugin_input = 0;
    loop {
        tabs_to_plugin_input += 1;
        assert!(
            tabs_to_plugin_input <= 4,
            "Tab did not reach plugin Popover Input within 4 presses"
        );
        send_tab(&view, cx, false);
        settle(cx, Duration::from_millis(40));
        if focused_field_text(&view, cx).as_deref() == Some("plugin-input") {
            break;
        }
    }

    // Chinese IME composition & commit directly inside the guest plugin's Popover <Input>!
    cx.simulate_keystrokes("end");
    let t_ime = Instant::now();
    view.update(cx, |v, _| crate::text_input::mark(v.surface(), "ni"));
    assert!(
        wait_probe(cx, &posts, "pop.ime.c:ni", t_ime, 3.0).is_some(),
        "plugin Input received IME composition start"
    );
    view.update(cx, |v, _| {
        crate::text_input::char_input(v.surface(), "你好")
    });
    assert!(
        wait_probe(cx, &posts, "pop.change.plugin-input你好", t_ime, 3.0).is_some(),
        "plugin Input received committed Chinese IME text"
    );

    // Continue Tab -> shell-after, then Shift+Tab -> back into plugin Popover Input!
    send_tab(&view, cx, false);
    wait_until(cx, Duration::from_secs(3), "focus on shell-after", |cx| {
        (focused_field_text(&view, cx).as_deref() == Some("shell-after")).then_some(())
    });
    send_tab(&view, cx, true);
    wait_until(
        cx,
        Duration::from_secs(3),
        "shift-tab back to plugin input",
        |cx| (focused_field_text(&view, cx).as_deref() == Some("plugin-input你好")).then_some(()),
    );

    // Click inside the plugin Popover (outside the 40x40 clipped container!) and verify delivery.
    let t_pop = Instant::now();
    cx.simulate_click(point(px(ax), px(ay)), Modifiers::default());
    assert!(
        wait_probe(cx, &posts, "pop.action.click", t_pop, 3.0).is_some(),
        "click inside escaped plugin Popover must reach plugin"
    );

    // Click outside on Shell background -> Popover closes with pressOutside AND Shell receives click!
    let (ox, oy) = view_window_center(&view, cx, "shell-outside").expect("shell-outside center");
    let t_out = Instant::now();
    cx.simulate_click(point(px(ox), px(oy)), Modifiers::default());
    assert!(
        wait_probe(cx, &posts, "pop.open.false.pressOutside", t_out, 3.0).is_some(),
        "clicking shell outside must close plugin Popover"
    );
    assert!(
        wait_probe(cx, &posts, "shell.outside.click", t_out, 3.0).is_some(),
        "click outside on desktop must pass through to shell"
    );

    println!(
        "cross-plugin Popover escaped 40x40 clip (action at {ax:.0},{ay:.0}), Tab/Shift-Tab crossed Shell<->Plugin<->Popover, Chinese IME committed '你好', pressOutside closed Popover + reached Shell, unified AccessKit built {a11y_count} nodes"
    );
    view.update(cx, |v, _| v.stop());
    cx.update(|window, _| window.remove_window());
    cx.run_until_parked();
}
