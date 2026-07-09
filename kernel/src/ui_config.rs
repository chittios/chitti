//! UI / shortcuts configuration, persisted as JSON under `/configs/core/` in the
//! Synapse store (`ui.json`, `shortcuts.json`) so it is durable on an installed
//! system and editable from the shell (`/ui`, `/shortcuts`) or the `/open`
//! editor. The UI config drives the framebuffer layout (pane split, font scale,
//! pane placement, titles) and the status-bar text templates; the status
//! templates support `${var}` substitution against the clock and system info.

use crate::json::Json;
use crate::mm::Locked;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const UI_PATH: &str = "/configs/core/ui.json";
const SHORTCUTS_PATH: &str = "/configs/core/shortcuts.json";

#[cfg(target_arch = "x86_64")]
const ARCH: &str = "x86_64";
#[cfg(target_arch = "aarch64")]
const ARCH: &str = "aarch64";
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const ARCH: &str = "?";

/// The Chitti brand colour palette (see DESIGN.md). Written into `ui.json`'s
/// `theme` object so every colour is discoverable + overridable; the framebuffer
/// applies them over its brand-dark default via `theme_from_pairs`.
const THEME_DEFAULTS: &[(&str, &str)] = &[
    ("accent", "#cc785c"),       // primary — active border, caret, brand, logo
    ("screen_bg", "#181715"),    // surface-dark
    ("chat_bg", "#1f1e1b"),      // surface-dark-soft
    ("logs_bg", "#141311"),
    ("chat_fg", "#faf9f5"),      // cream / on-dark
    ("logs_fg", "#a09d96"),      // on-dark-soft
    ("border_dim", "#3a3733"),
    ("title_active", "#cc785c"),
    ("title_dim", "#6c6a64"),    // muted
    ("sep_dim", "#2a2825"),
    ("status_bg", "#252320"),    // surface-dark-elevated
    ("status_fg", "#a09d96"),
    ("editor_bg", "#1f1e1b"),
    ("editor_fg", "#faf9f5"),
    ("editor_lineno", "#6c6a64"),
    ("editor_sel", "#5a3a2e"),   // terracotta-tinted selection
    ("composer_bg", "#252320"),  // elevated input box (Grok-style)
    ("composer_border", "#3a3733"),
    ("composer_hint", "#6c6a64"),
];

/// The persisted UI configuration.
#[derive(Clone)]
pub struct UiConfig {
    pub chat_pct: u64,
    pub font_scale: u64, // 0 = auto
    pub swap_panes: bool,
    pub chat_title: String,
    pub logs_title: String,
    pub status_left: String,  // template
    pub status_right: String, // template
    pub tz_offset: i32,       // seconds east of UTC
    pub splash: bool,         // show the boot splash (logo + wordmark)
    /// Colour palette as `(name, "#rrggbb")` pairs (kept as strings so the config
    /// layer stays independent of the framebuffer, which is absent in test builds).
    pub theme: Vec<(String, String)>,
}

impl Default for UiConfig {
    fn default() -> Self {
        UiConfig {
            chat_pct: 56,
            font_scale: 0,
            swap_panes: false,
            chat_title: "Shell Agent".to_string(),
            logs_title: "ktrace".to_string(),
            status_left: "ChittiOS v${version}".to_string(),
            status_right: "${kbd} ${mouse}  ${net}  ${mem}  ${cpu} ${cores}  ${datetime} ${tz}".to_string(),
            tz_offset: 0,
            splash: true,
            theme: THEME_DEFAULTS.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        }
    }
}

impl UiConfig {
    fn to_json(&self) -> Json {
        let theme = Json::Obj(self.theme.iter().map(|(k, v)| (k.clone(), Json::Str(v.clone()))).collect());
        Json::Obj(alloc::vec![
            ("chat_pct".to_string(), Json::Num(self.chat_pct as f64)),
            ("font_scale".to_string(), Json::Num(self.font_scale as f64)),
            ("swap_panes".to_string(), Json::Bool(self.swap_panes)),
            ("chat_title".to_string(), Json::Str(self.chat_title.clone())),
            ("logs_title".to_string(), Json::Str(self.logs_title.clone())),
            ("status_left".to_string(), Json::Str(self.status_left.clone())),
            ("status_right".to_string(), Json::Str(self.status_right.clone())),
            ("tz_offset".to_string(), Json::Num(self.tz_offset as f64)),
            ("splash".to_string(), Json::Bool(self.splash)),
            ("theme".to_string(), theme),
        ])
    }

    fn from_json(j: &Json) -> UiConfig {
        let d = UiConfig::default();
        let s = |k: &str, def: &str| j.get(k).and_then(|v| v.as_str()).map(|x| x.to_string()).unwrap_or_else(|| def.to_string());
        // Theme: start from the brand defaults, override any key present in the
        // config's `theme` object.
        let mut theme = d.theme.clone();
        if let Some(obj) = j.get("theme") {
            for (name, hex) in theme.iter_mut() {
                if let Some(v) = obj.get(name).and_then(|v| v.as_str()) {
                    *hex = v.to_string();
                }
            }
        }
        // Migrate stale persisted defaults: configs written before the pane was
        // renamed / the status bar grew system info keep the old literal — map
        // it forward rather than pinning users to it forever.
        let chat_title = match s("chat_title", &d.chat_title) {
            t if t == "chat" => d.chat_title.clone(),
            t => t,
        };
        let status_right = match s("status_right", &d.status_right) {
            // Forward-migrate the earlier built-in defaults to the current one.
            t if t == "${datetime}  ${tz}" => d.status_right.clone(),
            t if t == "${kbd} ${mouse}  ${net}  ${mem}  ${cpu}  ${datetime} ${tz}" => d.status_right.clone(),
            t => t,
        };
        UiConfig {
            chat_pct: j.get("chat_pct").and_then(|v| v.as_i64()).map(|n| n as u64).unwrap_or(d.chat_pct),
            font_scale: j.get("font_scale").and_then(|v| v.as_i64()).map(|n| n as u64).unwrap_or(d.font_scale),
            swap_panes: j.get("swap_panes").and_then(|v| v.as_bool()).unwrap_or(d.swap_panes),
            chat_title,
            logs_title: s("logs_title", &d.logs_title),
            status_left: s("status_left", &d.status_left),
            status_right,
            tz_offset: j.get("tz_offset").and_then(|v| v.as_i64()).map(|n| n as i32).unwrap_or(d.tz_offset),
            splash: j.get("splash").and_then(|v| v.as_bool()).unwrap_or(d.splash),
            theme,
        }
    }

    /// The framebuffer layout knobs derived from this config.
    #[cfg(not(test))]
    fn layout_cfg(&self) -> crate::framebuffer::LayoutCfg {
        crate::framebuffer::LayoutCfg {
            chat_pct: self.chat_pct,
            scale: self.font_scale,
            swap: self.swap_panes,
            chat_title: self.chat_title.clone(),
            logs_title: self.logs_title.clone(),
            theme: crate::framebuffer::theme_from_pairs(&self.theme),
            splash: self.splash,
            fullscreen: 0,
        }
    }
}

static CONFIG: Locked<Option<UiConfig>> = Locked::new(None);

/// The live config (a clone), loading defaults if not yet initialized.
pub fn current() -> UiConfig {
    CONFIG.with(|c| c.clone()).unwrap_or_default()
}

/// The framebuffer layout (theme, splash, panes) for the initial boot screen —
/// the current config if already loaded, else the brand defaults.
#[cfg(not(test))]
pub fn boot_layout() -> crate::framebuffer::LayoutCfg {
    current().layout_cfg()
}

fn store(cfg: UiConfig) {
    CONFIG.with(|c| *c = Some(cfg));
}

/// Read `ui.json` from the store into the live config, writing defaults first if
/// it does not exist. Also syncs the clock's timezone from the config.
pub fn load() {
    let cfg = match crate::synapse::fs::read(UI_PATH) {
        Some(bytes) => core::str::from_utf8(&bytes)
            .ok()
            .and_then(Json::parse)
            .map(|j| UiConfig::from_json(&j))
            .unwrap_or_default(),
        None => {
            let d = UiConfig::default();
            write_ui(&d);
            d
        }
    };
    crate::clock::set_tz(cfg.tz_offset);
    store(cfg);
}

fn write_ui(cfg: &UiConfig) {
    let text = cfg.to_json().to_pretty();
    crate::synapse::fs::write(UI_PATH, text.as_bytes());
}

/// Load `ui.json`, apply it to the framebuffer layout, and ensure the shortcuts
/// file exists. Called once at shell start.
pub fn load_and_apply() {
    load();
    ensure_shortcuts();
    #[cfg(not(test))]
    crate::framebuffer::relayout(&current().layout_cfg());
}

/// Re-read the config from disk and re-apply the layout (`/ui reload`).
pub fn reload_and_apply() {
    load();
    #[cfg(not(test))]
    crate::framebuffer::relayout(&current().layout_cfg());
}

/// Reset the config to defaults, persist, and re-apply (`/ui reset`).
pub fn reset() {
    let d = UiConfig::default();
    write_ui(&d);
    crate::clock::set_tz(d.tz_offset);
    store(d);
    #[cfg(not(test))]
    crate::framebuffer::relayout(&current().layout_cfg());
}

/// The current `ui.json` text (pretty-printed) for `/ui`.
pub fn ui_json_text() -> String {
    current().to_json().to_pretty()
}

/// The on-disk config file path (for `/open` / `/ui`).
pub fn ui_path() -> &'static str {
    UI_PATH
}
pub fn shortcuts_path() -> &'static str {
    SHORTCUTS_PATH
}

/// Persist a new timezone offset into the config (called by `/datetime tz`).
pub fn persist_tz(offset_secs: i32) {
    let mut cfg = current();
    cfg.tz_offset = offset_secs;
    write_ui(&cfg);
    store(cfg);
}

/// Resolve the status-bar `(left, right)` strings from the config templates.
pub fn status_strings() -> (String, String) {
    let cfg = current();
    (resolve_template(&cfg.status_left), resolve_template(&cfg.status_right))
}

/// Substitute `${var}` tokens: brand, version, date, time, datetime, tz, model,
/// arch, uptime. Unknown vars are left as-is.
fn resolve_template(t: &str) -> String {
    let mut out = String::new();
    let bytes = t.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            if let Some(end) = t[i + 2..].find('}') {
                let var = &t[i + 2..i + 2 + end];
                out.push_str(&resolve_var(var));
                i += 2 + end + 1;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn resolve_var(var: &str) -> String {
    match var {
        "brand" => "ChittiOS".to_string(),
        "version" => crate::VERSION.to_string(),
        "build" => crate::BUILD_TIME.to_string(),
        "date" => crate::clock::format_date(),
        "time" => crate::clock::format_time(),
        "datetime" => crate::clock::format_datetime(),
        "tz" => crate::clock::format_tz(),
        // The booted GGUF's own `general.name` (runtime, not compiled in).
        "model" => crate::cortex::model_name().unwrap_or_else(|| "no model".to_string()),
        "arch" => ARCH.to_string(),
        "uptime" => {
            let s = crate::arch::now_ms() / 1000;
            alloc::format!("up {}:{:02}:{:02}", s / 3600, s % 3600 / 60, s % 60)
        }
        // System info: memory (kernel heap used out of total physical RAM),
        // CPU busy %, net link, input-device activity (`*` = active <1.5 s).
        "mem" => {
            let m = crate::mm::mem_stats();
            let mib = 1024 * 1024;
            let gib = 1024 * mib;
            // Denominator is the machine's RAM (e.g. 6.0G); numerator is what
            // the kernel actually uses (heap allocated + loaded model).
            if m.ram_total >= gib {
                alloc::format!("mem {}M/{}.{}G", (m.heap_used + (m.ram_reserved - m.heap_total)) / mib, m.ram_total / gib, (m.ram_total % gib) * 10 / gib)
            } else {
                alloc::format!("mem {}/{}M", m.heap_used / mib, m.ram_total / mib)
            }
        }
        "cpu" => alloc::format!("cpu {:>3}%", crate::shell::cpu_percent()),
        "cores" => alloc::format!("{}c", crate::arch::cpu_count()),
        "net" => (if crate::net::is_up() { "net up" } else { "net --" }).to_string(),
        "kbd" => {
            let last = crate::console::input_activity_ms();
            let active = last != 0 && crate::arch::now_ms().saturating_sub(last) < 1500;
            (if active { "kbd*" } else { "kbd " }).to_string()
        }
        "mouse" => {
            let last = crate::mouse::activity_ms();
            let active = last != 0 && crate::arch::now_ms().saturating_sub(last) < 1500;
            (if active { "mse*" } else { "mse " }).to_string()
        }
        other => alloc::format!("${{{}}}", other),
    }
}

// --- Shortcuts -----------------------------------------------------------

const DEFAULT_SHORTCUTS: &[(&str, &str, &str)] = &[
    ("Enter", "submit", "send the current line (chat message or /command)"),
    ("Ctrl+C", "stop", "stop the model, or interrupt a running command (/http, /ping, /ws…)"),
    ("Ctrl+D", "exit", "power off on an empty line (EOF)"),
    ("Ctrl+W", "close-pane", "close the action (right) pane; chat full-width"),
    ("Ctrl+V", "paste", "paste the clipboard into the shell line"),
    ("Backspace", "erase", "delete the character before the cursor"),
    ("/ktrace", "toggle-ktrace", "show/hide the ktrace stream in the action pane"),
    ("/open <file>", "open-editor", "open a file in the vim-like editor (right pane)"),
    ("Esc", "editor:normal", "editor: leave insert / visual mode"),
    ("i / a / o", "editor:insert", "editor: insert before / after / open line below"),
    ("h j k l", "editor:move", "editor: move left / down / up / right"),
    ("v / y / p", "editor:yank", "editor: visual select / yank (copy) / paste"),
    ("yy / dd / x", "editor:linedit", "editor: yank line / delete line / delete char"),
    (":w / :q / :wq", "editor:exwrite", "editor: write / quit / write-quit"),
];

fn ensure_shortcuts() {
    if !crate::synapse::fs::exists(SHORTCUTS_PATH) {
        crate::synapse::fs::write(SHORTCUTS_PATH, default_shortcuts_json().as_bytes());
    }
}

fn default_shortcuts_json() -> String {
    let arr: Vec<Json> = DEFAULT_SHORTCUTS
        .iter()
        .map(|(keys, action, desc)| {
            Json::Obj(alloc::vec![
                ("keys".to_string(), Json::Str(keys.to_string())),
                ("action".to_string(), Json::Str(action.to_string())),
                ("desc".to_string(), Json::Str(desc.to_string())),
            ])
        })
        .collect();
    Json::Arr(arr).to_pretty()
}

/// The shortcuts list as `(keys, desc)` pairs, read from `shortcuts.json` (or the
/// built-in defaults).
pub fn shortcuts() -> Vec<(String, String)> {
    let text = crate::synapse::fs::read(SHORTCUTS_PATH)
        .and_then(|b| String::from_utf8(b).ok())
        .unwrap_or_else(default_shortcuts_json);
    match Json::parse(&text).and_then(|j| j.as_array().map(|a| a.to_vec())) {
        Some(arr) => arr
            .iter()
            .filter_map(|item| {
                let keys = item.get("keys")?.as_str()?.to_string();
                let desc = item.get("desc").and_then(|v| v.as_str()).unwrap_or("").to_string();
                Some((keys, desc))
            })
            .collect(),
        None => DEFAULT_SHORTCUTS.iter().map(|(k, _, d)| (k.to_string(), d.to_string())).collect(),
    }
}
