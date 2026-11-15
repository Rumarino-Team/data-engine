use std::env;

/// WebKitGTK workarounds for Linux, selected with `DATA_ENGINE_WEBKIT_MODE`:
///   dmabuf         (default) disable WebKit's DMA-BUF renderer; avoids Wayland/NVIDIA crashes, slower path
///   none           no workaround; fastest path, may crash or show a blank window on NVIDIA + Wayland
///   explicit-sync  only `__NV_DISABLE_EXPLICIT_SYNC=1`; reported to avoid Error 71 without the speed cost
///   both           both workarounds
/// Example: `DATA_ENGINE_WEBKIT_MODE=explicit-sync bun run tauri:dev`
#[cfg(target_os = "linux")]
fn apply_webkit_workarounds() {
    let mode = env::var("DATA_ENGINE_WEBKIT_MODE").unwrap_or_else(|_| "dmabuf".into());
    let (dmabuf, explicit_sync) = match mode.as_str() {
        "none" => (false, false),
        "explicit-sync" => (false, true),
        "both" => (true, true),
        "dmabuf" => (true, false),
        other => {
            eprintln!("unknown DATA_ENGINE_WEBKIT_MODE '{other}', using 'dmabuf'");
            (true, false)
        }
    };
    if dmabuf {
        env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
    if explicit_sync {
        env::set_var("__NV_DISABLE_EXPLICIT_SYNC", "1");
    }
    eprintln!("webkit mode: {mode} (dmabuf disabled: {dmabuf}, nv explicit sync disabled: {explicit_sync})");
}

#[cfg(not(target_os = "linux"))]
fn apply_webkit_workarounds() {}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    apply_webkit_workarounds();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
