use std::env;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Fix WebKit crashes on Wayland
    env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
