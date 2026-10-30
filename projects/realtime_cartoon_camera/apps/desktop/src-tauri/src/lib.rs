#[cfg(not(target_os = "android"))]
mod desktop;
#[cfg(not(target_os = "android"))]
pub use desktop::run;

#[cfg(target_os = "android")]
#[tauri::mobile_entry_point]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_cartoon::init())
        .run(tauri::generate_context!())
        .expect("error while running Android cartoon camera");
}
