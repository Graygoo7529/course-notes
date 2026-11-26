const COMMANDS: &[&str] = &["initialize", "process_frame", "reset", "stop", "save_png"];

fn main() {
  tauri_plugin::Builder::new(COMMANDS)
    .android_path("android")
    .build();
}
