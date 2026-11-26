#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;

struct Worker {
    child: Child,
    _input: ChildStdin,
    descriptor: Value,
}

impl Worker {
    fn start() -> Result<Self, String> {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let script = manifest.join("../../../python/worker.py");
        let python = std::env::var("CARTOON_PYTHON").unwrap_or_else(|_| "python".into());
        let mut command = Command::new(python);
        command.arg("-X").arg("utf8").arg(script)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        #[cfg(windows)] {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command
            .spawn().map_err(|e| format!("启动 Python worker 失败：{e}"))?;
        let input = child.stdin.take().ok_or("worker stdin 不可用")?;
        let stdout = child.stdout.take().ok_or("worker stdout 不可用")?;
        let mut line = String::new();
        BufReader::new(stdout).read_line(&mut line).map_err(|e| format!("读取 worker 地址失败：{e}"))?;
        if line.trim().is_empty() { return Err("worker 没有返回 WebSocket 地址".into()); }
        let descriptor = serde_json::from_str(&line).map_err(|e| format!("worker 地址不是 JSON：{e}"))?;
        Ok(Self { child, _input: input, descriptor })
    }
}

impl Drop for Worker {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); }
}

struct AppState { worker: Mutex<Option<Worker>> }

#[tauri::command(async)]
fn start_worker(state: tauri::State<'_, AppState>) -> Result<Value, String> {
    let mut guard = state.worker.lock().map_err(|_| "worker 锁已损坏")?;
    if let Some(worker) = guard.as_mut() {
        if worker.child.try_wait().map_err(|e| e.to_string())?.is_some() { guard.take(); }
    }
    if guard.is_none() { *guard = Some(Worker::start()?); }
    Ok(guard.as_ref().unwrap().descriptor.clone())
}

#[tauri::command]
fn stop_worker(token: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut guard = state.worker.lock().map_err(|_| "worker 锁已损坏")?;
    if guard.as_ref().and_then(|w| w.descriptor["token"].as_str()) == Some(token.as_str()) {
        guard.take();
    }
    Ok(())
}

#[tauri::command(async)]
fn save_png(bytes: Vec<u8>, filename: String) -> Result<String, String> {
    if bytes.len() > 24 * 1024 * 1024 || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err("只允许保存不超过 24 MB 的 PNG".into());
    }
    let safe_name: String = filename.chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')).collect();
    if !safe_name.starts_with("cartoon-") || !safe_name.ends_with(".png") { return Err("无效的图片文件名".into()); }
    let name = safe_name;
    let directory = dirs_next::download_dir().or_else(|| std::env::current_dir().ok()).ok_or("找不到保存目录")?;
    let path = directory.join(name);
    std::fs::write(&path, bytes).map_err(|e| format!("保存失败：{e}"))?;
    Ok(path.to_string_lossy().into_owned())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState { worker: Mutex::new(None) })
        .invoke_handler(tauri::generate_handler![start_worker, stop_worker, save_png])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
