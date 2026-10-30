use serde_json::Value;
use tauri::{plugin::{Builder, PluginHandle, TauriPlugin}, AppHandle, Manager, Runtime};

struct Bridge<R: Runtime>(PluginHandle<R>);

macro_rules! forward {
    ($name:ident, $native:literal) => {
        #[tauri::command]
        async fn $name<R: Runtime>(app: AppHandle<R>, payload: Value) -> Result<Value, String> {
            app.state::<Bridge<R>>().0.run_mobile_plugin($native, payload).map_err(|e| e.to_string())
        }
    };
}
forward!(initialize, "initialize");
forward!(process_frame, "processFrame");
forward!(reset, "reset");
forward!(stop, "stop");
forward!(save_png, "savePng");

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("cartoon")
        .invoke_handler(tauri::generate_handler![initialize, process_frame, reset, stop, save_png])
        .setup(|app, api| {
            let handle = api.register_android_plugin("cn.coursenotes.cartoon", "CartoonPlugin")?;
            app.manage(Bridge(handle));
            Ok(())
        })
        .build()
}
