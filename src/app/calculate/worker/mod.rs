use eframe::wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use web_sys::DedicatedWorkerGlobalScope;
use web_sys::js_sys;

#[derive(Serialize, Deserialize)]
pub struct WorkerReq {
    pub source: crate::app::preset::UnprocessedPreset,
    pub settings: super::GenerationSettings,
}

use crate::app::calculate::ProgressMsg;
use crate::app::calculate::process;

// thread_local! {
//     static CANCELLED: Rc<Cell<bool>> = Rc::new(Cell::new(false));
// }

#[wasm_bindgen]
/// Installs the dedicated-worker message handler used by the WASM app.
pub fn worker_entry() {
    let global: DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    let global_for_handler = global.clone();

    let handler = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
        let req: WorkerReq = match serde_wasm_bindgen::from_value(e.data()) {
            Ok(v) => v,
            Err(err) => {
                let _ = global_for_handler.post_message(&match serde_wasm_bindgen::to_value(
                    &ProgressMsg::Error(format!("bad req: {err}")),
                ) {
                    Ok(v) => v,
                    Err(e) => serde_wasm_bindgen::to_value(&ProgressMsg::Error(format!(
                        "serialization error: {e}"
                    )))
                    .unwrap_or(JsValue::NULL),
                });
                return;
            }
        };

        let global2 = global_for_handler.clone();
        let mut sink = |msg: ProgressMsg| {
            let val = match serde_wasm_bindgen::to_value(&msg) {
                Ok(v) => v,
                Err(e) => {
                    let _ = global2.post_message(
                        &serde_wasm_bindgen::to_value(&ProgressMsg::Error(format!(
                            "serialization error: {e}"
                        )))
                        .unwrap_or(JsValue::NULL),
                    );
                    return;
                }
            };
            let _ = global2.post_message(&val);
        };

        if let Err(e) = process(req.source, req.settings, &mut sink) {
            sink(ProgressMsg::Error(e.to_string()));
        }
    }) as Box<dyn FnMut(_)>);

    global.set_onmessage(Some(handler.as_ref().unchecked_ref()));
    handler.forget();
}
