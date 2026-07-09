const params = new URLSearchParams(self.location.search)
const scriptName = params.get("script") || "./obamify.js"

const obamifyModule = await import(scriptName)
const wasmName = scriptName.replace(".js", "_bg.wasm")

await obamifyModule.default(wasmName)
