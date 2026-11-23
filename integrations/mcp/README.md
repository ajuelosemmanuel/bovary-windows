# Ask Bovary from Codex

Use `ask_bovary` to send a short question, note, or code snippet to either local model. Choose `model="gemma"` for Gemma 26B on CPU (the default), or `model="inkling"` for Inkling 2 Mini on NVIDIA CUDA. For example, ask Codex, “Use Bovary with Inkling to review this function.”

The adapter starts over stdio. It loads the worker on demand and releases it after a request. Mini accepts up to 8192 UTF-8 input bytes and 1–256 output tokens. The calling MCP client receives the generated text.

## Connect another installation

Install your preferred model with the Windows launcher and ask it a question to verify setup. Keep a copy of the Bovary source checkout. Install the verified Bovary wheel and `mcp==1.30.0` in a separate Python environment. In the trusted project's `.codex/config.toml`, use absolute paths for that machine.

```toml
[mcp_servers.bovary]
command = 'C:\path\to\mcp-env\Scripts\python.exe'
args = ['-I', 'C:\path\to\integrations\mcp\server.py']
startup_timeout_sec = 45
tool_timeout_sec = 210

[mcp_servers.bovary.env]
BOVARY_HOME = 'C:\path\to\Bovary Data\runtime'
BOVARY_MODEL = 'C:\path\to\Bovary Data\models\gemma-4-26B-A4B-it-Q4_0.gguf'
BOVARY_MODEL_MANIFEST = 'C:\path\to\Bovary Data\models\model-provenance.json'
BOVARY_INKLING_BIN = 'C:\path\to\Bovary Data\inkling\bin'
BOVARY_INKLING_RECEIPT = 'C:\path\to\Bovary Data\inkling\inputs.json'
```

For a friend-kit installation, `Bovary Data/model-path.json` contains the actual model and manifest paths. Run `codex mcp get bovary` in the project to check registration. Open a fresh project session and ask Codex to use Bovary.

The model-choice kit downloads Gemma, Inkling, or both. Omit the settings for a model you haven't installed. The adapter uses the checkout's `previews/inkling` helpers. Each request selects exactly one model.

To test the connection end to end, use the MCP environment's Python to run `integrations/mcp/smoke.py --config <config.toml>`. The check sends a short arithmetic question to each configured model and verifies the reply.

If Gemma reports that its checkpoint no longer matches its verification proof, run `python tools/verify_model.py --model "<checkpoint>" --manifest "<new-manifest-path>"` from the checkout. Once verification succeeds, set `BOVARY_MODEL_MANIFEST` to the new file and restart the MCP connection.
