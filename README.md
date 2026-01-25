# LLM gateway - personal runner for LLAMA.CPP compatible LLM engines

## Overview

The LLM gateway is a lightweight, self-contained solution that dynamically manages multiple Llama.cpp-compatible engines on demand. It acts as a proxy between incoming OpenAI API-compliant requests and the underlying LLM engine processes, automatically selecting and launching appropriate model variants based on context size requirements and other parameters.

Designed for personal use with local hardware, this gateway enables efficient switching between different models—such as Qwen3-MoE or smaller instruction-tuned versions. It intelligently estimates token requirements, restarts engines with optimal configurations when needed, and gracefully handles idle timeouts to conserve system resources.

This document provides comprehensive usage guidance for users who wish to set up, configure, and operate the LLM gateway.

NOTE: This is highly experimental software written for my experiments and testing; it may or may not work for you (most probably it will not).

## Getting Started

To begin using the LLM gateway, follow these steps:

1. **Install Python dependencies**: Run `init.bat` (on Windows) or `init.sh` (on Linux/macOS) from within your project directory. This will install a dedicated Python environment using `uv`, create a virtual environment (`venv/`), and set up all required packages (including `aiohttp` for HTTP server/client functionality and `python-lua-helper` for configuration parsing).

2. **Prepare configuration file**: Edit `example.cfg.lua` to define your models, engine binaries, paths, and desired settings (e.g., context size, memory quantization). You can use the included Qwen3-MoE model examples as templates for real-world setups. The configuration is validated automatically on load via embedded Lua validation scripts.

3. **Start the gateway**:
   - On Windows: Run `run.bat -c example.cfg.lua`
   - On Linux/macOS: Run `run.sh -c example.cfg.lua`

The server will start listening on the specified IPv4/IPv6 address (e.g., `127.0.0.1:7777`) and begin accepting OpenAI API-compatible requests at endpoints like `/v1/chat/completions` and `/v1/models`.

## Command Line Flags

The LLM gateway supports one primary command-line argument:

| Flag | Description |
|------|-------------|
| `-c`, `--config` | Specifies the path to the Lua configuration file (e.g., `example.cfg.lua`). This is required and must point to a valid config file. |

Example:

```bash
run.sh -c example.cfg.lua
```

or

```bash
run.bat -c example.cfg.lua
```

or

```bash
uv run main.py -c example.cfg.lua
```

## Configuration

The LLM gateway uses Lua scripts for advanced configuration capabilities. Configuration files are processed by use of standard Lua interpreter and exports the configuration to Python. All settings are defined in a single `.cfg.lua` file using structured tables.

### Basic Structure

Your config must define two top-level tables:

- `server`: Contains general server parameters like listening addresses and timeouts.
- `models`: An array of model definitions, each representing a distinct LLM variant with multiple engine configurations (variants).

Each model entry includes:

- `name` – unique identifier for the model (used in API requests)
- `engine` – specifies the engine type (`presets.engines.llamacpp`)
- `connect` – base URL where the Llama.cpp server listens
- `tokenization` – settings to estimate context size before launching a model
- `variants` – multiple configurations with varying `context`, `args`, and memory settings

### Example Configuration Snippet

```lua
qwen3_30b_instruct_model = {
    engine = presets.engines.llamacpp,
    name = "qwen3-30b-instruct",
    connect = "http://127.0.0.1:8080",
    tokenization = {
        binary = "/path/to/llama-tokenize",
        extra_args = { "-m", "/path/to/model.gguf" },
        extra_tokens_per_message = 8
    },
    variants = {
        {
            binary = "/path/to/llama-server",
            args = {"-c", "40960", "-ngl", "999", "--fit-ctx", "40960"},
            context = 40960
        }
    }
}

models = { qwen3_30b_instruct_model }
```

### Key Configuration Parameters

| Parameter | Description |
|---------|-------------|
| `server.listen_v4` / `listen_v6` | IPv4 and IPv6 addresses to bind the server (use `"none"` to disable) |
| `server.health_check_timeout` | Timeout in seconds for checking engine health after startup (must be > 0) |
| `server.engine_startup_timeout` | Maximum time allowed for a new engine to start successfully (must be > 0) |
| `server.engine_idle_timeout` | Time in seconds before idle engines are automatically shut down (must be > 0) |
| `model.context` | The maximum context size (in tokens) supported by this variant |
| `model.variants[i].binary` | Path to the `llama-server` executable for this variant |
| `model.variants[i].args` | Arguments passed directly to the Llama.cpp server binary |
| `model.tokenization.binary` | Path to the `llama-tokenize` executable for estimating token counts |
| `model.tokenization.extra_args` | Required arguments for the tokenizer (typically model path) |
| `model.tokenization.extra_tokens_per_message` | Additional tokens to add per message for chat template overhead |

### Advanced Configuration Tips

- Use helper functions like `get_llama_moe_args()` in your config scripts to generate optimized startup arguments for MoE models.
- Leverage `concat_arrays()` and `merge_tables()` helper functions defined in `pre.lua` for building complex argument lists.
- Set up multiple variants per model with different context sizes and memory settings (e.g., low VRAM vs high performance) for dynamic switching.
- Timeout parameters cascade: variant-level settings override model-level settings, which override server-level defaults.

## Disclaimer & Warnings

### Safety Considerations

- The LLM gateway launches external processes (`llama-server`, `llama-tokenize`) directly from your filesystem.
- Ensure that the paths to `llama-server` and `llama-tokenize` binaries are correct and accessible.
- **Only one request can be processed simultaneously** - the gateway uses request locking to prevent concurrent request handling.
- Configuration is validated on load using embedded Lua validation scripts (`post.lua`). Invalid configurations will prevent the gateway from starting.

### Performance & Resource Usage

**WARNING:** Improper configuration parameters (especially context size, batch size, and cache settings) can lead to severe memory exhaustion and may freeze or halt your system. Start with conservative settings and monitor resource usage carefully. The gateway automatically shuts down idle engines after `server.engine_idle_timeout` to help reduce resource consumption.

## Usage Examples

Once the gateway is running, you can send requests using curl or any OpenAI-compatible client:

```bash
curl -X POST http://127.0.0.1:7777/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b-instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 256,
    "stream": false
  }'
```

For streaming responses, set `"stream": true` in the request body:

```bash
curl -X POST http://127.0.0.1:7777/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b-instruct",
    "messages": [
      {"role": "user", "content": "Tell me a short story."}
    ],
    "max_tokens": 2048,
    "stream": true
  }'
```

The gateway will:

1. Parse the request and identify the model (`qwen3-30b-instruct`)
2. Estimate required context using `llama-tokenize` (standalone tokenizer) if the engine is not already running
3. Re-estimate context size using the running engine's tokenization endpoint if needed
4. Launch or switch to the appropriate engine variant with sufficient `context` size
5. Forward the request and stream responses back in real-time

You can also list available models:

```bash
curl http://127.0.0.1:7777/v1/models
```

This will return an OpenAI-compatible list of all configured model names.

## Support & Contribution

**IMPORTANT:** This project is purely experimental and was developed primarily for personal research using LLM-assisted coding agents. The codebase is in active development and may contain bugs, incomplete features, and unoptimized implementations.

**Known Limitations:**

- Only the `/v1/chat/completions` endpoint is implemented and somewhat tested
- The `/v1/embeddings` endpoint is not yet implemented
- Only `llama.cpp` engine type is supported (other engine types can be added via the extensible architecture)
- Error handling may be incomplete in edge cases
- Documentation may lag behind implementation changes
- Requests are processed sequentially - no concurrent request handling (not planned)

Contributions and bug reports are welcome, but please understand that this is a hobby project with no guaranteed support or maintenance schedule. Users should thoroughly test the gateway in their own environment before relying on it for any important workflows.

## Disclaimer

This project and its associated materials are provided "as is" and without warranty of any kind, either expressed or implied. The author(s) do not accept any liability for any issues, damages, or losses that may arise from the use of this project or its components. Users are responsible for their own use of the project and should exercise caution and due diligence when incorporating any of the provided code or functionality into their own projects.

The project is intended for educational and experimental purposes only. It should not be used in production environments or for any mission-critical applications without thorough testing and validation. The author(s) make no guarantees about the reliability, security, or suitability of this project for any particular use case.

Users are encouraged to review the project's documentation, logs, and source code carefully before relying on it. If you encounter any problems or have suggestions for improvement, please feel free to reach out to the project maintainers.
