# Implementing Phase 1 of application: Engine Management Foundation (Bottom Layer)

Here you need to implement `engine/` sub-package classes accorting to the plan below.

## Step 1: `engine/engine_client.py` - Base EngineClient Class

**Purpose:** Abstract base class for HTTP communication with LLM engines.

**Implementation Details:**

- Create abstract base class `EngineClient` with the following responsibilities:
  - Define abstract method `estimate_tokens(request_data: dict) -> int` that subclasses must implement to calculate token requirements from incoming requests
  - Define abstract method `transform_request(request_data: dict) -> dict` for engine-specific request transformations (e.g., removing unsupported fields, renaming fields)
  - Define abstract method `transform_response(response_data: dict) -> dict` for engine-specific response transformations
  - Define abstract method `get_supported_endpoints() -> List[str]` to return list of supported API endpoints (e.g., `/v1/chat/completions`)
  - Implement concrete method `forward_request(session: aiohttp.ClientSession, url: str, request_data: dict) -> aiohttp.ClientResponse` that:
    - Calls `transform_request()` to modify the request
    - Forwards the transformed request to the engine's HTTP endpoint
    - Returns the response for streaming
  - Implement concrete method `check_health(base_url: str, timeout: float = 5.0) -> bool` that:
    - Uses aiohttp.ClientSession to perform a health check request (e.g., GET to `/health` or `/v1/models`)
    - Returns True if engine endpoint is available and responding
    - Handles connection errors, timeouts gracefully
    - Uses asyncio for async operations
  - Store base_url for the engine connection
  - Store logger instance for this class

**Dependencies:** `aiohttp`, `asyncio`, `logging`, `abc` (for abstract base class)

**Why:** This is the foundation for all engine communication. Having the abstract interface defined first allows us to implement concrete engines and test them independently.

## Step 2: `engine/llamacpp_engine.py` - LlamaCppEngine Class

**Purpose:** Concrete implementation of EngineClient for llama.cpp engines.

**Implementation Details:**

- Create `LlamaCppEngine` class that inherits from `EngineClient`
- Implement `estimate_tokens(request_data: dict) -> int`:
  - Extract the chat history or prompt from request_data
  - Determine request type (chat completions vs text completions)
  - For chat completions: build the messages array
  - For text completions: extract the prompt
  - Make async HTTP POST request to llama.cpp tokenization endpoint (typically `/tokenize`)
  - Send the appropriate payload based on request type
  - Parse response to get token count from the tokenization result
  - If `max_tokens` is present in request, add it to the estimated tokens
  - Return total estimated tokens (prompt tokens + max_tokens)
  - Handle errors (engine not available, invalid response) and raise appropriate exceptions
- Implement `transform_request(request_data: dict) -> dict`:
  - Check if request contains unsupported fields for llama.cpp
  - Remove or transform unsupported fields (e.g., certain OpenAI-specific parameters)
  - Return transformed request dictionary
  - Log warnings for removed fields
- Implement `transform_response(response_data: dict) -> dict`:
  - Transform llama.cpp response format to OpenAI-compatible format if needed
  - Handle streaming vs non-streaming responses differently
  - Return transformed response
- Implement `get_supported_endpoints() -> List[str]`:
  - Return list: `["/v1/chat/completions", "/v1/completions"]`
  - These are the endpoints that llama.cpp supports
- Override `check_health()` if llama.cpp has specific health check requirements:
  - Use `/health` endpoint if available
  - Otherwise use `/v1/models` endpoint to verify engine is responsive
- Store any llama.cpp-specific configuration

**Dependencies:** `engine_client.py`, `aiohttp`, `asyncio`, `json`, `logging`

**Why:** This provides a concrete implementation we can test. We can verify the engine client pattern works before building the process management layer.

## Step 3: `engine/engine_process.py` - EngineProcess Class

**Purpose:** Wrapper for managing a single LLM engine subprocess.

**Implementation Details:**

- Create `EngineProcess` class with the following responsibilities:
  - Constructor accepts: `binary_path: str`, `args: List[str]`, `work_dir: str = None`
  - Implement `async start()` method:
    - Use `asyncio.subprocess.create_subprocess_exec()` to spawn the engine binary
    - Pass binary_path and args to the subprocess
    - Set `stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE` for log capture
    - Set working directory if provided
    - Store the subprocess handle
    - Create asyncio tasks to continuously read and log stdout/stderr
    - Store process start time
    - Set status to "running"
  - Implement `async stop(timeout: float = 10.0)` method:
    - Send SIGTERM to the process (graceful shutdown)
    - Wait for process to exit with timeout
    - If timeout expires, send SIGKILL (forceful shutdown)
    - Clean up stdout/stderr reader tasks
    - Set status to "stopped"
  - Implement `is_running() -> bool` property:
    - Check if subprocess is still alive
    - Return True if running, False otherwise
  - Implement `get_pid() -> Optional[int]` property:
    - Return process PID if running
    - Return None if not running
  - Implement `get_status() -> str` property:
    - Return status string: "running", "stopped", "crashed"
    - Detect crashed state if process exited unexpectedly
  - Implement async log reader methods:
    - `_read_stdout()` - continuously read stdout and log with INFO level
    - `_read_stderr()` - continuously read stderr and log with WARNING level
    - Both should run as background tasks while process is alive
  - Store logger instance for output logging

**Dependencies:** `asyncio`, `asyncio.subprocess`, `logging`, `typing`, `signal`

**Why:** Process management is the next layer. We can test spawning and controlling engine processes independently before integrating with the client layer.

## Step 4: `engine/engine_manager.py` - EngineManager Class

**Purpose:** Coordinate engine lifecycle - stop old engines, start new ones, track state.

**Implementation Details:**

- Create `EngineManager` class with singleton pattern (one manager for the entire application)
- Maintain state:
  - `current_engine_process: Optional[EngineProcess]` - currently running engine process
  - `current_engine_client: Optional[EngineClient]` - client for current engine
  - `current_variant_config: Optional[dict]` - configuration of currently running variant
  - `current_model_name: Optional[str]` - name of currently loaded model
  - `_lock: asyncio.Lock` - to prevent concurrent engine switches
- Implement `async ensure_engine(model_name: str, variant_config: dict, engine_type: str) -> EngineClient`:
  - Acquire lock to prevent race conditions
  - Compare requested variant with current variant:
    - If same variant is already running, verify health and return current client
    - If different variant needed, proceed with engine switch
  - If engine switch needed:
    - Call `await self._stop_current_engine()`
    - Call `await self._start_new_engine(model_name, variant_config, engine_type)`
  - Return the engine client
- Implement `async _stop_current_engine()`:
  - If no engine running, return immediately
  - Log the shutdown
  - Call `await current_engine_process.stop(timeout=15.0)`
  - Set current_engine_process and current_engine_client to None
  - Clear current variant config
- Implement `async _start_new_engine(model_name: str, variant_config: dict, engine_type: str)`:
  - Extract binary, args, connect URL from variant_config
  - Create appropriate EngineClient based on engine_type:
    - If `engine_type == "llama.cpp"`, create `LlamaCppEngine(connect_url)`
    - Future: add other engine types
  - Create `EngineProcess(binary, args)`
  - Call `await engine_process.start()`
  - Call `await self._wait_for_engine_ready(engine_client, timeout=60.0)`
  - Store engine_process, engine_client, variant_config, model_name in instance variables
- Implement `async _wait_for_engine_ready(engine_client: EngineClient, timeout: float) -> bool`:
  - Loop with sleep intervals (e.g., every 1 second)
  - Call `await engine_client.check_health()`
  - If health check passes, return True
  - If timeout expires, raise TimeoutError
  - Log progress (e.g., "Waiting for engine to be ready... Xsec")
- Implement `get_current_client() -> Optional[EngineClient]`:
  - Return current_engine_client if available
  - Return None if no engine running
- Implement `async shutdown()`:
  - Stop current engine if running
  - Clean up resources

**Dependencies:** `engine_process.py`, `engine_client.py`, `llamacpp_engine.py`, `asyncio`, `logging`, `typing`

**Why:** This ties together process management and client communication. We can now test the full engine lifecycle: start, health check, stop, restart with different config.

## Step 5: `engine/__init__.py` - Engine Package Interface

**Purpose:** Expose public API of the engine package.

**Implementation Details:**

- Import and expose:
  - `EngineManager` (main interface)
  - `EngineClient` (for type hints)
  - `LlamaCppEngine` (for instantiation)
- Define `__all__` list with exported classes

**Dependencies:** All engine package modules

**Why:** Package initialization to provide clean import interface.

---

# Implementing Phase 2: Model Selection Logic (Middle Layer)

Here you need to implement `models/` sub-package classes accorting to the plan below.

## Step 1: `models/model_selector.py` - ModelSelector Class

**Purpose:** Analyze requests and select appropriate model variant based on context requirements.

**Implementation Details:**

- Create `ModelSelector` class with the following responsibilities:
  - Constructor accepts `cfg: PyLuaHelper` (configuration object)
  - Parse and index all models from configuration:
    - Iterate through `cfg.get_list("models")` to get model indices
    - For each model, extract name, engine type, and variants
    - Build internal data structure: `{model_name: {engine: str, variants: [variant_configs]}}`
    - Sort variants by context size (smallest to largest) for efficient selection
- Implement `async select_variant(model_name: str, request_data: dict, engine_manager: EngineManager) -> dict`:
  - Validate that model_name exists in configuration
  - If not found, raise `ValueError(f"Model '{model_name}' not found in configuration")`
  - Get engine type for this model
  - Retrieve model's variant list
  - Estimate required context size:
    - Check if there's a currently running engine for this model that supports tokenization
    - Get current engine client from engine_manager
    - If available and variant supports tokenization:
      - Call `await engine_client.estimate_tokens(request_data)`
      - Get token count
    - If tokenization not available:
      - Use fallback estimation (e.g., character count / 4 as rough token estimate)
      - Log warning about imprecise estimation
  - Add safety margin to estimated tokens (e.g., 10% or fixed amount like 512 tokens)
  - Select smallest variant where `variant.context >= estimated_tokens`:
    - Iterate through sorted variants
    - Find first variant with sufficient context
    - If no variant has enough context, raise `ValueError(f"Request requires {estimated_tokens} tokens, but largest variant only supports {max_context}")`
  - Return selected variant configuration dictionary
- Implement `get_model_info(model_name: str) -> dict`:
  - Return model information: name, engine type, available context sizes
  - Used for /v1/models endpoint
- Implement `list_models() -> List[str]`:
  - Return list of all configured model names
  - Used for /v1/models endpoint

**Dependencies:** `config.ConfigLoader`, `engine.EngineManager`, `typing`, `logging`

**Why:** This implements the intelligence of variant selection. It requires the engine layer to be complete (for tokenization), but is independent of the HTTP server layer.

## Step 2: `models/__init__.py` - Models Package Interface

**Purpose:** Expose public API of the models package.

**Implementation Details:**

- Import and expose `ModelSelector`
- Define `__all__` list

**Dependencies:** `model_selector.py`

---

# Implementing Phase 3: HTTP Server Layer (Top Layer)

Here you need to implement `server/` sub-package classes accorting to the plan below.

## Step 1: `server/request_handler.py` - RequestHandler Class

**Purpose:** Process individual HTTP requests, coordinate model selection and engine management.

**Implementation Details:**

- Create `RequestHandler` class with the following responsibilities:
  - Constructor accepts:
    - `model_selector: ModelSelector`
    - `engine_manager: EngineManager`
    - `cfg: PyLuaHelper` (for any server-level config)
  - Implement `async handle_chat_completion(request: aiohttp.web.Request) -> aiohttp.web.Response`:
    - Parse JSON body from request
    - Validate required fields (model, messages)
    - Extract model name from request
    - Call `await model_selector.select_variant(model_name, request_data, engine_manager)`
    - Get selected variant config
    - Call `await engine_manager.ensure_engine(model_name, variant_config, engine_type)`
    - Get engine client from engine_manager
    - Check if endpoint is supported by this engine (call `client.get_supported_endpoints()`)
    - If not supported, return HTTP 400 with error message
    - Call `await client.forward_request(url, request_data)` to send request to engine
    - If request has `stream=True`:
      - Create StreamResponse
      - Read response chunks from engine
      - Write chunks to client
      - Handle Server-Sent Events (SSE) format for streaming
    - If request has `stream=False`:
      - Read complete response
      - Transform if needed
      - Return JSON response
    - Handle errors and return appropriate HTTP status codes
  - Implement `async handle_completion(request: aiohttp.web.Request) -> aiohttp.web.Response`:
    - Similar to chat completion but for `/v1/completions` endpoint
    - Parse request, select variant, ensure engine, forward request
  - Implement `async handle_models_list(request: aiohttp.web.Request) -> aiohttp.web.Response`:
    - Call `model_selector.list_models()`
    - Build OpenAI-compatible response format:
      - `{"object": "list", "data": [{"id": model_name, "object": "model", ...}, ...]}`
    - Return JSON response
  - Implement `async handle_model_info(request: aiohttp.web.Request) -> aiohttp.web.Response`:
    - Extract model name from URL path
    - Call `model_selector.get_model_info(model_name)`
    - Build OpenAI-compatible response
    - Return JSON response
  - Implement error handling:
    - Create method `_error_response(status: int, message: str, error_type: str = "invalid_request_error") -> aiohttp.web.Response`
    - Return OpenAI-compatible error format
  - Store logger for request/response logging

**Dependencies:** `aiohttp`, `json`, `logging`, `models.ModelSelector`, `engine.EngineManager`

**Why:** This is the business logic layer that connects all components. It requires both model selection and engine management to be complete.

## Step 2: `server/gateway_server.py` - GatewayServer Class

**Purpose:** Main HTTP server that listens and routes requests.

**Implementation Details:**

- Create `GatewayServer` class with the following responsibilities:
  - Constructor accepts `cfg: PyLuaHelper`
  - Parse server configuration:
    - Extract `listen_v4` address and port
    - Extract `listen_v6` address and port
  - Initialize components:
    - Create `ModelSelector(cfg)`
    - Create `EngineManager()`
    - Create `RequestHandler(model_selector, engine_manager, cfg)`
  - Implement `async start()`:
    - Create `aiohttp.web.Application()`
    - Register routes:
      - `POST /v1/chat/completions` → `request_handler.handle_chat_completion`
      - `POST /v1/completions` → `request_handler.handle_completion`
      - `GET /v1/models` → `request_handler.handle_models_list`
      - `GET /v1/models/{model_id}` → `request_handler.handle_model_info`
    - Create list of runners for IPv4 and IPv6:
      - If `listen_v4 != "none"`, parse address/port and create IPv4 runner
      - If `listen_v6 != "none"`, parse address/port and create IPv6 runner
    - Setup and start all runners
    - Log listening addresses
    - Store runners and sites for cleanup
  - Implement `async stop()`:
    - Stop all site runners
    - Call `await engine_manager.shutdown()`
    - Clean up application
  - Implement `async run()`:
    - Call `await start()`
    - Wait forever (until interrupted)
    - Handle graceful shutdown on interrupt
  - Implement helper method `_parse_listen_address(address: str) -> tuple[str, int]`:
    - Parse "host:port" string
    - Return (host, port) tuple
    - Validate format

**Dependencies:** `aiohttp`, `asyncio`, `logging`, `server.RequestHandler`, `models.ModelSelector`, `engine.EngineManager`, `config.ConfigLoader`

**Why:** This is the entry point that assembles all components and starts the HTTP server.

## Step 3: `server/__init__.py` - Server Package Interface

**Purpose:** Expose public API of the server package.

**Implementation Details:**

- Import and expose:
  - `GatewayServer` (main server class)
  - `RequestHandler` (for testing)
- Define `__all__` list

**Dependencies:** All server package modules

---

# Implementing Phase 4: Integration and Main Entry Point

## Step 1: Update `main.py` - Application Entry Point

**Purpose:** Wire everything together and provide command-line interface.

**Implementation Details:**

- Modify existing `main()` function:
  - After loading configuration, create `GatewayServer(cfg)`
  - Replace placeholder comment `# start server here and wait for interrupt` with:
    - `await server.run()`
  - Make `main()` async: `async def main():`
  - Add proper signal handling for graceful shutdown (SIGINT, SIGTERM)
  - In finally block:
    - Call `await server.stop()` before cleanup
  - Handle asyncio properly:
    - If script run directly, use `asyncio.run(main())`
- Add signal handlers:
  - Register handlers for SIGINT and SIGTERM
  - Set flag to trigger graceful shutdown
  - Server should detect flag and stop gracefully

**Dependencies:** `asyncio`, `signal`, `server.GatewayServer`

**Why:** Final integration step that creates a runnable application.
