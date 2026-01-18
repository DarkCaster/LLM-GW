# Implementing Phase 1 of application: Engine Management Foundation (Bottom Layer)

You need to implement `engine/` sub-package classes accorting to the plan below. Also implement some tests if needed.

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
