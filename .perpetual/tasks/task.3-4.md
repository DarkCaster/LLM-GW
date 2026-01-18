# Implementing Phase 3: HTTP Server Layer (Top Layer)

You need to implement `server/` sub-package classes accorting to the plan below.

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
