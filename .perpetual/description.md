# LLM gateway - runner for OpenAI compatible LLM engines

This project is a runner for a OpenAI API compatible LLM engines providing proxifying of LLM-requests (http+json) from entry point into LLM engine and vise versa.
It intercepts incoming inference request (like /v1/chat/completions), tokenize the request (if needed) and select proper engine variant with proper settings for the request depending on model name, context requirements, and other parameters.

## Project Structure and Workflow

### Configuration

Configuration performed via lua config files using `python-lua-helper` package. See example.cfg.lua for current configuration structure. Main config parameters defined at `server` and `models` table.

`ConfigLoader` sub-package will parse and load lua configuration via python-lua-helper, making `server` and `model` tables available in the python application by use of helper `PyLuaHelper` class. All validation performed at lua side on configuration load time, so you not needed to re-validate configuration values at python code.

### Typical Workflow (simplified)

LLM gateway selects particular LLM engine and its configuration variant by incoming-request contents depending on model name and context size requirements. If new engine and configuration differs - it shuting down currently running engine and spawn a new engine with new configuration. Then it waits while engine http-entrypoint become available and then if forward incoming request to engine entrypoint and stream back the response with http error code.

NOTES:

- Main configuration unit is a model with unique name. LLM engine type with corresponding parameters and different confioguration variants tied to the model name.
- When forwarding request to the running engine, some request json fields may be changed on the fly for request and/or response to match particular LLM engine format.
- For the `llama.cpp` engine type, if there is already running engine instance present it can be used for pre-tokenization call to estimate context size requirements, and after that engine may be restarted with a more suitable configuration.

## Main Project Classes and Subpackages

Here are the brief descriptions of project sub-packages and its functions, some packages may be work-in-progress or not implemented at all yet.

### Subpackage `server/`

Main HTTP server implementation that handles incoming requests.

**Classes:**

- `GatewayServer`
  - Purpose: Main HTTP server that listens on configured addresses (IPv4/IPv6)
  - Responsibilities:
    - Initialize aiohttp web application
    - Register routes for OpenAI API endpoints (/v1/chat/completions, /v1/completions, etc.)
    - Handle incoming HTTP requests
    - Coordinate with RequestHandler to process requests
    - Manage server lifecycle (start, stop, graceful shutdown)

- `RequestHandler`
  - Purpose: Process individual incoming requests
  - Responsibilities:
    - Parse and validate incoming JSON requests
    - Extract model name
    - Coordinate with ModelSelector to choose appropriate model variant
    - Coordinate with EngineManager to ensure correct engine is running
    - Forward request to engine via EngineClient
    - Stream response back to client
    - Handle errors and return appropriate HTTP status codes

### Subpackage: `models/`

Model selection and management logic.

**Classes:**

- `ModelSelector`
  - Purpose: Select appropriate model variant based on request requirements
  - Responsibilities:
    - Analyze request to determine context size needed
    - Use `engine/` subpackage to estimate context size requirements in tokens
    - Select variant from model that satisfies context requirements
    - Return selected variant configuration
    - Handle variant selection strategy (for now only smallest sufficient context suitable for request + answer)

### Subpackage: `engine/`

Engine lifecycle management and interaction.

**Classes:**

- `EngineManager`
  - Purpose: Manage the lifecycle of LLM engine processes
  - Responsibilities:
    - Track currently running engine (if any)
    - Stop/kill running engine
    - Start engine with selected variant configuration
    - Monitor engine process health
    - Wait for engine HTTP endpoint to become available after start
    - Coordinate engine state changes

- `EngineProcess`
  - Purpose: Wrapper for a single running engine process
  - Responsibilities:
    - Spawn subprocess with configured binary and arguments
    - Monitor process status (running, stopped, crashed)
    - Terminate process gracefully or forcefully
    - Capture stdout/stderr for logging
    - Provide process information (PID, status, etc.)

- `EngineClient` (base class)
  - Purpose: HTTP client for communicating with running engine
  - Responsibilities:
    - Forward requests to engine HTTP endpoint with transformation (if needed)
    - Handle request transformation if needed (modify JSON fields for specific engines)
    - Handle response transformation if needed
    - Stream responses back
    - Perform health checks (check if endpoint is available)
    - Define method for token estimation

- `LlamaCppEngine` (specific implementation of `EngineClient`)
  - Purpose: Llama.cpp-specific engine handling
  - Responsibilities:
    - Implement llama.cpp-specific request/response transformations, return http error on unsuported request-types
    - Implement token estimation method: call llama.cpp tokenization endpoint, calculate and return total tokens from chat history + max_tokens if present
    - Parse llama.cpp specific responses

### Subpackage: `utils/`

Utility functions and helpers.

**Classes/Modules:**

- `logger.py`
  - Purpose: Logging configuration and utilities
  - Functions:
    - Setup logging with appropriate formats
    - Provide logger instances for different modules

## Bundled Python Packages

Consider using following internal python packages:

- `json` for working with OpenAI API LLM requests and responses.
- `asyncio` and `asyncio.subprocess` for async process spawning and monitoring, for managing llama.cpp server processes (`llama-server`) and possibly others.

## External Python Packages

Project using following external python packages:

- `aiohttp` for implementing http server and client functionality with streaming support. `aiohttp.ClientSession` for checking if engine endpoints are ready.
- `python-lua-helper` for loading configuration from lua files, see below how to use it.

## Accessing lua configuration from python

Here is a trimmed down source code of `PyLuaHelper` class from `python-lua-helper` library for reference how to access configuration via `cfg` property from `ConfigLoader` helper class.

```py3
class PyLuaHelper:
    """
    Python helper for loading Lua configuration files by running them with Lua interpreter and exporting requested tables to Python dictionaries.
    """

    # Initialization code was removed, methods for accessing values from exported lua tables provided below:

    def __getitem__(self, key: str) -> str:
        """Get item from exported variables dictionary."""
        return self._variables.get(key, "")

    def __contains__(self, key: str) -> bool:
        """Check if variable is available."""
        return key in self._metadata and self._metadata[key] != ""

    def __iter__(self):
        """Iterate over exported variable names."""
        return iter(self._variables)

    def __len__(self) -> int:
        """Get number of exported variables."""
        return len(self._variables)

    def keys(self) -> List[str]:
        """Get list of exported variable names."""
        return list(self._variables.keys())

    def values(self) -> List[str]:
        """Get list of exported variable values."""
        return list(self._variables.values())

    def items(self) -> List[tuple]:
        """Get list of (name, value) tuples."""
        return list(self._variables.items())

    def is_table(self, key: str) -> bool:
        """Check variable is a table, return true or false"""
        # code trimmed down

    def get_type(self, key: str) -> str:
        """Get variable type"""
        if self.is_table(key):
            return "table"
        if key in self._metadata and re.match(r"^string.*", self._metadata[key]):
            return "string"
        return self._metadata.get(key, "none")

    def get(self, key: str, default: str = None) -> str:
        """Get variable value with default."""
        # cannot get value of table directly, so, return default
        if self.is_table(key):
            return default
        return self._variables.get(key, default)

    def get_int(self, key: str, default: int = None) -> int:
        """Get variable value as integer with defaults on type conversion error."""
        # code trimmed down

    def get_float(self, key: str, default: float = None) -> float:
        """Get variable value as float with defaults on type conversion error."""
        # code trimmed down

    def get_bool(self, key: str, default: bool = None) -> bool:
        """Get variable value as bool with defaults on type conversion error."""
        # code trimmed down

    def get_list(self, key: str) -> List:
        """Get indexed elements of table as list of strings if variable is a table and indexed (keyless) elements present, empty list if no elements present or variable is not a table"""
        result = []
        for i in self.get_table_seq(key):
            result.append(self.get(f"{key}.{i}"))
        return result

    def get_table_start(self, key: str) -> int:
        """Get start indexed element index of table if variable is a table and indexed (keyless) elements present, 0 if no indexed elements present"""
        # code trimmed down

    def get_table_end(self, key: str) -> int:
        """Get end position of table if variable is a table, last indexable element is less than this number"""
        # code trimmed down

    def get_table_seq(self, key: str) -> List[int]:
        """Get sequence of table indices if variable is a table with indexed elements, intended to be used in for loops"""
        # code trimmed down

    def __repr__(self) -> str:
        """String representation."""
        return f"PyLuaHelper({len(self._variables)} variables)"

    def __str__(self) -> str:
        """String representation."""
        return f"PyLuaHelper with {len(self._variables)} exported variables"
```

Some examples, how to use `cfg` object:

```py3
cfg.get('server.listen_v4') # return "127.0.0.1:7777" string
cfg.get_type('server') # return "table" result
cfg.get_type('server.listen_v4') # return "string" result
cfg.get('server.listen_xxx','NOT FOUND') # return "NOT FOUND" string
```
