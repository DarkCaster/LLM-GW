# LLM gateway - runner for OpenAI compatible LLM engines

This project is a runner for a OpenAI API compatible LLM engines providing proxifying of LLM-requests (http+json) from entry point into LLM engine and vise versa.
It intercepts incoming inference request (like /v1/chat/completions), tokenize the request (if needed) and select proper engine variant with proper settings for the request depending on model name, context requirements, and other parameters.

## Project Structure

### Configuration

Configuration performed via lua config files using `python-lua-helper` package. See example.cfg.lua for current configuration structure. Main config parameters defined at `server` and `models` table.

`ConfigLoader` sub-package will parse and load lua configuration via python-lua-helper, making `server` and `model` tables available in the python application by use of helper `PyLuaHelper` class. All validation performed at lua side on configuration load time, so you not needed to re-validate configuration values at python code.

### Typical Workflow (simplified)

LLM gateway selects particular LLM engine and its configuration variant by incoming-request contents depending on model name and context size requirements. If new engine and configuration differs - it shuting down currently running engine and spawn a new engine with new configuration. Then it waits while engine http-entrypoint become available and then if forward incoming request to engine entrypoint and stream back the response with http error code.

NOTES:

- Main configuration unit is a model with unique name. LLM engine type with corresponding parameters and different confioguration variants tied to the model name.
- When forwarding request to the running engine, some request json fields may be changed on the fly for request and/or response to match particular LLM engine format.
- For the `llama.cpp` engine type, if there is already running engine instance present it can be used for pre-tokenization call to estimate context size requirements, and after that engine may be restarted with a more suitable configuration.
