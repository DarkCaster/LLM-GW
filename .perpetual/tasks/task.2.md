# Implementing Phase 2: Model Selection Logic (Middle Layer)

You need to implement `models/` sub-package classes accorting to the plan below.

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
