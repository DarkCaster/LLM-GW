# Implementing Phase 5: Testing and Validation

## Step 1: Unit Tests - `tests/test_engine_client.py`

**Purpose:** Test base EngineClient and LlamaCppEngine functionality.

**Implementation Details:**

- Test abstract base class pattern
- Test LlamaCppEngine:
  - Mock aiohttp responses for tokenization endpoint
  - Test `estimate_tokens()` with various request formats
  - Test `transform_request()` with unsupported fields
  - Test `check_health()` with different server states
- Use `unittest.mock` for HTTP mocking
- Use async test methods with `asyncio`

## Step 2: Unit Tests - `tests/test_engine_process.py`

**Purpose:** Test EngineProcess subprocess management.

**Implementation Details:**

- Test process spawning with mock subprocess
- Test graceful and forceful shutdown
- Test status detection
- Test stdout/stderr capture
- Use mock subprocess to avoid spawning real processes

## Step 3: Unit Tests - `tests/test_engine_manager.py`

**Purpose:** Test EngineManager coordination logic.

**Implementation Details:**

- Test engine switching logic
- Test health check waiting
- Test concurrent request handling (locking)
- Mock EngineProcess and EngineClient
- Verify proper cleanup on shutdown

## Step 4: Unit Tests - `tests/test_model_selector.py`

**Purpose:** Test model selection and variant choosing logic.

**Implementation Details:**

- Test model parsing from configuration
- Test variant selection with various token requirements
- Test error handling for missing models
- Test fallback estimation when tokenization unavailable
- Mock engine manager and tokenization

## Step 5: Integration Tests - `tests/test_integration.py`

**Purpose:** Test end-to-end request flow.

**Implementation Details:**

- Test complete request flow from HTTP to engine
- Mock actual engine binary with a simple HTTP server
- Test configuration loading → server start → request handling → response
- Test engine switching on context size changes
- Test error handling at integration level
