"""
RequestHandler - Processes HTTP requests, coordinates model selection and engine management.

This module handles incoming OpenAI API requests, selects appropriate model variants,
ensures correct engines are running, and forwards requests to engines.
"""

import json
import logging
from typing import Dict, Any

import aiohttp
from aiohttp import web
import python_lua_helper

from models.model_selector import ModelSelector
from engine.engine_manager import EngineManager


class RequestHandler:
    """Handles incoming HTTP requests and coordinates model/engine operations."""

    def __init__(
        self,
        model_selector: ModelSelector,
        engine_manager: EngineManager,
        cfg: python_lua_helper.PyLuaHelper,
    ):
        """
        Initialize RequestHandler with required components.

        Args:
            model_selector: ModelSelector instance for variant selection
            engine_manager: EngineManager instance for engine lifecycle
            cfg: Configuration object for server-level settings
        """
        self.model_selector = model_selector
        self.engine_manager = engine_manager
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

    async def handle_chat_completion(self, request: web.Request) -> web.Response:
        """
        Handle /v1/chat/completions endpoint.

        Args:
            request: aiohttp HTTP request

        Returns:
            HTTP response with completion result or error
        """
        try:
            # Parse and validate request
            request_data = await self._parse_request(request)
            self._validate_chat_completion(request_data)

            # Extract model name and endpoint
            model_name = request_data["model"]
            endpoint = "/v1/chat/completions"
            request_data["endpoint"] = endpoint

            # Process the request
            return await self._process_request(request, model_name, request_data)

        except ValueError as e:
            self.logger.error(f"Validation error: {e}")
            return self._error_response(400, str(e))
        except KeyError as e:
            self.logger.error(f"Missing required field: {e}")
            return self._error_response(400, f"Missing required field: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON: {e}")
            return self._error_response(400, "Invalid JSON in request body")
        except Exception as e:
            self.logger.error(f"Unexpected error in chat completion: {e}")
            return self._error_response(500, f"Internal server error: {str(e)}")

    async def handle_completion(self, request: web.Request) -> web.Response:
        """
        Handle /v1/completions endpoint.

        Args:
            request: aiohttp HTTP request

        Returns:
            HTTP response with completion result or error
        """
        try:
            # Parse and validate request
            request_data = await self._parse_request(request)
            self._validate_completion(request_data)

            # Extract model name and endpoint
            model_name = request_data["model"]
            endpoint = "/v1/completions"
            request_data["endpoint"] = endpoint

            # Process the request
            return await self._process_request(request, model_name, request_data)

        except ValueError as e:
            self.logger.error(f"Validation error: {e}")
            return self._error_response(400, str(e))
        except KeyError as e:
            self.logger.error(f"Missing required field: {e}")
            return self._error_response(400, f"Missing required field: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON: {e}")
            return self._error_response(400, "Invalid JSON in request body")
        except Exception as e:
            self.logger.error(f"Unexpected error in completion: {e}")
            return self._error_response(500, f"Internal server error: {str(e)}")

    async def handle_models_list(self, request: web.Request) -> web.Response:
        """
        Handle GET /v1/models endpoint.

        Args:
            request: aiohttp HTTP request

        Returns:
            HTTP response with list of available models
        """
        try:
            models = self.model_selector.get_available_models()

            # Build OpenAI-compatible response
            response_data = {"object": "list", "data": models}

            self.logger.debug(f"Returning models list with {len(models)} models")
            return web.json_response(response_data)

        except Exception as e:
            self.logger.error(f"Error getting models list: {e}")
            return self._error_response(500, f"Internal server error: {str(e)}")

    async def handle_model_info(self, request: web.Request) -> web.Response:
        """
        Handle GET /v1/models/{model_id} endpoint.

        Args:
            request: aiohttp HTTP request with model_id in path

        Returns:
            HTTP response with model information
        """
        try:
            model_id = request.match_info.get("model_id")
            if not model_id:
                return self._error_response(400, "Missing model_id in path")

            # Get model info from selector
            model_info = self.model_selector.get_model_info(model_id)

            self.logger.debug(f"Returning info for model: {model_id}")
            return web.json_response(model_info)

        except ValueError as e:
            self.logger.error(f"Model not found: {e}")
            return self._error_response(404, str(e))
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return self._error_response(500, f"Internal server error: {str(e)}")

    async def _process_request(
        self, request: web.Request, model_name: str, request_data: Dict[str, Any]
    ) -> web.Response:
        """
        Process a request by selecting variant, ensuring engine, and forwarding.

        Args:
            request: Original HTTP request (for streaming detection)
            model_name: Name of the model to use
            request_data: Request data dictionary

        Returns:
            HTTP response from engine or error response
        """
        try:
            # Select appropriate variant
            variant_config = await self.model_selector.select_variant(
                model_name=model_name,
                request_data=request_data,
                engine_manager=self.engine_manager,
            )

            # Get engine type for the model
            engine_type = self.model_selector.get_model_engine_type(model_name)

            # Ensure correct engine is running
            engine_client = await self.engine_manager.ensure_engine(
                model_name=model_name,
                variant_config=variant_config,
                engine_type=engine_type,
            )

            # Check if endpoint is supported
            endpoint = request_data.get("endpoint", "/v1/chat/completions")
            if endpoint not in engine_client.get_supported_endpoints():
                self.logger.error(f"Endpoint {endpoint} not supported by engine")
                return self._error_response(
                    400, f"Endpoint {endpoint} is not supported by this engine"
                )

            # Check if streaming is requested
            stream = request_data.get("stream", False)

            if stream:
                return await self._handle_streaming_request(
                    engine_client=engine_client,
                    endpoint=endpoint,
                    request_data=request_data,
                )
            else:
                return await self._handle_non_streaming_request(
                    engine_client=engine_client,
                    endpoint=endpoint,
                    request_data=request_data,
                )

        except ValueError as e:
            self.logger.error(f"Model/variant selection error: {e}")
            return self._error_response(400, str(e))
        except RuntimeError as e:
            self.logger.error(f"Engine management error: {e}")
            return self._error_response(503, f"Engine unavailable: {str(e)}")
        except aiohttp.ClientError as e:
            self.logger.error(f"Engine communication error: {e}")
            return self._error_response(502, f"Bad gateway: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error in request processing: {e}")
            return self._error_response(500, f"Internal server error: {str(e)}")

    async def _handle_streaming_request(
        self, engine_client: Any, endpoint: str, request_data: Dict[str, Any]
    ) -> web.StreamResponse:
        """
        Handle streaming request with Server-Sent Events.

        Args:
            engine_client: EngineClient instance
            endpoint: API endpoint path
            request_data: Request data

        Returns:
            StreamResponse with SSE-formatted chunks
        """
        self.logger.debug("Handling streaming request")

        # Create stream response
        response = web.StreamResponse()
        response.headers["Content-Type"] = "text/event-stream"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"

        await response.prepare(request)

        try:
            # Forward request to engine
            async with aiohttp.ClientSession() as session:
                engine_response = await engine_client.forward_request(
                    session=session,
                    endpoint=endpoint,
                    request_data=request_data,
                    timeout=300.0,  # Longer timeout for streaming
                )

                # Stream chunks to client
                async for chunk in engine_response.content:
                    if chunk:
                        # Format as SSE event
                        sse_data = f"data: {chunk.decode('utf-8')}\n\n"
                        await response.write(sse_data.encode("utf-8"))

                await response.write_eof()

        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            try:
                error_data = json.dumps(
                    {
                        "error": {
                            "message": f"Stream interrupted: {str(e)}",
                            "type": "stream_error",
                        }
                    }
                )
                sse_error = f"data: {error_data}\n\n"
                await response.write(sse_error.encode("utf-8"))
                await response.write_eof()
            except:
                pass  # Client may have disconnected

        return response

    async def _handle_non_streaming_request(
        self, engine_client: Any, endpoint: str, request_data: Dict[str, Any]
    ) -> web.Response:
        """
        Handle non-streaming request.

        Args:
            engine_client: EngineClient instance
            endpoint: API endpoint path
            request_data: Request data

        Returns:
            JSON response from engine
        """
        self.logger.debug("Handling non-streaming request")

        try:
            # Forward request to engine
            async with aiohttp.ClientSession() as session:
                engine_response = await engine_client.forward_request(
                    session=session,
                    endpoint=endpoint,
                    request_data=request_data,
                    timeout=60.0,
                )

                # Read full response
                response_text = await engine_response.text()
                response_data = json.loads(response_text)

                # Apply response transformation if needed
                if hasattr(engine_client, "transform_response"):
                    response_data = engine_client.transform_response(response_data)

                # Return as JSON response
                return web.json_response(response_data, status=engine_response.status)

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON from engine: {e}")
            return self._error_response(502, "Invalid response from engine")
        except Exception as e:
            self.logger.error(f"Error in non-streaming request: {e}")
            raise

    async def _parse_request(self, request: web.Request) -> Dict[str, Any]:
        """
        Parse JSON request body.

        Args:
            request: HTTP request

        Returns:
            Parsed request data dictionary

        Raises:
            json.JSONDecodeError: If JSON is invalid
        """
        try:
            return await request.json()
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            raise

    def _validate_chat_completion(self, request_data: Dict[str, Any]) -> None:
        """
        Validate chat completion request.

        Args:
            request_data: Request data dictionary

        Raises:
            ValueError: If validation fails
            KeyError: If required fields are missing
        """
        # Check required fields
        if "model" not in request_data:
            raise KeyError("model")
        if "messages" not in request_data:
            raise KeyError("messages")

        # Validate messages
        messages = request_data["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("Messages must be a non-empty list")

        # Validate each message
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Message at index {i} must be an object")
            if "role" not in message:
                raise ValueError(f"Message at index {i} missing 'role' field")
            if "content" not in message:
                raise ValueError(f"Message at index {i} missing 'content' field")

    def _validate_completion(self, request_data: Dict[str, Any]) -> None:
        """
        Validate text completion request.

        Args:
            request_data: Request data dictionary

        Raises:
            ValueError: If validation fails
            KeyError: If required fields are missing
        """
        # Check required fields
        if "model" not in request_data:
            raise KeyError("model")
        if "prompt" not in request_data:
            raise KeyError("prompt")

        # Validate prompt
        prompt = request_data["prompt"]
        if not isinstance(prompt, (str, list)):
            raise ValueError("Prompt must be a string or list of strings")
        if isinstance(prompt, list) and len(prompt) == 0:
            raise ValueError("Prompt list must not be empty")

    def _error_response(
        self, status: int, message: str, error_type: str = "invalid_request_error"
    ) -> web.Response:
        """
        Create OpenAI-compatible error response.

        Args:
            status: HTTP status code
            message: Error message
            error_type: OpenAI error type

        Returns:
            HTTP response with error JSON
        """
        error_data = {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        }

        self.logger.debug(f"Returning error: {status} - {message}")
        return web.json_response(error_data, status=status)
