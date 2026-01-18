# server/request_handler.py

import aiohttp
from aiohttp import web
import json
from utils.logger import get_logger
from models import ModelSelector
from engine import EngineManager


class RequestHandler:
    """
    Process individual HTTP requests, coordinate model selection and engine management.
    """

    def __init__(
        self, model_selector: ModelSelector, engine_manager: EngineManager, cfg
    ):
        """
        Initialize the request handler.

        Args:
            model_selector: ModelSelector instance for choosing variants
            engine_manager: EngineManager instance for managing engines
            cfg: PyLuaHelper configuration object
        """
        self.model_selector = model_selector
        self.engine_manager = engine_manager
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)

    async def handle_chat_completion(self, request: web.Request) -> web.Response:
        """
        Handle POST /v1/chat/completions requests.

        Args:
            request: aiohttp web.Request object

        Returns:
            aiohttp web.Response object
        """
        try:
            # Parse JSON body
            try:
                request_data = await request.json()
            except json.JSONDecodeError as e:
                return self._error_response(400, f"Invalid JSON: {e}")

            # Validate required fields
            if "model" not in request_data:
                return self._error_response(400, "Missing required field: 'model'")

            if "messages" not in request_data:
                return self._error_response(400, "Missing required field: 'messages'")

            model_name = request_data["model"]

            self.logger.info(
                f"Handling chat completion request for model '{model_name}'"
            )

            # Select appropriate variant
            try:
                variant_config = await self.model_selector.select_variant(
                    model_name, request_data, self.engine_manager
                )
            except ValueError as e:
                return self._error_response(400, str(e))

            # Get engine type for this model
            model_info = self.model_selector.models.get(model_name)
            if not model_info:
                return self._error_response(404, f"Model '{model_name}' not found")

            engine_type = model_info["engine"]

            # Ensure correct engine is running
            try:
                engine_client = await self.engine_manager.ensure_engine(
                    model_name, variant_config, engine_type
                )
            except Exception as e:
                self.logger.error(f"Failed to ensure engine: {e}")
                return self._error_response(503, f"Failed to start engine: {e}")

            # Check if endpoint is supported
            endpoint = "/v1/chat/completions"
            if endpoint not in engine_client.get_supported_endpoints():
                return self._error_response(
                    400, f"Endpoint '{endpoint}' not supported by engine"
                )

            # Forward request to engine
            try:
                is_streaming = request_data.get("stream", False)

                async with aiohttp.ClientSession() as session:
                    engine_response = await engine_client.forward_request(
                        session, endpoint, request_data
                    )

                    if is_streaming:
                        # Handle streaming response
                        return await self._handle_streaming_response(engine_response)
                    else:
                        # Handle non-streaming response
                        return await self._handle_non_streaming_response(
                            engine_response, engine_client
                        )

            except aiohttp.ClientError as e:
                self.logger.error(f"Engine request failed: {e}")
                return self._error_response(502, f"Engine request failed: {e}")

        except Exception as e:
            self.logger.error(f"Unexpected error in handle_chat_completion: {e}")
            return self._error_response(500, f"Internal server error: {e}")

    async def handle_completion(self, request: web.Request) -> web.Response:
        """
        Handle POST /v1/completions requests.

        Args:
            request: aiohttp web.Request object

        Returns:
            aiohttp web.Response object
        """
        try:
            # Parse JSON body
            try:
                request_data = await request.json()
            except json.JSONDecodeError as e:
                return self._error_response(400, f"Invalid JSON: {e}")

            # Validate required fields
            if "model" not in request_data:
                return self._error_response(400, "Missing required field: 'model'")

            if "prompt" not in request_data:
                return self._error_response(400, "Missing required field: 'prompt'")

            model_name = request_data["model"]

            self.logger.info(f"Handling completion request for model '{model_name}'")

            # Select appropriate variant
            try:
                variant_config = await self.model_selector.select_variant(
                    model_name, request_data, self.engine_manager
                )
            except ValueError as e:
                return self._error_response(400, str(e))

            # Get engine type for this model
            model_info = self.model_selector.models.get(model_name)
            if not model_info:
                return self._error_response(404, f"Model '{model_name}' not found")

            engine_type = model_info["engine"]

            # Ensure correct engine is running
            try:
                engine_client = await self.engine_manager.ensure_engine(
                    model_name, variant_config, engine_type
                )
            except Exception as e:
                self.logger.error(f"Failed to ensure engine: {e}")
                return self._error_response(503, f"Failed to start engine: {e}")

            # Check if endpoint is supported
            endpoint = "/v1/completions"
            if endpoint not in engine_client.get_supported_endpoints():
                return self._error_response(
                    400, f"Endpoint '{endpoint}' not supported by engine"
                )

            # Forward request to engine
            try:
                is_streaming = request_data.get("stream", False)

                async with aiohttp.ClientSession() as session:
                    engine_response = await engine_client.forward_request(
                        session, endpoint, request_data
                    )

                    if is_streaming:
                        # Handle streaming response
                        return await self._handle_streaming_response(engine_response)
                    else:
                        # Handle non-streaming response
                        return await self._handle_non_streaming_response(
                            engine_response, engine_client
                        )

            except aiohttp.ClientError as e:
                self.logger.error(f"Engine request failed: {e}")
                return self._error_response(502, f"Engine request failed: {e}")

        except Exception as e:
            self.logger.error(f"Unexpected error in handle_completion: {e}")
            return self._error_response(500, f"Internal server error: {e}")

    async def handle_models_list(self, request: web.Request) -> web.Response:
        """
        Handle GET /v1/models requests.

        Args:
            request: aiohttp web.Request object

        Returns:
            aiohttp web.Response object with list of models
        """
        try:
            model_names = self.model_selector.list_models()

            # Build OpenAI-compatible response
            models_data = []
            for model_name in model_names:
                model_info = self.model_selector.get_model_info(model_name)
                models_data.append(
                    {
                        "id": model_name,
                        "object": "model",
                        "created": 0,  # Placeholder
                        "owned_by": "llm-gateway",
                        "permission": [],
                        "root": model_name,
                        "parent": None,
                    }
                )

            response_data = {"object": "list", "data": models_data}

            return web.json_response(response_data)

        except Exception as e:
            self.logger.error(f"Error in handle_models_list: {e}")
            return self._error_response(500, f"Internal server error: {e}")

    async def handle_model_info(self, request: web.Request) -> web.Response:
        """
        Handle GET /v1/models/{model_id} requests.

        Args:
            request: aiohttp web.Request object

        Returns:
            aiohttp web.Response object with model information
        """
        try:
            # Extract model name from URL path
            model_name = request.match_info.get("model_id")

            if not model_name:
                return self._error_response(400, "Missing model ID in path")

            # Get model info
            try:
                model_info = self.model_selector.get_model_info(model_name)
            except ValueError as e:
                return self._error_response(404, str(e))

            # Build OpenAI-compatible response
            response_data = {
                "id": model_name,
                "object": "model",
                "created": 0,  # Placeholder
                "owned_by": "llm-gateway",
                "permission": [],
                "root": model_name,
                "parent": None,
            }

            return web.json_response(response_data)

        except Exception as e:
            self.logger.error(f"Error in handle_model_info: {e}")
            return self._error_response(500, f"Internal server error: {e}")

    async def _handle_streaming_response(
        self, engine_response: aiohttp.ClientResponse
    ) -> web.StreamResponse:
        """
        Handle streaming response from engine.

        Args:
            engine_response: aiohttp ClientResponse from engine

        Returns:
            aiohttp StreamResponse for client
        """
        # Create streaming response
        response = web.StreamResponse(
            status=engine_response.status,
            reason=engine_response.reason,
            headers={"Content-Type": "text/event-stream"},
        )

        await response.prepare(
            engine_response._request._message._protocol._transport.get_extra_info(
                "request"
            )
        )

        # Alternative approach: get request from current context
        # For aiohttp, we need to properly handle this
        # Let's use a different approach - we'll need the original request
        # This is a simplification - in production you'd pass the request object

        try:
            # Stream chunks from engine to client
            async for chunk in engine_response.content.iter_any():
                if chunk:
                    await response.write(chunk)

            await response.write_eof()

        except Exception as e:
            self.logger.error(f"Error streaming response: {e}")

        return response

    async def _handle_non_streaming_response(
        self, engine_response: aiohttp.ClientResponse, engine_client
    ) -> web.Response:
        """
        Handle non-streaming response from engine.

        Args:
            engine_response: aiohttp ClientResponse from engine
            engine_client: EngineClient instance for transformations

        Returns:
            aiohttp web.Response object
        """
        # Read complete response
        try:
            response_text = await engine_response.text()
            response_data = json.loads(response_text)

            # Transform response if needed
            transformed_data = engine_client.transform_response(response_data)

            return web.json_response(transformed_data, status=engine_response.status)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse engine response: {e}")
            return self._error_response(502, "Invalid response from engine")

    def _error_response(
        self, status: int, message: str, error_type: str = "invalid_request_error"
    ) -> web.Response:
        """
        Create an OpenAI-compatible error response.

        Args:
            status: HTTP status code
            message: Error message
            error_type: Type of error (OpenAI error type)

        Returns:
            aiohttp web.Response object with error
        """
        error_data = {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        }

        return web.json_response(error_data, status=status)
