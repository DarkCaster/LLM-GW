# server/request_handler.py

import asyncio
import sys
import aiohttp.web
import python_lua_helper
from utils.logger import get_logger
from models import ModelSelector


class RequestHandler:
    """
    Process individual HTTP requests, coordinate model selection and engine management.
    """

    def __init__(
        self,
        model_selector: ModelSelector,
        cfg: python_lua_helper.PyLuaHelper,
    ):
        """
        Initialize RequestHandler.

        Args:
            model_selector: ModelSelector instance for selecting model variants
            cfg: PyLuaHelper configuration object
        """
        self.model_selector = model_selector
        self.cfg = cfg
        self.idle_timeout = sys.float_info.max
        self.request_lock = asyncio.Lock()
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("RequestHandler initialized")

    async def handle_models_list(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        """
        Handle OpenAI API /v1/models endpoint.

        Args:
            request: aiohttp web request

        Returns:
            JSON response with list of available models
        """
        self.logger.info("Handling /v1/models request")

        try:
            # Get list of models from model selector
            model_names = self.model_selector.list_models()

            # Build OpenAI-compatible response
            models_data = []
            for model_name in model_names:
                models_data.append(
                    {
                        "id": model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "system",
                    }
                )

            response_data = {"object": "list", "data": models_data}

            self.logger.info(f"Returning {len(models_data)} models")
            return aiohttp.web.json_response(response_data)

        except Exception as e:
            self.logger.error(f"Error handling /v1/models request: {e}")
            return aiohttp.web.json_response(
                {"error": {"message": str(e), "type": "internal_error"}}, status=500
            )

    async def handle_request(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        """
        Handle any other endpoints (like /v1/chat/completions).

        Args:
            request: aiohttp web request

        Returns:
            Response from the engine or error response
        """
        if self.request_lock.locked():
            self.logger.warning(
                "Request waiting for lock (another request in progress)"
            )
        async with self.request_lock:
            self.logger.debug("Acquired request lock, processing request")
            try:
                # Parse JSON body
                try:
                    request_data = await request.json()
                except Exception as e:
                    self.logger.error(f"Failed to parse JSON body: {e}")
                    return aiohttp.web.json_response(
                        {
                            "error": {
                                "message": f"Invalid JSON in request body: {e}",
                                "type": "invalid_request_error",
                            }
                        },
                        status=400,
                    )
                # Validate required fields
                if "model" not in request_data:
                    self.logger.error("Missing 'model' field in request")
                    return aiohttp.web.json_response(
                        {
                            "error": {
                                "message": "Missing required field: 'model'",
                                "type": "invalid_request_error",
                            }
                        },
                        status=400,
                    )
                # Extract model name
                model_name = request_data["model"]
                self.logger.info(f"Handling request for model '{model_name}'")
                # Extract endpoint
                path = request.path
                self.logger.debug(f"Request path: {path}")
                # Select appropriate variant
                try:
                    (
                        engine_client,
                        self.idle_timeout,
                    ) = await self.model_selector.select_variant(
                        model_name, request_data
                    )
                except ValueError as e:
                    self.logger.error(f"Model selection failed: {e}")
                    return aiohttp.web.json_response(
                        {
                            "error": {
                                "message": str(e),
                                "type": "invalid_request_error",
                            }
                        },
                        status=400,
                    )
                except Exception as e:
                    self.logger.error(f"Unexpected error during model selection: {e}")
                    return aiohttp.web.json_response(
                        {
                            "error": {
                                "message": f"Model selection error: {e}",
                                "type": "internal_error",
                            }
                        },
                        status=500,
                    )
                # Forward request to engine
                try:
                    engine_response = await engine_client.forward_request(
                        path, request_data
                    )
                except ValueError as e:
                    self.logger.error(f"Request forwarding failed: {e}")
                    return aiohttp.web.json_response(
                        {
                            "error": {
                                "message": str(e),
                                "type": "invalid_request_error",
                            }
                        },
                        status=400,
                    )
                except Exception as e:
                    self.logger.error(f"Unexpected error forwarding request: {e}")
                    return aiohttp.web.json_response(
                        {
                            "error": {
                                "message": f"Engine communication error: {e}",
                                "type": "internal_error",
                            }
                        },
                        status=502,
                    )
                # Return the engine response
                try:
                    content_type = engine_response.headers.get("Content-Type", "")
                    if "text/event-stream" in content_type or request_data.get(
                        "stream", False
                    ):
                        # Streaming response
                        self.logger.debug("Returning streaming response")
                        response = aiohttp.web.StreamResponse(
                            status=engine_response.status,
                            headers={"Content-Type": content_type},
                        )
                        await response.prepare(request)
                        # Stream the response
                        async for chunk in engine_response.content.iter_any():
                            await response.write(chunk)
                        await response.write_eof()
                        return response
                    else:
                        # Non-streaming response
                        self.logger.debug("Returning non-streaming response")
                        body = await engine_response.read()
                        return aiohttp.web.Response(
                            body=body,
                            status=engine_response.status,
                            headers={"Content-Type": content_type},
                        )
                finally:
                    engine_response.release()

            except Exception as e:
                self.logger.error(
                    f"Unexpected error handling request: {e}", exc_info=True
                )
                return aiohttp.web.json_response(
                    {
                        "error": {
                            "message": f"Internal server error: {e}",
                            "type": "internal_error",
                        }
                    },
                    status=500,
                )

            finally:
                # TODO: reset idle-watchdog here to self.idle_timeout
                self.logger.debug("Releasing request lock")
