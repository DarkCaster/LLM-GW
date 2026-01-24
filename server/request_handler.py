# server/request_handler.py

import asyncio
import sys
import aiohttp.web
import python_lua_helper

from .idle_watchdog import IdleWatchdog
from engine import EngineManager
from models import ModelSelector
from utils.logger import get_logger


class RequestHandler:
    """
    Process individual HTTP requests, coordinate model selection and engine management.
    """

    def __init__(
        self,
        model_selector: ModelSelector,
        engine_manager: EngineManager,
        idle_watchdog: IdleWatchdog,
        cfg: python_lua_helper.PyLuaHelper,
    ):
        """
        Initialize RequestHandler.

        Args:
            model_selector: ModelSelector instance for selecting model variants
            cfg: PyLuaHelper configuration object
        """
        self.disconnect_check_interval = 1.0  # constant for now
        self.model_selector = model_selector
        self.engine_manager = engine_manager
        self.idle_watchdog = idle_watchdog
        self.cfg = cfg
        self.idle_timeout = sys.float_info.max
        self._is_disposed = False
        self._request_lock = asyncio.Lock()
        self.logger = get_logger(self.__class__.__name__)

    def _is_client_connected(self, request: aiohttp.web.Request) -> bool:
        """
        Check if the client is still connected.

        Args:
            request: aiohttp web request

        Returns:
            True if client is connected, False otherwise
        """
        try:
            # Check if transport is available and not closing
            if request.transport is None:
                return False
            if request.transport.is_closing():
                return False
            return True
        except Exception as e:
            self.logger.debug(f"Error checking client connection: {e}")
            return False

    async def _monitor_client_connection(
        self, request: aiohttp.web.Request, cancel_event: asyncio.Event
    ) -> None:
        """
        Monitor client connection and set cancel_event if disconnect is detected.

        Args:
            request: aiohttp web request
            cancel_event: Event to set when disconnect is detected
        """
        try:
            while not cancel_event.is_set():
                if not self._is_client_connected(request):
                    self.logger.info(f"Client disconnect detected for {request.path}")
                    cancel_event.set()
                    break
                await asyncio.sleep(self.disconnect_check_interval)
        except asyncio.CancelledError:
            # Monitor task was cancelled, this is normal shutdown
            pass
        except Exception as e:
            self.logger.error(f"Error in connection monitor: {e}")

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
        if self._request_lock.locked():
            self.logger.warning(
                "Request waiting for lock (another request in progress)"
            )
        async with self._request_lock:
            self.logger.debug("Acquired request lock, processing request")
            if self._is_disposed:
                return aiohttp.web.json_response(
                    {
                        "error": {
                            "message": "RequestHandler is shuting down",
                            "type": "internal_error",
                        }
                    },
                    status=500,
                )
            self.idle_watchdog.disarm()
            # Create event for client disconnect detection
            disconnect_event = asyncio.Event()
            monitor_task = None
            try:
                # Start monitoring client connection
                monitor_task = asyncio.create_task(
                    self._monitor_client_connection(request, disconnect_event)
                )
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
                # Check for disconnect before proceeding
                if disconnect_event.is_set():
                    self.logger.info("Client disconnected before request processing")
                    return aiohttp.web.Response(status=499)  # Client Closed Request
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
                    # Check for disconnect after model selection
                    if disconnect_event.is_set():
                        self.logger.info("Client disconnected during model selection")
                        return aiohttp.web.Response(status=499)
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
                engine_response = None
                try:
                    engine_response = await engine_client.forward_request(
                        path, request_data
                    )
                    # Check for disconnect after getting engine response
                    if disconnect_event.is_set():
                        self.logger.info(
                            "Client disconnected after engine response received"
                        )
                        return aiohttp.web.Response(status=499)
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
                        # Stream the response with disconnect detection
                        try:
                            async for chunk in engine_response.content.iter_any():
                                # Check for client disconnect before writing each chunk
                                if disconnect_event.is_set():
                                    self.logger.info(
                                        "Client disconnected during streaming response"
                                    )
                                    break
                                try:
                                    await response.write(chunk)
                                except (ConnectionResetError, BrokenPipeError) as e:
                                    self.logger.info(
                                        f"Client connection error during streaming: {e}"
                                    )
                                    disconnect_event.set()
                                    break
                                except Exception as e:
                                    self.logger.error(
                                        f"Error writing chunk to client: {e}"
                                    )
                                    disconnect_event.set()
                                    break
                            if not disconnect_event.is_set():
                                await response.write_eof()
                        except asyncio.CancelledError:
                            self.logger.info(
                                "Streaming response cancelled (client disconnect)"
                            )
                            raise
                        return response
                    else:
                        # Non-streaming response
                        self.logger.debug("Returning non-streaming response")
                        # Check for disconnect before reading response body
                        if disconnect_event.is_set():
                            self.logger.info(
                                "Client disconnected before reading engine response"
                            )
                            return aiohttp.web.Response(status=499)
                        body = await engine_response.read()
                        # Check for disconnect after reading response body
                        if disconnect_event.is_set():
                            self.logger.info(
                                "Client disconnected after reading engine response"
                            )
                            return aiohttp.web.Response(status=499)
                        return aiohttp.web.Response(
                            body=body,
                            status=engine_response.status,
                            headers={"Content-Type": content_type},
                        )
                finally:
                    if engine_response is not None:
                        engine_response.release()
            except asyncio.CancelledError:
                self.logger.info("Request cancelled (likely due to client disconnect)")
                # Re-raise to properly propagate cancellation
                raise
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
                # Cancel and cleanup monitor task
                if monitor_task is not None and not monitor_task.done():
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass
                self.idle_watchdog.rearm(self.idle_timeout, self.handle_idle_timeout)
                self.logger.debug("Releasing request lock")

    async def handle_idle_timeout(self):
        """
        Handle idle timeout, when no incoming requests received in specified time.
        """
        if self._request_lock.locked():
            self.logger.warning(
                "Idle timeout handler waiting for lock (another request in progress)"
            )
        async with self._request_lock:
            if self._is_disposed:
                return
            await self.engine_manager.stop_current_engine()

    async def shutdown(self) -> None:
        """
        Shutdown, deinitialize stuff
        """
        if self._request_lock.locked():
            self.logger.warning(
                "Idle timeout handler waiting for lock (another request in progress)"
            )
        async with self._request_lock:
            if self._is_disposed:
                return
            self._is_disposed = True
            self.idle_watchdog.disarm()
