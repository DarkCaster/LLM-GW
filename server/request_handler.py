# server/request_handler.py

import asyncio
import sys
import json
import aiohttp.web
import python_lua_helper
import logger
from .idle_watchdog import IdleWatchdog
from .dump_writer import DumpWriter
from engine import EngineClient
from engine import EngineManager
from models import ModelSelector


class RequestHandler:
    """
    Process individual HTTP requests, coordinate model selection and engine management.
    """

    def __init__(
        self,
        model_selector: ModelSelector,
        primary_engine_manager: EngineManager,
        secondary_engine_manager: EngineManager,
        cfg: python_lua_helper.PyLuaHelper,
    ):
        """
        Initialize RequestHandler.

        Args:
            model_selector: ModelSelector instance for selecting model variants
            primary_engine_manager: EngineManager instance for managing engine lifecycle
            secondary_engine_manager: EngineManager instance for managing engine lifecycle
            cfg: PyLuaHelper configuration object
            dumps_dir: Optional directory path for dumping requests/responses
        """
        self._model_selector = model_selector
        self._primary_engine_manager = primary_engine_manager
        self._secondary_engine_manager = secondary_engine_manager
        self._primary_idle_watchdog = IdleWatchdog()
        self._secondary_idle_watchdog = IdleWatchdog()
        self._cfg = cfg
        self._idle_timeout = sys.float_info.max
        self._is_disposed = False
        self._is_stopped = False
        self._request_lock = asyncio.Lock()
        self._primary_idle_lock = asyncio.Lock()
        self._secondary_idle_lock = asyncio.Lock()
        # disconnection logic
        self._disconnect_check_interval = 1.0
        self._monitor_task = None
        self._disconnect_event = None
        # logger
        self.logger = logger.get_logger(self.__class__.__name__)

    def _extract_model_name_from_text(self, request_text: str) -> str:
        """
        Attempt to extract model name from request text.

        Args:
            request_text: Raw request text

        Returns:
            Model name if found, otherwise "unknown"
        """
        try:
            data = json.loads(request_text)
            if isinstance(data, dict) and "model" in data:
                return str(data["model"])
        except Exception:
            pass
        return "unknown"

    def _get_model_index(self, model_name: str) -> int:
        model_index = None
        for i in self._cfg.get_table_seq("models"):
            if self._cfg.get(f"models.{i}.name") == model_name:
                model_index = i
                break
        if model_index is None:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        return model_index

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

    async def _monitor_task_worker(
        self, request: aiohttp.web.Request, engine_client: EngineClient
    ) -> None:
        if not self._disconnect_event:
            raise RuntimeError("Internal error: self._disconnect_event is not set")
        try:
            while not self._disconnect_event.is_set():
                if not self._is_client_connected(request):
                    self.logger.info(f"Client disconnect detected for {request.path}")
                    self._disconnect_event.set()
                    break
                await asyncio.sleep(self._disconnect_check_interval)
        except Exception as e:
            self.logger.error(f"Error in connection monitor: {e}")
        engine_client.terminate_request()

    def _start_monitoring_task(
        self, request: aiohttp.web.Request, engine_client: EngineClient
    ) -> None:
        # create new monitoring event and task, add extra checks
        if self._disconnect_event is not None:
            raise RuntimeError(
                "Internal error: self._disconnect_event already initialized"
            )
        if self._monitor_task is not None:
            raise RuntimeError("Internal error: self._monitor_task already started")
        self._disconnect_event = asyncio.Event()
        self._monitor_task = asyncio.create_task(
            self._monitor_task_worker(request, engine_client)
        )

    async def _stop_monitoring_task(self) -> None:
        # Cancel and cleanup monitor task
        if self._monitor_task is None:
            self.logger.debug("Monitoring task was not running!")
            return
        if self._disconnect_event is None:
            raise RuntimeError(
                "Internal error: self._disconnect_event was not created!"
            )
        # Command monitoring to stop, and also terminate processing for engine_client
        self._disconnect_event.set()
        try:
            await self._monitor_task
        except Exception as e:
            self.logger.error(f"Error while awaiting monitoring task to stop: {e}")
        if not self._monitor_task.done():
            self.logger.error(
                "Internal error: monitoring task not done yet (should not happen)!"
            )
        self._monitor_task = None
        self._disconnect_event = None

    def _return_error(
        self, message: str, code: int, e: Exception | None = None
    ) -> aiohttp.web.json_response:
        if e is not None:
            message = f"{message}: {e}"
            self.logger.error(message)
        else:
            self.logger.warning(message)
        return aiohttp.web.json_response(
            {"error": {"message": message, "type": "invalid_request_error"}},
            status=code,
        )

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
        if self._is_stopped or self._is_disposed:
            return self._return_error("RequestHandler is shuting down", 500)
        self.logger.info("Handling /v1/models request")
        try:
            # Get list of models from model selector
            model_names = self._model_selector.list_models()
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
            return self._return_error("Error handling /v1/models request", 500, e)

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
        if self._is_stopped or self._is_disposed:
            return self._return_error("RequestHandler is shuting down", 500)
        # if self._request_lock.locked():
        #     self.logger.warning(
        #         "Request waiting for lock (another request in progress)"
        #    )
        async with self._request_lock:
            self.logger.debug("Acquired request lock, processing request")
            if self._is_stopped or self._is_disposed:
                return self._return_error("RequestHandler is shuting down", 500)
            # Initialize dump writer if dumps_dir is configured
            dump_writer = None
            request_text = None
            is_primary = None
            try:
                # Read raw request text
                try:
                    request_text = await request.text()
                except Exception as e:
                    self.logger.error(f"Failed to read request text: {e}")
                    if self._cfg.get("server.dumps_dir") is not None:
                        dump_writer = DumpWriter(
                            self._cfg.get("server.dumps_dir"), "request_read_error"
                        )
                        dump_writer.write_error(e)
                    return self._return_error("Failed to read request body", 400, e)
                # Parse JSON from the already-read text
                try:
                    request_data = json.loads(request_text)
                except Exception as e:
                    self.logger.error(f"Failed to parse JSON body: {e}")
                    if self._cfg.get("server.dumps_dir") is not None:
                        dump_writer = DumpWriter(
                            self._cfg.get("server.dumps_dir"), "request_parse_error"
                        )
                        dump_writer.write_request(request_text)
                        dump_writer.write_error(e)
                    return self._return_error("Invalid JSON in request body", 400, e)
                # Now we have a request_data, parse model name from it
                if "model" not in request_data:
                    self.logger.error("Missing 'model' field in request")
                    if self._cfg.get("server.dumps_dir") is not None:
                        dump_writer = DumpWriter(
                            self._cfg.get("server.dumps_dir"), "model_missing_error"
                        )
                        dump_writer.write_request(request_text)
                        dump_writer.write_error(
                            ValueError("Missing required field: 'model'")
                        )
                    return self._return_error("Missing required field: 'model'", 400)
                # Extract model name and engine parameters
                model_name = request_data["model"]
                model_index = self._get_model_index(model_name)
                is_primary = self._cfg.get_bool(f"models.{model_index}.primary", True)
                # Disarm primary or secondary _idle_watchdog (idle-trigger task should also have primary or secondary lock)
                if is_primary:
                    async with self._primary_idle_lock:
                        self._primary_idle_watchdog.disarm()
                else:
                    async with self._secondary_idle_lock:
                        self._secondary_idle_watchdog.disarm()
                # Now create proper dump_writer with parsed model name
                if self._cfg.get("server.dumps_dir") is not None:
                    dump_writer = DumpWriter(
                        self._cfg.get("server.dumps_dir"), model_name
                    )
                    # Write JSON request with human readable indentation (reserialize it)
                    try:
                        formatted_request = json.dumps(
                            request_data, indent=2, ensure_ascii=False
                        )
                        dump_writer.write_request(formatted_request)
                    except Exception as e:
                        self.logger.error(
                            f"Failed to write formatted request to dump: {e}"
                        )
                        # Fallback to raw request text if formatting fails
                        dump_writer.write_request(request_text)
                # Handle request
                self.logger.info(f"Handling request for model '{model_name}'")
                # Extract endpoint
                path = request.path
                self.logger.debug(f"Request path: {path}")
                # Select appropriate variant
                try:
                    (
                        engine_client,
                        self._idle_timeout,
                    ) = await self._model_selector.select_variant(
                        path, model_name, request_data
                    )
                except ValueError as e:
                    if dump_writer:
                        dump_writer.write_error(e)
                    return self._return_error("Model selection failed", 400, e)
                except Exception as e:
                    if dump_writer:
                        dump_writer.write_error(e)
                    return self._return_error(
                        "Unexpected error during model selection", 500, e
                    )
                # Start monitoring for client connection, add engine_client
                self._start_monitoring_task(request, engine_client)
                # Forward request to engine
                engine_response = None
                try:
                    engine_response = await engine_client.forward_request(
                        path, request_data
                    )
                    self.logger.debug("Engine response received")
                except ValueError as e:
                    if dump_writer:
                        dump_writer.write_error(e)
                    return self._return_error("Request forwarding failed", 400, e)
                except Exception as e:
                    if dump_writer:
                        dump_writer.write_error(e)
                    return self._return_error("Engine communication error", 502, e)
                # Process the engine response
                try:
                    content_type = engine_response.headers.get("Content-Type", "")
                    if "text/event-stream" in content_type or request_data.get(
                        "stream", False
                    ):
                        # Streaming response
                        self.logger.debug("Returning streaming response")
                        # Write response header to dump if dumping is enabled
                        if dump_writer:
                            dump_writer.write_response_start()
                        response = aiohttp.web.StreamResponse(
                            status=engine_response.status,
                            headers={"Content-Type": content_type},
                        )
                        await response.prepare(request)
                        # Stream the response with disconnect detection
                        try:
                            write_ok = True
                            async for chunk in engine_response.content.iter_any():
                                try:
                                    await response.write(chunk)
                                    # Write chunk to dump file
                                    if dump_writer:
                                        dump_writer.write_response_chunk(chunk)
                                except (ConnectionResetError, BrokenPipeError) as e:
                                    self.logger.info(
                                        f"Client connection error during streaming: {e}"
                                    )
                                    write_ok = False
                                    break
                                except Exception as e:
                                    self.logger.error(
                                        f"Error writing chunk to client: {e}"
                                    )
                                    write_ok = False
                                    break
                            if write_ok:
                                await response.write_eof()
                            # Write response footer to dump if dumping is enabled
                            if dump_writer:
                                dump_writer.write_response_end()
                        except asyncio.CancelledError:
                            self.logger.info(
                                "Streaming response cancelled (client disconnect)"
                            )
                            raise
                        return response
                    else:
                        # Non-streaming response
                        self.logger.debug("Returning non-streaming response")
                        body = await engine_response.read()
                        # Write response to dump if dumping is enabled
                        if dump_writer:
                            try:
                                response_text = body.decode("utf-8", errors="replace")
                            except Exception as e:
                                self.logger.error(
                                    f"Failed to decode response for dump: {e}"
                                )
                                dump_writer.write_error(e)
                                response_text = None
                            try:
                                if response_text is not None:
                                    # Write JSON response to dump with human readable indentation (reserialize it)
                                    response_json = json.loads(response_text)
                                    formatted_response = json.dumps(
                                        response_json, indent=2, ensure_ascii=False
                                    )
                                    dump_writer.write_response(formatted_response)
                            except Exception as e:
                                self.logger.error(
                                    f"Failed to write formatted response to dump: {e}"
                                )
                                dump_writer.write_error(e)
                                if response_text is not None:
                                    dump_writer.write_response(response_text)
                        # Return response to the client
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
                if dump_writer:
                    dump_writer.write_error(e)
                return self._return_error("Internal server error", 500, e)
            finally:
                # Close dump writer if it was created
                if dump_writer:
                    dump_writer.close()
                await self._stop_monitoring_task()
                # Restart idle watchdog for primary or secondary engine
                if is_primary is not None:
                    if is_primary:
                        self._primary_idle_watchdog.rearm(
                            self._idle_timeout, lambda: self.handle_idle_timeout(True)
                        )
                    else:
                        self._secondary_idle_watchdog.rearm(
                            self._idle_timeout, lambda: self.handle_idle_timeout(False)
                        )
                self.logger.debug("Releasing request lock")

    async def handle_idle_timeout(self, is_primary: bool) -> None:
        """
        Handle idle timeout, when no incoming requests received in specified time.
        """
        if self._is_stopped or self._is_disposed:
            return
        if is_primary:
            async with self._primary_idle_lock:
                if self._is_stopped or self._is_disposed:
                    return
                await self._primary_engine_manager.stop_current_engine()
        else:
            async with self._secondary_idle_lock:
                if self._is_stopped or self._is_disposed:
                    return
                await self._secondary_engine_manager.stop_current_engine()

    async def shutdown(self) -> None:
        """
        Shutdown, deinitialize stuff
        """
        if self._request_lock.locked():
            self.logger.warning(
                "shutdown handler waiting for lock (another request in progress)"
            )
        async with self._request_lock:
            if self._is_disposed:
                return
            self._is_stopped = True
            self._is_disposed = True
            async with self._primary_idle_lock:
                self._primary_idle_watchdog.disarm()
            async with self._secondary_idle_lock:
                self._secondary_idle_watchdog.disarm()

    def stop(self) -> None:
        """
        Stop processing any new requests
        """
        self._is_stopped = True
