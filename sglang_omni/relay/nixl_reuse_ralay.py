# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import asyncio
from typing import Any
import logging

logger = logging.getLogger(__name__)


try:
    from dynamo.nixl_connect import Connector, ReadableOperation, ReadOperation
except ImportError:
    logger.warning("dynamo.nixl_connect not available, NixlReuseConnector will not work")
    Connector = None
    ReadableOperation = None
    ReadOperation = None


class NixlReuseConnector:
    """Nixl-based distributed connector with connection reuse support."""

    def __init__(self, config: dict[str, Any]):
        if Connector is None:
            raise ImportError("dynamo.nixl_connect not available")

        self.config = config
        self.host = config.get("host", "127.0.0.1")
        self.metadata_server = config.get("metadata_server", "http://127.0.0.1:8080/metadata")
        self.device_name = config.get("device_name", "")
        self.gpu_id = config.get("gpu_id", 0)
        
        self.connector: Connector | None = None
        self._pending_operations: dict[str, tuple[Any, asyncio.Event]] = {}
        self._shared_connection: Any | None = None  # 共享的 Connection，用于复用
        self._active_read_ops: list[Any] = []  # 保持 ReadOperation 对象引用，避免被垃圾回收
        self._max_active_read_ops: int = 5  # 最多保持的 ReadOperation 数量
        
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "timeouts": 0,
        }

        self._init_sync()
    
    def _run_maybe_async(self, coro_or_result):
        """Run a coroutine if it's a coroutine, otherwise return the result directly."""
        if asyncio.iscoroutine(coro_or_result):
            try:
                asyncio.get_running_loop()
                raise RuntimeError(
                    "Cannot run async operation synchronously while in an async context. "
                    "If the connector uses async methods, you must be in a sync context or use asyncio.run() at a higher level."
                )
            except RuntimeError as e:
                if "no running event loop" in str(e) or "no current event loop" in str(e):
                    try:
                        return asyncio.run(coro_or_result)
                    except RuntimeError as run_error:
                        if "cannot be called from a running event loop" in str(run_error):
                            raise RuntimeError(
                                "Cannot run async operation synchronously while in an async context. "
                                "If the connector uses async methods, you must be in a sync context."
                            ) from run_error
                        raise
                else:
                    raise
            except AttributeError:
                return asyncio.run(coro_or_result)
        else:
            return coro_or_result

    def _init_sync(self):
        """Initialize Nixl connector synchronously."""
        try:
            self.connector = Connector(worker_id=self.config.get("worker_id"))
            logger.info("NixlReuseConnector initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Nixl connector: %s", e)
            raise

    async def _get_or_create_shared_connection(self, try_reuse: bool = True):
        """获取或创建共享的 Connection，用于 Put 操作复用以减少连接创建开销"""
        if try_reuse and self._shared_connection is not None:
            return self._shared_connection
        else:
            self._shared_connection = await self.connector._create_connection()
            return self._shared_connection

    
    async def put_async(self, descriptors: list[Any]) -> Any:
        """
        Put descriptors into the distributed store with connection reuse.
        
        This method reuses a shared connection to avoid connection creation overhead (~1s).
        ReadableOperation only registers local descriptors and does not create Remote objects,
        so it's safe to reuse the same connection for multiple Put operations.
        
        Parameters
        ----------
        descriptors : list[Any]
            List of Descriptor objects containing tensor data
            
        Returns
        -------
        Any
            Readable operation object with metadata() and wait_for_completion() methods
        """
        if not self.connector:
            logger.error("Connector not initialized")
            raise RuntimeError("Connector not initialized")

        try:
            # 复用共享的 Connection，直接创建 ReadableOperation
            # ReadableOperation 只注册本地描述符，不会创建 Remote，可以安全复用
            conn = await self._get_or_create_shared_connection()
            readable_op = ReadableOperation(conn, descriptors)
            logger.debug("NixlReuseConnector: Using shared connection for put")

            total_size = 0
            for desc in descriptors:
                total_size += desc.size

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += total_size

            logger.info(
                "NixlReuseConnector: created readable for %d descriptors, %d bytes",
                len(descriptors),
                total_size,
            )

            return readable_op

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("NixlReuseConnector put failed: %s", e)
            raise
    
    async def get_async(self, metadata: Any, descriptors: list[Any], reuse_connection: bool = True) -> Any:
        """
        Get data from the distributed store with connection reuse attempt.
        
        This method attempts to reuse a shared connection to reduce connection creation overhead (~1s).
        If reuse fails due to state conflicts, it will automatically fall back to creating a new connection.
        
        Note: Get operations create Remote objects that modify connection state. 
        Connection reuse may succeed if the previous Remote has been properly cleaned up.
        
        Parameters
        ----------
        metadata : Any
            Metadata from readable operation (returned by put)
        descriptors : list[Any]
            List of Descriptor objects for receiving data
        reuse_connection : bool, optional
            If True, attempt to reuse shared connection first, fall back to new connection if it fails.
            Default is True (attempt reuse to reduce overhead).
            
        Returns
        -------
        Any
            Read operation object with wait_for_completion() method
        """
        if not self.connector:
            logger.error("Connector not initialized")
            raise RuntimeError("Connector not initialized")

        try:
            if reuse_connection:
                # 尝试复用共享连接，如果失败则创建新连接
                logger.error("NixlReuseConnector: Attempting to reuse connection for get")
                try:
                    conn = await self._get_or_create_shared_connection()
                    logger.error("NixlReuseConnector: Got connection (trying to reuse), creating ReadOperation...")
                    # 尝试创建 ReadOperation - 这里可能抛出异常
                    read_op = ReadOperation(conn, metadata, descriptors)
                    
                    # 保持 ReadOperation 对象的引用，避免被垃圾回收
                    # 这样可以避免 Remote 对象被过早释放，从而避免远程 agent 被移除
                    self._active_read_ops.append(read_op)
                    logger.debug(f"NixlReuseConnector: Kept ReadOperation reference (total: {len(self._active_read_ops)})")
                    
                    # 如果超过了最大数量，清理最旧的操作
                    if len(self._active_read_ops) > self._max_active_read_ops:
                        old_op = self._active_read_ops.pop(0)
                        logger.debug(f"NixlReuseConnector: Releasing old ReadOperation (keeping {len(self._active_read_ops)} active)")
                        # 手动触发清理（通过删除引用来允许垃圾回收）
                        # 注意：这会释放 Remote 对象，移除远程 agent
                        try:
                            del old_op
                        except:
                            pass
                    
                    logger.error("NixlReuseConnector: Successfully reused shared connection for get")
                except Exception as reuse_error:
                    # 复用失败（可能因为状态冲突），创建新连接
                    error_str = str(reuse_error)
                    error_type = type(reuse_error).__name__
                    logger.error(
                        "NixlReuseConnector: Connection reuse failed during ReadOperation creation"
                    )
                    logger.error(f"NixlReuseConnector: Error type: {error_type}")
                    logger.error(f"NixlReuseConnector: Error message: {error_str}")
                    
                    # 检查是否是状态冲突相关的错误
                    if "invalidated" in error_str or "NIXL_ERR_NOT_FOUND" in error_str or "remove_remote_agent" in error_str:
                        logger.warning(
                            "NixlReuseConnector: Connection reuse failed (state conflict detected), creating new connection"
                        )
                        logger.warning(f"NixlReuseConnector: State conflict error: {reuse_error}")
                        # 记录完整的错误信息（包括 traceback）
                        import traceback
                        error_traceback = traceback.format_exc()
                        logger.warning(f"NixlReuseConnector: Full traceback:\n{error_traceback}")
                        
                        # 创建新连接
                        logger.info("NixlReuseConnector: Creating new connection (avoiding state conflict)...")
                        conn = await self._get_or_create_shared_connection(try_reuse=False)
                        read_op = ReadOperation(conn, metadata, descriptors)
                        logger.info("NixlReuseConnector: Successfully created new connection for get")
                    else:
                        # 其他错误，直接抛出
                        logger.error(f"NixlReuseConnector: Unexpected error during connection reuse: {reuse_error}")
                        import traceback
                        logger.error(f"NixlReuseConnector: Full traceback:\n{traceback.format_exc()}")
                        raise
            else:
                # 不复用，直接创建新连接
                conn = await self._get_or_create_shared_connection(try_reuse=False)
                read_op = ReadOperation(conn, metadata, descriptors)
                
                # 不复用时也保持引用（如果后续需要复用连接）
                self._active_read_ops.append(read_op)
                if len(self._active_read_ops) > self._max_active_read_ops:
                    old_op = self._active_read_ops.pop(0)
                    try:
                        del old_op
                    except:
                        pass
                
                logger.debug("NixlReuseConnector: Created new connection for get")

            total_size = 0
            for desc in descriptors:
                total_size += desc.size

            self._metrics["gets"] += 1

            logger.debug(
                "NixlReuseConnector: began read for %d descriptors, %d bytes",
                len(descriptors),
                total_size,
            )

            return read_op
    
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("NixlReuseConnector get failed: %s", e)
            raise

    def cleanup_old_read_ops(self, keep_last_n: int = 0) -> None:
        """
        清理旧的 ReadOperation 对象。
        
        保持 ReadOperation 对象的引用可以避免 Remote 对象被过早释放，
        从而避免远程 agent 被移除。但需要定期清理，避免内存泄漏。
        
        Parameters
        ----------
        keep_last_n : int
            保留最近 N 个 ReadOperation 对象，默认 0（全部清理）
        """
        if keep_last_n > 0:
            old_ops = self._active_read_ops[:-keep_last_n]
            self._active_read_ops = self._active_read_ops[-keep_last_n:]
        else:
            old_ops = self._active_read_ops
            self._active_read_ops = []
        
        # 清理旧操作（删除引用，允许垃圾回收）
        for op in old_ops:
            try:
                del op
            except:
                pass
        
        logger.debug(
            f"NixlReuseConnector: Cleaned up {len(old_ops)} old ReadOperation objects, "
            f"keeping {len(self._active_read_ops)} active"
        )
    
    def cleanup(self, request_id: str) -> None:
        """Clean up resources for a request."""
        if not self.connector:
            return

        # Clean up pending operations
        keys_to_remove = [k for k in self._pending_operations.keys() if request_id in k]
        for key in keys_to_remove:
            del self._pending_operations[key]
        
        logger.debug("NixlReuseConnector: cleanup requested for %s", request_id)

    def health(self) -> dict[str, Any]:
        """Get connector health status."""
        if not self.connector:
            return {"status": "unhealthy", "error": "Connector not initialized"}

        return {
            "status": "healthy",
            "host": self.host,
            "metadata_server": self.metadata_server,
            "device_name": self.device_name,
            "gpu_id": self.gpu_id,
            "connection_reused": self._shared_connection is not None,
            **self._metrics,
        }

    def close(self):
        """Clean shutdown."""
        # 清理所有活动的 ReadOperation 对象
        self.cleanup_old_read_ops(keep_last_n=0)
        
        # 清理共享的 Connection
        self._shared_connection = None
        
        if self.connector:
            try:
                if hasattr(self.connector, 'close'):
                    result = self.connector.close()
                    self._run_maybe_async(result)
                
                self.connector = None
                logger.info("NixlReuseConnector closed")
            except Exception as e:
                logger.error("Error closing Nixl connector: %s", e)

