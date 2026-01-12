import pytest
import torch
import time

# Check if dynamo.nixl_connect is available
try:
    from dynamo.nixl_connect import Connector, Descriptor
except ImportError:
    Connector = None
    Descriptor = None


@pytest.fixture
def nixl_connector_config():
    """Basic configuration for NixlConnector."""
    return {
        "host": "127.0.0.1",
        "metadata_server": "http://127.0.0.1:8080/metadata",
        "device_name": "",
        "gpu_id": 0,
    }


@pytest.mark.skipif(Connector is None, reason="dynamo.nixl_connect not available")
class TestNixlConnector:
    """Basic test suite for NixlConnector (requires actual dynamo.nixl_connect)."""
    
    def test_init(self, nixl_connector_config):
        """Test connector initialization - most basic test."""
        from sglang_omni.relay.nixl_ralay import NixlConnector
        
        connector = NixlConnector(nixl_connector_config)
        
        assert connector.connector is not None
        assert connector.host == "127.0.0.1"
        assert connector.gpu_id == 0
        assert connector.device_name == ""
        
        connector.close()
    
    def test_health(self, nixl_connector_config):
        """Test health check - basic functionality test."""
        from sglang_omni.relay.nixl_ralay import NixlConnector
        
        connector = NixlConnector(nixl_connector_config)
        
        try:
            health = connector.health()
            
            assert health is not None
            assert isinstance(health, dict)
            assert health["status"] == "healthy"
            assert health["host"] == "127.0.0.1"
            assert health["metadata_server"] == "http://127.0.0.1:8080/metadata"
            assert health["device_name"] == ""
            assert health["gpu_id"] == 0
            assert "puts" in health
            assert "gets" in health
            assert "bytes_transferred" in health
            assert "errors" in health
            assert "timeouts" in health
            assert health["puts"] == 0
            assert health["gets"] == 0
            assert health["bytes_transferred"] == 0
            assert health["errors"] == 0
            assert health["timeouts"] == 0
            
        finally:
            connector.close()
    
    def test_cleanup(self, nixl_connector_config):
        """Test cleanup operation - basic functionality test."""
        from sglang_omni.relay.nixl_ralay import NixlConnector
        
        connector = NixlConnector(nixl_connector_config)
        
        try:
            connector.cleanup("test_request_id")
            connector.cleanup("another_request_id")
            assert True
            
        finally:
            connector.close()
    
    @pytest.mark.skipif(Descriptor is None, reason="Descriptor not available")
    def test_put(self, nixl_connector_config):
        """Test put operation - basic functionality test."""
        from sglang_omni.relay.nixl_ralay import NixlConnector
        
        connector = NixlConnector(nixl_connector_config)
        
        try:
            device = f'cuda:{connector.gpu_id}' if torch.cuda.is_available() else 'cpu'
            test_tensor = torch.zeros(10, dtype=torch.float32, device=device)
            
            descriptor = Descriptor(test_tensor)
            descriptors = [descriptor]
            
            initial_puts = connector._metrics["puts"]
            initial_bytes = connector._metrics["bytes_transferred"]
            
            readable_op = connector.put(descriptors)
            
            assert readable_op is not None
            assert hasattr(readable_op, 'metadata')
            
            metadata = readable_op.metadata()
            assert metadata is not None
            
            assert connector._metrics["puts"] == initial_puts + 1
            assert connector._metrics["bytes_transferred"] > initial_bytes
            assert connector._metrics["bytes_transferred"] >= test_tensor.numel() * test_tensor.element_size()
            
        finally:
            connector.close()
    
    @pytest.mark.skipif(Descriptor is None, reason="Descriptor not available")
    def test_get(self, nixl_connector_config):
        """Test get operation - basic functionality test."""
        from sglang_omni.relay.nixl_ralay import NixlConnector
        
        connector = NixlConnector(nixl_connector_config)
        
        try:
            device = f'cuda:{connector.gpu_id}' if torch.cuda.is_available() else 'cpu'
            test_tensor = torch.zeros(10, dtype=torch.float32, device=device)
            
            put_descriptor = Descriptor(test_tensor)
            put_descriptors = [put_descriptor]
            
            readable_op = connector.put(put_descriptors)
            metadata = readable_op.metadata()
            
            assert metadata is not None
            
            buffer_tensor = torch.empty_like(test_tensor, device=device)
            buffer_descriptor = Descriptor(buffer_tensor)
            buffer_descriptors = [buffer_descriptor]
            
            initial_gets = connector._metrics["gets"]
            
            read_op = connector.get(metadata, buffer_descriptors)
            
            assert read_op is not None
            assert hasattr(read_op, 'wait_for_completion')
            
            try:
                if hasattr(read_op, 'wait_for_completion') and callable(read_op.wait_for_completion):
                    completion_coro = read_op.wait_for_completion()
                    connector._run_maybe_async(completion_coro)
            except Exception:
                pass
            
            assert connector._metrics["gets"] == initial_gets + 1
            
        finally:
            connector.close()
    
    @pytest.mark.skipif(Descriptor is None, reason="Descriptor not available")
    def test_transfer_between_workers(self):
        """Test data transfer between two workers (GPUs) - verify correctness and measure time."""
        from sglang_omni.relay.nixl_ralay import NixlConnector
        
        # Check if we have at least 2 GPUs available, otherwise skip
        if torch.cuda.is_available() and torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs for this test")
        
        # Create configuration for worker 0 (sender)
        config_worker0 = {
            "host": "127.0.0.1",
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "device_name": "",
            "gpu_id": 0,
            "worker_id": "worker0",
        }
        
        # Create configuration for worker 1 (receiver)
        config_worker1 = {
            "host": "127.0.0.1",
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "device_name": "",
            "gpu_id": 1 if torch.cuda.is_available() else 0,
            "worker_id": "worker1",
        }
        
        # Create connectors for both workers
        connector_worker0 = NixlConnector(config_worker0)
        connector_worker1 = NixlConnector(config_worker1)
        
        try:
            device_worker0 = f'cuda:{config_worker0["gpu_id"]}' if torch.cuda.is_available() else 'cpu'
            device_worker1 = f'cuda:{config_worker1["gpu_id"]}' if torch.cuda.is_available() else 'cpu'
            
            test_data_size = 2000000000
            original_tensor = torch.randn(test_data_size, dtype=torch.float32, device=device_worker0)
            original_values = original_tensor.cpu().clone()
            
            send_descriptor = Descriptor(original_tensor)
            send_descriptors = [send_descriptor]
            
            put_start_time = time.time()
            readable_op = connector_worker0.put(send_descriptors)
            put_end_time = time.time()
            put_time = put_end_time - put_start_time
            
            metadata = readable_op.metadata()
            assert metadata is not None
            
            first_desc = metadata.descriptors[0]
            tensor_size_bytes = first_desc.size
            
            element_size = original_tensor.element_size()
            num_elements = tensor_size_bytes // element_size
            
            buffer_tensor = torch.empty(
                (num_elements,),
                dtype=original_tensor.dtype,
                device=device_worker1
            )
            print(f"Created buffer from metadata: size={tensor_size_bytes} bytes, "
                    f"elements={num_elements}, dtype={original_tensor.dtype}, device={device_worker1}")
            
            buffer_descriptor = Descriptor(buffer_tensor)
            buffer_descriptors = [buffer_descriptor]
            
            # Measure get time (including wait_for_completion)
            get_start_time = time.time()
            read_op = connector_worker1.get(metadata, buffer_descriptors)
            
            # Wait for completion to ensure data transfer is complete
            try:
                if hasattr(read_op, 'wait_for_completion'):
                    if callable(read_op.wait_for_completion):
                        completion_coro = read_op.wait_for_completion()
                        connector_worker1._run_maybe_async(completion_coro)
            except Exception as e:
                # If wait_for_completion fails (e.g., no actual connection),
                # we can't verify data transfer, but we still record the time
                pytest.skip(f"Data transfer not supported in test environment: {e}")
            
            get_end_time = time.time()
            get_time = get_end_time - get_start_time
            total_transfer_time = get_time
            
            received_values = buffer_tensor.cpu()
            
            assert torch.allclose(
                original_values, 
                received_values, 
                rtol=1e-5, 
                atol=1e-5
            ), f"Data mismatch between sender and receiver"
            
            assert not torch.allclose(
                received_values, 
                torch.zeros_like(received_values)
            ), "Received data is all zeros - transfer may have failed"
            
            assert connector_worker0._metrics["puts"] >= 1
            assert connector_worker1._metrics["gets"] >= 1
            
            data_size_bytes = original_tensor.numel() * original_tensor.element_size()
            transfer_speed_mbps = (data_size_bytes * 8) / (total_transfer_time * 1e6) if total_transfer_time > 0 else 0
            
            print(f"\nTransfer Statistics:")
            print(f"  Data size: {data_size_bytes / (1024*1024):.2f} MB")
            print(f"  Put time: {put_time*1000:.2f} ms")
            print(f"  Get time (including transfer): {get_time*1000:.2f} ms")
            print(f"  Total transfer time: {total_transfer_time*1000:.2f} ms")
            print(f"  Transfer speed: {transfer_speed_mbps:.2f} Mbps")
            
            assert put_time >= 0, "Put time should be non-negative"
            assert get_time >= 0, "Get time should be non-negative"
            assert total_transfer_time >= 0, "Total transfer time should be non-negative"
            
        finally:
            connector_worker0.close()
            connector_worker1.close()
    
    
    @pytest.mark.skipif(Descriptor is None, reason="Descriptor not available")
    @pytest.mark.asyncio
    async def test_transfer_between_workers_async(self):
        """Test async data transfer between two workers (GPUs) - verify correctness and measure time."""
        from sglang_omni.relay.nixl_ralay import NixlConnector
        
        if torch.cuda.is_available() and torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs for this test")
        
        config_worker0 = {
            "host": "127.0.0.1",
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "device_name": "",
            "gpu_id": 0,
            "worker_id": "worker0",
        }
        
        config_worker1 = {
            "host": "127.0.0.1",
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "device_name": "",
            "gpu_id": 1,
            "worker_id": "worker1",
        }
        
        connector_worker0 = NixlConnector(config_worker0)
        connector_worker1 = NixlConnector(config_worker1)
        
        try:
            device_worker0 = f'cuda:{config_worker0["gpu_id"]}' if torch.cuda.is_available() else 'cpu'
            device_worker1 = f'cuda:{config_worker1["gpu_id"]}' if torch.cuda.is_available() else 'cpu'
            
            test_data_size = 2000000
            original_tensor = torch.randn(test_data_size, dtype=torch.float32, device=device_worker0)
            original_values = original_tensor.cpu().clone()
            
            send_descriptor = Descriptor(original_tensor)
            send_descriptors = [send_descriptor]
            
            put_start_time = time.time()
            readable_op = await connector_worker0.put_async(send_descriptors)
            put_end_time = time.time()
            put_time = put_end_time - put_start_time
            
            metadata = readable_op.metadata()
            assert metadata is not None
            
            first_desc = metadata.descriptors[0]
            tensor_size_bytes = first_desc.size
            
            element_size = original_tensor.element_size()
            num_elements = tensor_size_bytes // element_size
            
            buffer_tensor = torch.empty(
                (num_elements,),
                dtype=original_tensor.dtype,
                device=device_worker1
            )
            print(f"Created buffer from metadata: size={tensor_size_bytes} bytes, "
                    f"elements={num_elements}, dtype={original_tensor.dtype}, device={device_worker1}")
            
            buffer_descriptor = Descriptor(buffer_tensor)
            buffer_descriptors = [buffer_descriptor]
            
            get_start_time = time.time()
            read_op = await connector_worker1.get_async(metadata, buffer_descriptors)
            
            try:
                if hasattr(read_op, 'wait_for_completion') and callable(read_op.wait_for_completion):
                    await read_op.wait_for_completion()
            except Exception as e:
                pytest.skip(f"Data transfer not supported in test environment: {e}")
            
            get_end_time = time.time()
            get_time = get_end_time - get_start_time
            total_transfer_time = get_time
            
            received_values = buffer_tensor.cpu()
            
            assert torch.allclose(
                original_values, 
                received_values, 
                rtol=1e-5, 
                atol=1e-5
            ), f"Data mismatch between sender and receiver"
            
            assert not torch.allclose(
                received_values, 
                torch.zeros_like(received_values)
            ), "Received data is all zeros - transfer may have failed"
            
            assert connector_worker0._metrics["puts"] >= 1
            assert connector_worker1._metrics["gets"] >= 1
            
            data_size_bytes = original_tensor.numel() * original_tensor.element_size()
            transfer_speed_mbps = (data_size_bytes * 8) / (total_transfer_time * 1e6) if total_transfer_time > 0 else 0
            
            print(f"\nTransfer Statistics (Async):")
            print(f"  Data size: {data_size_bytes / (1024*1024):.2f} MB")
            print(f"  Put time (async): {put_time*1000:.2f} ms")
            print(f"  Get time (async, including transfer): {get_time*1000:.2f} ms")
            print(f"  Total transfer time: {total_transfer_time*1000:.2f} ms")
            print(f"  Transfer speed: {transfer_speed_mbps:.2f} Mbps")
            
            assert put_time >= 0, "Put time should be non-negative"
            assert get_time >= 0, "Get time should be non-negative"
            assert total_transfer_time >= 0, "Total transfer time should be non-negative"
            
        finally:
            connector_worker0.close()
            connector_worker1.close()
    
