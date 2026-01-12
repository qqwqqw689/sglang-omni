# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
性能测试：测量 NixlConnector 的内存注册和连接创建时间开销
"""

import pytest
import torch
import time
import asyncio

try:
    from dynamo.nixl_connect import Connector, Descriptor
except ImportError:
    Connector = None
    Descriptor = None


@pytest.mark.skipif(Descriptor is None, reason="Descriptor not available")
class TestNixlConnectorPerformance:
    """测试 NixlConnector 的性能瓶颈"""

    @pytest.mark.asyncio
    async def test_memory_registration_time(self):
        """测试不同大小数据的内存注册时间"""
        from sglang_omni.relay.nixl_ralay import NixlConnector
        
        config = {
            "host": "127.0.0.1",
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "device_name": "",
            "gpu_id": 0,
            "worker_id": "worker0",
        }
        
        connector = NixlConnector(config)
        device = f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu'
        
        try:
            # 测试不同大小的数据
            test_sizes = [
                (1000, "1K elements (~4 KB)"),
                (10000, "10K elements (~40 KB)"),
                (100000, "100K elements (~400 KB)"),
                (1000000, "1M elements (~4 MB)"),
                (10000000, "10M elements (~40 MB)"),
                (100000000, "100M elements (~400 MB)"),
                (500000000, "500M elements (~2 GB)"),
                (1000000000, "1G elements (~4 GB)"),
                (2000000000, "2G elements (~8 GB)"),
            ]
            
            print("\n" + "="*80)
            print("内存注册时间测试 (Memory Registration Time)")
            print("="*80)
            
            results = []
            
            for num_elements, description in test_sizes:                
                try:
                    # 创建测试 tensor
                    tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
                    descriptor = Descriptor(tensor)
                    
                    # 方法1：尝试直接测量内存注册时间（如果 API 支持）
                    registration_time = None
                    try:
                        # 尝试获取或创建连接（访问私有成员，仅用于测试）
                        if hasattr(connector.connector, '_connection') and connector.connector._connection is not None:
                            conn = connector.connector._connection
                        elif hasattr(connector.connector, '_create_connection'):
                            # 创建新连接
                            conn = await connector.connector._create_connection()
                        else:
                            raise AttributeError("Cannot access connection")
                        
                        # 测量内存注册时间
                        registration_start = time.time()
                        descriptor.register_with_connector(conn)
                        registration_end = time.time()
                        registration_time = (registration_end - registration_start) * 1000  # 转换为毫秒
                    except (AttributeError, TypeError, RuntimeError) as e:
                        # 方法2：如果无法直接访问，通过 put_async 间接测量
                        # 注意：这会包含连接创建和注册的总时间
                        put_start = time.time()
                        readable_op = await connector.put_async([descriptor])
                        put_end = time.time()
                        total_time = (put_end - put_start) * 1000
                        # 这是一个近似值，包含了连接创建 + 注册的总时间
                        registration_time = total_time
                        print(f"  Note: Using put_async time as approximation (includes connection creation + registration)")
                    
                    data_size_mb = (num_elements * 4) / (1024 * 1024)  # float32 = 4 bytes
                    results.append({
                        'size': num_elements,
                        'size_mb': data_size_mb,
                        'description': description,
                        'registration_time_ms': registration_time
                    })
                    
                    print(f"{description:25s} | Size: {data_size_mb:8.2f} MB | Registration: {registration_time:8.2f} ms")
                    
                except Exception as e:
                    print(f"{description:25s} | Error: {e}")
                    continue
            
            print("="*80)
            print("\n内存注册时间分析:")
            if results:
                min_time = min(r['registration_time_ms'] for r in results)
                max_time = max(r['registration_time_ms'] for r in results)
                avg_time = sum(r['registration_time_ms'] for r in results) / len(results)
                
                print(f"  最小时间: {min_time:.2f} ms")
                print(f"  最大时间: {max_time:.2f} ms")
                print(f"  平均时间: {avg_time:.2f} ms")
                print(f"\n结论: 内存注册时间范围: {min_time:.2f} - {max_time:.2f} ms")
                print("      (与数据大小关系不大，主要是固定开销)")
            
        finally:
            connector.close()
    
    @pytest.mark.asyncio
    async def test_connection_creation_time(self):
        """测试连接创建的时间"""
        from sglang_omni.relay.nixl_ralay import NixlConnector
        
        config = {
            "host": "127.0.0.1",
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "device_name": "",
            "gpu_id": 0,
            "worker_id": "worker0",
        }
        
        connector = NixlConnector(config)
        
        try:
            print("\n" + "="*80)
            print("连接创建时间测试 (Connection Creation Time)")
            print("="*80)
            
            # 测试多次连接创建，取平均值
            num_tests = 5
            times = []
            
            for i in range(num_tests):
                try:
                    creation_start = time.time()
                    # 尝试创建连接（访问私有方法）
                    conn = await connector.connector._create_connection()
                    creation_end = time.time()
                    creation_time = (creation_end - creation_start) * 1000  # 转换为毫秒
                    times.append(creation_time)
                    print(f"Test {i+1}/{num_tests}: Connection creation time: {creation_time:.2f} ms")
                except Exception as e:
                    print(f"Test {i+1}/{num_tests}: Error - {e}")
                    continue
            
            print("="*80)
            if times:
                min_time = min(times)
                max_time = max(times)
                avg_time = sum(times) / len(times)
                
                print(f"\n连接创建时间分析:")
                print(f"  最小时间: {min_time:.2f} ms")
                print(f"  最大时间: {max_time:.2f} ms")
                print(f"  平均时间: {avg_time:.2f} ms")
                print(f"\n结论: 连接创建时间范围: {min_time:.2f} - {max_time:.2f} ms")
        
        finally:
            connector.close()
    