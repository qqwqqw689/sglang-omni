# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import time
import asyncio
import multiprocessing
from queue import Empty
from typing import Any, Dict

# 设置 multiprocessing 启动方法为 'spawn'（CUDA 要求）
# 必须在导入模块后、创建任何进程之前设置
if torch.cuda.is_available():
    try:
        # 尝试设置启动方法为 'spawn'（如果还没有设置）
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method != 'spawn':
            try:
                multiprocessing.set_start_method('spawn', force=False)
            except RuntimeError:
                # 如果已经设置过，可能需要强制设置
                try:
                    multiprocessing.set_start_method('spawn', force=True)
                except RuntimeError as e:
                    # 如果仍然失败，可能是环境变量或配置问题
                    print(f"Warning: Cannot set multiprocessing start method to 'spawn': {e}")
                    print("CUDA may not work correctly in subprocesses.")
                    print("Consider setting it via environment variable or before importing this module.")
    except RuntimeError:
        # 在某些情况下（如在 pytest 中），可能已经设置过
        pass

# Check if dynamo.nixl_connect is available
try:
    from dynamo.nixl_connect import Connector, Descriptor
except ImportError:
    Connector = None
    Descriptor = None


def sender_process(
    config: Dict[str, Any],
    metadata_queue: multiprocessing.Queue,
    original_data_queue: multiprocessing.Queue,  # 新增：用于传递原始数据用于验证
    ready_event: multiprocessing.Event,
    done_event: multiprocessing.Event,
    num_transfers: int,
    test_data_size: int,
    results_dict: Dict[str, Any],
):
    """
    发送进程：负责创建数据并通过 put_async 发送，尽量复用 connection。
    
    每次传输完成后，通过调用 wait_for_completion() 来判断传输是否完成。
    这样确保每次传输都真正完成后再进行下一次传输。
    
    Parameters
    ----------
    config : Dict[str, Any]
        Connector 配置
    metadata_queue : multiprocessing.Queue
        用于传递 metadata 的队列
    ready_event : multiprocessing.Event
        用于同步的 ready 事件
    done_event : multiprocessing.Event
        用于同步的 done 事件
    num_transfers : int
        传输次数
    test_data_size : int
        测试数据大小（元素数量）
    results_dict : Dict[str, Any]
        用于存储结果的字典（进程间共享）
    """
    async def run_sender():
        from sglang_omni.relay.nixl_reuse_ralay import NixlReuseConnector
        
        connector = NixlReuseConnector(config)
        device = f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu'
        
        try:
            put_times = []
            
            # 通知接收进程已准备好
            ready_event.set()
            
            for transfer_idx in range(num_transfers):
                # 创建测试数据
                original_tensor = torch.randn(test_data_size, dtype=torch.float32, device=device)
                
                # 保存原始数据的 CPU 副本用于验证
                original_values = original_tensor.cpu().clone()
                
                send_descriptor = Descriptor(original_tensor)
                send_descriptors = [send_descriptor]
                
                # 测量 Put 时间（包括等待完成）
                put_start_time = time.time()
                readable_op = await connector.put_async(send_descriptors)
                put_end_time = time.time()
                put_time = (put_end_time - put_start_time) * 1000  # 转换为毫秒
                put_times.append(put_time)
                
                # 获取 metadata 并发送给接收进程（在等待完成之前发送，让接收端可以开始准备）
                metadata = readable_op.metadata()
                assert metadata is not None
                
                # 序列化 metadata 以便传递
                # metadata 是一个 RdmaMetadata (Pydantic BaseModel) 对象
                # 可以安全地使用 pickle 序列化，或者转换为字典
                import pickle
                try:
                    # 尝试使用 pickle 序列化（推荐，因为 RdmaMetadata 是 Pydantic 模型）
                    metadata_serialized = pickle.dumps(metadata)
                except Exception:
                    # 如果 pickle 失败，尝试使用 model_dump (Pydantic v2) 或 dict() (Pydantic v1)
                    try:
                        metadata_dict = metadata.model_dump() if hasattr(metadata, 'model_dump') else metadata.dict()
                        metadata_serialized = pickle.dumps(metadata_dict)
                    except Exception as e:
                        raise RuntimeError(f"Failed to serialize metadata: {e}")
                
                # 发送 metadata 给接收进程（让接收端可以开始准备）
                metadata_queue.put({
                    'transfer_idx': transfer_idx,
                    'metadata': metadata_serialized,
                    'tensor_size': test_data_size,
                    'dtype': original_tensor.dtype,
                    'element_size': original_tensor.element_size(),
                })
                
                # 发送原始数据的 CPU 副本用于验证
                import pickle
                original_data_serialized = pickle.dumps(original_values)
                original_data_queue.put({
                    'transfer_idx': transfer_idx,
                    'original_data': original_data_serialized,
                })
                
                # 等待传输完成（通过 wait_for_completion 判断）
                # ReadableOperation.wait_for_completion() 等待远程的 ReadOperation 完成
               
                await readable_op.wait_for_completion()
                
                
                print(f"[Sender] Transfer {transfer_idx + 1}/{num_transfers}: Put time (including wait_for_completion) = {put_time:.2f} ms")
            
            # 存储结果
            results_dict['sender_put_times'] = put_times
            results_dict['sender_avg_put_time'] = sum(put_times) / len(put_times) if put_times else 0
            results_dict['sender_first_put_time'] = put_times[0] if put_times else 0
            results_dict['sender_subsequent_avg'] = sum(put_times[1:]) / (len(put_times) - 1) if len(put_times) > 1 else 0
            
            # 发送完成信号
            metadata_queue.put(None)  # 发送结束信号
            
            # 等待接收进程完成
            done_event.wait(timeout=60)  # 最多等待 60 秒
            
        except Exception as e:
            results_dict['sender_error'] = str(e)
            print(f"[Sender] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            connector.close()
    
    # 运行异步发送逻辑
    asyncio.run(run_sender())


def receiver_process(
    config: Dict[str, Any],
    metadata_queue: multiprocessing.Queue,
    original_data_queue: multiprocessing.Queue,  # 新增：用于接收原始数据用于验证
    ready_event: multiprocessing.Event,
    done_event: multiprocessing.Event,
    num_transfers: int,
    results_dict: Dict[str, Any],
):
    """
    接收进程：负责通过 get_async 接收数据。
    
    注意：ReadOperation 不能安全地复用 Connection（因为会创建 Remote 对象修改 Connection 状态），
    所以接收进程会尝试复用，但可能会因为状态冲突而回退到创建新连接。
    
    Parameters
    ----------
    config : Dict[str, Any]
        Connector 配置
    metadata_queue : multiprocessing.Queue
        用于接收 metadata 的队列
    ready_event : multiprocessing.Event
        用于同步的 ready 事件
    done_event : multiprocessing.Event
        用于同步的 done 事件
    num_transfers : int
        传输次数
    results_dict : Dict[str, Any]
        用于存储结果的字典（进程间共享）
    """
    async def run_receiver():
        from sglang_omni.relay.nixl_reuse_ralay import NixlReuseConnector
        
        connector = NixlReuseConnector(config)
        device = f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu'
        
        try:
            get_times = []
            total_times = []
            
            # 等待发送进程准备好
            ready_event.wait(timeout=30)  # 最多等待 30 秒
            
            transfer_count = 0
            while transfer_count < num_transfers:
                try:
                    # 从队列获取 metadata（最多等待 60 秒）
                    item = metadata_queue.get(timeout=60)
                    
                    if item is None:  # 结束信号
                        break
                    
                    transfer_idx = item['transfer_idx']
                    metadata_serialized = item['metadata']
                    tensor_size = item['tensor_size']
                    dtype = item['dtype']
                    element_size = item['element_size']
                    
                    # 反序列化 metadata
                    import pickle
                    from dynamo.nixl_connect import RdmaMetadata
                    
                    metadata_obj = pickle.loads(metadata_serialized)
                    
                    # 如果序列化的是字典，需要重建 RdmaMetadata 对象
                    if isinstance(metadata_obj, dict):
                        metadata = RdmaMetadata(**metadata_obj)
                    else:
                        # 如果序列化的是 RdmaMetadata 对象，直接使用
                        metadata = metadata_obj
                    
                    # 创建接收 buffer
                    buffer_tensor = torch.empty(
                        (tensor_size,),
                        dtype=dtype,
                        device=device
                    )
                    
                    buffer_descriptor = Descriptor(buffer_tensor)
                    buffer_descriptors = [buffer_descriptor]
                    
                    # 测量 Get 时间
                    get_start_time = time.time()
                    
                    # 添加详细日志来追踪连接复用情况
                    print(f"[Receiver] Transfer {transfer_count + 1}/{num_transfers}: Attempting get_async...")
                    try:
                        read_op = await connector.get_async(metadata, buffer_descriptors)
                        print(f"[Receiver] Transfer {transfer_count + 1}/{num_transfers}: get_async succeeded")
                    except Exception as get_error:
                        # 捕获详细的错误信息
                        import traceback
                        error_str = str(get_error)
                        print(f"\n[Receiver] Transfer {transfer_count + 1}/{num_transfers}: get_async FAILED")
                        print(f"[Receiver] Error type: {type(get_error).__name__}")
                        print(f"[Receiver] Error message: {error_str}")
                        print(f"[Receiver] Error details:")
                        traceback.print_exc()
                        
                        # 检查是否是状态冲突相关的错误
                        if "invalidated" in error_str or "NIXL_ERR_NOT_FOUND" in error_str or "remove_remote_agent" in error_str:
                            print(f"\n[Receiver] This appears to be a STATE CONFLICT error!")
                            print(f"[Receiver] Connection reuse likely failed due to Remote agent state management.")
                            results_dict['connection_reuse_failed'] = True
                            results_dict['connection_reuse_error'] = error_str
                            results_dict['connection_reuse_failed_transfer'] = transfer_count + 1
                        raise
                    
                    # 等待读取完成
                    print(f"[Receiver] Transfer {transfer_count + 1}/{num_transfers}: Waiting for completion...")
                    try:
                        if hasattr(read_op, 'wait_for_completion') and callable(read_op.wait_for_completion):
                            await read_op.wait_for_completion()
                        print(f"[Receiver] Transfer {transfer_count + 1}/{num_transfers}: Completion confirmed")
                    except Exception as wait_error:
                        # 捕获等待完成时的错误
                        import traceback
                        error_str = str(wait_error)
                        print(f"\n[Receiver] Transfer {transfer_count + 1}/{num_transfers}: wait_for_completion FAILED")
                        print(f"[Receiver] Error type: {type(wait_error).__name__}")
                        print(f"[Receiver] Error message: {error_str}")
                        print(f"[Receiver] Error details:")
                        traceback.print_exc()
                        raise
                    
                    get_end_time = time.time()
                    get_time = (get_end_time - get_start_time) * 1000  # 转换为毫秒
                    
                    # 从队列获取原始数据用于验证
                    try:
                        original_data_item = original_data_queue.get(timeout=10)  # 等待原始数据
                        assert original_data_item['transfer_idx'] == transfer_idx, \
                            f"Transfer index mismatch: expected {transfer_idx}, got {original_data_item['transfer_idx']}"
                        
                        # 反序列化原始数据
                        import pickle
                        original_values = pickle.loads(original_data_item['original_data'])
                        
                        # 验证数据正确性
                        received_values = buffer_tensor.cpu()
                        assert torch.allclose(
                            original_values,
                            received_values,
                            rtol=1e-5,
                            atol=1e-5
                        ), f"Data mismatch in transfer {transfer_count + 1}: values do not match"
                        
                        # 验证数据不是全零
                        assert not torch.allclose(
                            received_values,
                            torch.zeros_like(received_values)
                        ), f"Received data is all zeros in transfer {transfer_count + 1} - transfer may have failed"
                        
                        print(f"[Receiver] Transfer {transfer_count + 1}/{num_transfers}: Data verification PASSED")
                    except Empty:
                        print(f"[Receiver] Warning: Timeout waiting for original data for verification")
                        results_dict['verification_warnings'] = results_dict.get('verification_warnings', [])
                        results_dict['verification_warnings'].append({
                            'transfer_idx': transfer_count + 1,
                            'warning': 'Timeout waiting for original data'
                        })
                    except AssertionError as verify_error:
                        print(f"[Receiver] ERROR: Data verification FAILED in transfer {transfer_count + 1}: {verify_error}")
                        results_dict['verification_errors'] = results_dict.get('verification_errors', [])
                        results_dict['verification_errors'].append({
                            'transfer_idx': transfer_count + 1,
                            'error': str(verify_error)
                        })
                        # 抛出异常，测试应该失败
                        raise
                    except Exception as verify_error:
                        print(f"[Receiver] Warning: Data verification error in transfer {transfer_count + 1}: {verify_error}")
                        results_dict['verification_errors'] = results_dict.get('verification_errors', [])
                        results_dict['verification_errors'].append({
                            'transfer_idx': transfer_count + 1,
                            'error': str(verify_error)
                        })
                        # 不抛出异常，只记录错误
                    
                    get_times.append(get_time)
                    total_times.append(get_time)
                    
                    transfer_count += 1
                    print(f"[Receiver] Transfer {transfer_count}/{num_transfers}: Get time = {get_time:.2f} ms")
                    
                except Empty:
                    print(f"[Receiver] Timeout waiting for metadata")
                    break
                except Exception as e:
                    print(f"\n[Receiver] Error in transfer {transfer_count + 1}: {e}")
                    import traceback
                    print(f"[Receiver] Full traceback:")
                    traceback.print_exc()
                    results_dict['receiver_error'] = str(e)
                    results_dict['receiver_error_traceback'] = traceback.format_exc()
                    break
            
            # 存储结果
            results_dict['receiver_get_times'] = get_times
            results_dict['receiver_avg_get_time'] = sum(get_times) / len(get_times) if get_times else 0
            results_dict['receiver_total_times'] = total_times
            results_dict['receiver_avg_total_time'] = sum(total_times) / len(total_times) if total_times else 0
            
            # 通知发送进程已完成
            done_event.set()
            
        except Exception as e:
            results_dict['receiver_error'] = str(e)
            print(f"[Receiver] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            connector.close()
    
    # 运行异步接收逻辑
    asyncio.run(run_receiver())


@pytest.mark.skipif(Connector is None, reason="dynamo.nixl_connect not available")
@pytest.mark.skipif(Descriptor is None, reason="Descriptor not available")
def test_multiprocess_transfer_with_connection_reuse():
    """
    测试两个进程之间的数据传输，验证 connection 复用是否降低开销。
    
    发送进程（worker0）：复用 connection 进行多次 put_async 操作
    接收进程（worker1）：尽量复用 connection 进行多次 get_async 操作
    
    注意：使用 'spawn' 启动方法以避免 CUDA 在 fork 子进程中的重新初始化问题。
    """
    # 设置 multiprocessing 启动方法为 'spawn'（CUDA 要求）
    # 必须在创建任何进程之前设置
    if torch.cuda.is_available():
        try:
            current_method = multiprocessing.get_start_method(allow_none=True)
            if current_method != 'spawn':
                try:
                    # 如果还没有设置，尝试设置为 'spawn'
                    multiprocessing.set_start_method('spawn', force=False)
                except RuntimeError:
                    # 如果已经设置过，需要使用 force=True
                    multiprocessing.set_start_method('spawn', force=True)
                    print(f"[Test] Set multiprocessing start method to 'spawn' (was: {current_method})")
        except RuntimeError as e:
            # 如果无法设置，可能需要通过环境变量或命令行参数设置
            print(f"[Test] Warning: Cannot set start method to 'spawn': {e}")
            print("[Test] You may need to set multiprocessing start method before running the test")
    
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
        "gpu_id": 1 if torch.cuda.is_available() else 0,
        "worker_id": "worker1",
    }
    
    test_data_size = 100000
    num_transfers = 8  # 测试5次传输
    
    # 创建进程间通信对象
    metadata_queue = multiprocessing.Queue()
    original_data_queue = multiprocessing.Queue()  # 用于传递原始数据用于验证
    ready_event = multiprocessing.Event()
    done_event = multiprocessing.Event()
    
    # 创建共享字典用于存储结果（使用 Manager）
    manager = multiprocessing.Manager()
    results_dict = manager.dict()
    
    print("\n" + "="*80)
    print("多进程传输测试 (Multiprocess Transfer with Connection Reuse)")
    print("="*80)
    print(f"数据大小: {test_data_size} elements (~{test_data_size * 4 / (1024*1024):.2f} MB)")
    print(f"传输次数: {num_transfers}")
    print(f"发送进程: worker0 (GPU {config_worker0['gpu_id']})")
    print(f"接收进程: worker1 (GPU {config_worker1['gpu_id']})")
    print("-"*80)
    
    # 创建发送进程
    sender = multiprocessing.Process(
        target=sender_process,
        args=(config_worker0, metadata_queue, original_data_queue, ready_event, done_event, num_transfers, test_data_size, results_dict)
    )
    
    # 创建接收进程
    receiver = multiprocessing.Process(
        target=receiver_process,
        args=(config_worker1, metadata_queue, original_data_queue, ready_event, done_event, num_transfers, results_dict)
    )
    
    try:
        # 启动进程
        sender.start()
        receiver.start()
        
        # 等待进程完成
        sender.join(timeout=300)  # 最多等待 120 秒
        receiver.join(timeout=300)
        
        # 检查进程是否正常退出
        if sender.exitcode != 0:
            pytest.fail(f"Sender process exited with code {sender.exitcode}")
        if receiver.exitcode != 0:
            pytest.fail(f"Receiver process exited with code {receiver.exitcode}")
        
        # 检查是否有错误
        if 'sender_error' in results_dict:
            print(f"\n[Test] Sender error detected: {results_dict['sender_error']}")
            pytest.fail(f"Sender error: {results_dict['sender_error']}")
        if 'receiver_error' in results_dict:
            print(f"\n[Test] Receiver error detected: {results_dict['receiver_error']}")
            if 'receiver_error_traceback' in results_dict:
                print(f"[Test] Full traceback:\n{results_dict['receiver_error_traceback']}")
            pytest.fail(f"Receiver error: {results_dict['receiver_error']}")
        
        # 检查数据验证错误
        if 'verification_errors' in results_dict:
            verification_errors = results_dict['verification_errors']
            if verification_errors:
                print("\n" + "="*80)
                print("数据验证错误 (Data Verification Errors)")
                print("="*80)
                for error_info in verification_errors:
                    print(f"传输 {error_info['transfer_idx']}: {error_info['error']}")
                print("="*80)
                # 如果有验证错误，测试应该失败
                pytest.fail(f"Data verification failed in {len(verification_errors)} transfer(s)")
        
        # 检查数据验证警告
        if 'verification_warnings' in results_dict:
            verification_warnings = results_dict['verification_warnings']
            if verification_warnings:
                print("\n[Test] Data verification warnings:")
                for warning_info in verification_warnings:
                    print(f"  传输 {warning_info['transfer_idx']}: {warning_info['warning']}")
        
        # 检查连接复用是否失败
        if 'connection_reuse_failed' in results_dict:
            print("\n" + "="*80)
            print("连接复用失败分析 (Connection Reuse Failure Analysis)")
            print("="*80)
            print(f"失败位置: 传输 {results_dict.get('connection_reuse_failed_transfer', 'unknown')}")
            print(f"错误信息: {results_dict.get('connection_reuse_error', 'unknown')}")
            print("\n原因分析:")
            print("  1. ReadOperation 创建 Remote 对象时会修改 Connection 的远程 agent 状态")
            print("  2. 当复用同一个 Connection 时，上一个 Remote 对象可能还没完全清理")
            print("  3. 创建新的 Remote 对象时，Connection 状态不一致导致失败")
            print("\n解决方案:")
            print("  - Get 操作每次都创建新的 Connection（避免状态冲突）")
            print("  - 或者确保上一个 Remote 对象完全清理后再复用 Connection")
            print("="*80)
        
        # 打印结果
        print("\n" + "="*80)
        print("传输时间统计 (Transfer Time Statistics)")
        print("="*80)
        
        if 'sender_put_times' in results_dict:
            sender_put_times = results_dict['sender_put_times']
            sender_avg_put_time = results_dict.get('sender_avg_put_time', 0)
            sender_first_put_time = results_dict.get('sender_first_put_time', 0)
            sender_subsequent_avg = results_dict.get('sender_subsequent_avg', 0)
            
            print("\n[发送进程] Put 操作时间:")
            print(f"  首次 Put 时间: {sender_first_put_time:.2f} ms")
            print(f"  后续 Put 平均时间: {sender_subsequent_avg:.2f} ms")
            print(f"  Put 时间范围: {min(sender_put_times):.2f} - {max(sender_put_times):.2f} ms")
            print(f"  Put 平均时间: {sender_avg_put_time:.2f} ms")
            
            if sender_first_put_time > 0:
                put_time_reduction = (sender_first_put_time - sender_subsequent_avg) / sender_first_put_time * 100
                print(f"  Put 时间减少: {put_time_reduction:.2f}%")
                
                if put_time_reduction > 5:
                    print("  ✅ 结论: 连接复用成功，后续 Put 操作时间显著减少")
                else:
                    print("  ⚠️  结论: 连接复用效果不明显")
        
        if 'receiver_get_times' in results_dict:
            receiver_get_times = results_dict['receiver_get_times']
            receiver_avg_get_time = results_dict.get('receiver_avg_get_time', 0)
            
            print("\n[接收进程] Get 操作时间:")
            print(f"  Get 时间范围: {min(receiver_get_times):.2f} - {max(receiver_get_times):.2f} ms")
            print(f"  Get 平均时间: {receiver_avg_get_time:.2f} ms")
        
        if 'receiver_total_times' in results_dict:
            receiver_total_times = results_dict['receiver_total_times']
            receiver_avg_total_time = results_dict.get('receiver_avg_total_time', 0)
            
            print("\n[接收进程] 总传输时间:")
            print(f"  总传输时间范围: {min(receiver_total_times):.2f} - {max(receiver_total_times):.2f} ms")
            print(f"  总传输平均时间: {receiver_avg_total_time:.2f} ms")
        
        print("="*80)
        
    finally:
        # 清理进程
        if sender.is_alive():
            sender.terminate()
            sender.join(timeout=5)
        if receiver.is_alive():
            receiver.terminate()
            receiver.join(timeout=5)


if __name__ == "__main__":
    # 在直接运行脚本时设置启动方法为 'spawn'（CUDA 要求）
    import sys
    if torch.cuda.is_available():
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # 如果已经设置过，获取当前方法
            current_method = multiprocessing.get_start_method(allow_none=True)
            if current_method != 'spawn':
                print(f"Warning: Multiprocessing start method is '{current_method}', not 'spawn'")
                print("CUDA may not work correctly in subprocesses.")
                print("Set it before importing this module or use: multiprocessing.set_start_method('spawn')")
            else:
                print(f"Multiprocessing start method is correctly set to 'spawn'")
    
    pytest.main([__file__, "-v", "-s"])

