"""Distributed communication metrics collection"""

import time
import functools
import torch

def patch_torch_distributed(logger):
    """
    Patch torch.distributed functions to log communication metrics
    
    Args:
        logger: DistributedMetricsLogger instance
    """
    if not hasattr(torch, 'distributed') or not torch.distributed.is_available():
        return
    
    # Store original functions
    original_all_reduce = torch.distributed.all_reduce
    original_all_gather = torch.distributed.all_gather
    original_broadcast = torch.distributed.broadcast
    original_reduce = torch.distributed.reduce
    
    # Wrap all_reduce to log metrics
    @functools.wraps(original_all_reduce)
    def instrumented_all_reduce(tensor, *args, **kwargs):
        # Calculate tensor size
        tensor_size = tensor.element_size() * tensor.numel()
        
        # Time the operation
        start_time = time.time()
        result = original_all_reduce(tensor, *args, **kwargs)
        end_time = time.time()
        
        # Calculate duration in milliseconds
        duration_ms = (end_time - start_time) * 1000
        
        # Log communication metrics
        logger.log_communication_metrics({
            "operation": "all_reduce",
            "tensor_size_bytes": tensor_size,
            "duration_ms": duration_ms,
            "bandwidth_gbps": (tensor_size * 8) / (duration_ms * 1000000) if duration_ms > 0 else 0
        })
        
        return result
    
    # Wrap all_gather to log metrics
    @functools.wraps(original_all_gather)
    def instrumented_all_gather(output_tensors, input_tensor, *args, **kwargs):
        # Calculate tensor size
        tensor_size = input_tensor.element_size() * input_tensor.numel()
        
        # Time the operation
        start_time = time.time()
        result = original_all_gather(output_tensors, input_tensor, *args, **kwargs)
        end_time = time.time()
        
        # Calculate duration in milliseconds
        duration_ms = (end_time - start_time) * 1000
        
        # Log communication metrics
        logger.log_communication_metrics({
            "operation": "all_gather",
            "tensor_size_bytes": tensor_size,
            "duration_ms": duration_ms,
            "bandwidth_gbps": (tensor_size * 8) / (duration_ms * 1000000) if duration_ms > 0 else 0
        })
        
        return result
    
    # Wrap broadcast to log metrics
    @functools.wraps(original_broadcast)
    def instrumented_broadcast(tensor, *args, **kwargs):
        # Calculate tensor size
        tensor_size = tensor.element_size() * tensor.numel()
        
        # Time the operation
        start_time = time.time()
        result = original_broadcast(tensor, *args, **kwargs)
        end_time = time.time()
        
        # Calculate duration in milliseconds
        duration_ms = (end_time - start_time) * 1000
        
        # Log communication metrics
        logger.log_communication_metrics({
            "operation": "broadcast",
            "tensor_size_bytes": tensor_size,
            "duration_ms": duration_ms,
            "bandwidth_gbps": (tensor_size * 8) / (duration_ms * 1000000) if duration_ms > 0 else 0
        })
        
        return result
    
    # Wrap reduce to log metrics
    @functools.wraps(original_reduce)
    def instrumented_reduce(tensor, *args, **kwargs):
        # Calculate tensor size
        tensor_size = tensor.element_size() * tensor.numel()
        
        # Time the operation
        start_time = time.time()
        result = original_reduce(tensor, *args, **kwargs)
        end_time = time.time()
        
        # Calculate duration in milliseconds
        duration_ms = (end_time - start_time) * 1000
        
        # Log communication metrics
        logger.log_communication_metrics({
            "operation": "reduce",
            "tensor_size_bytes": tensor_size,
            "duration_ms": duration_ms,
            "bandwidth_gbps": (tensor_size * 8) / (duration_ms * 1000000) if duration_ms > 0 else 0
        })
        
        return result
    
    # Apply patches
    torch.distributed.all_reduce = instrumented_all_reduce
    torch.distributed.all_gather = instrumented_all_gather
    torch.distributed.broadcast = instrumented_broadcast
    torch.distributed.reduce = instrumented_reduce
    
    # Return a function to restore original implementations
    def restore_torch_distributed():
        torch.distributed.all_reduce = original_all_reduce
        torch.distributed.all_gather = original_all_gather
        torch.distributed.broadcast = original_broadcast
        torch.distributed.reduce = original_reduce
    
    return restore_torch_distributed

class CommunicationMetrics:
    """Tracks and aggregates communication metrics across nodes"""
    
    def __init__(self, logger, world_size):
        """
        Initialize communication metrics tracker
        
        Args:
            logger: DistributedMetricsLogger instance
            world_size: Number of processes
        """
        self.logger = logger
        self.world_size = world_size
        self.metrics = {
            "operations": 0,
            "bytes_transferred": 0,
            "cumulative_duration_ms": 0,
            "op_count_by_type": {
                "all_reduce": 0,
                "all_gather": 0,
                "broadcast": 0,
                "reduce": 0
            },
            "bytes_by_type": {
                "all_reduce": 0,
                "all_gather": 0,
                "broadcast": 0,
                "reduce": 0
            },
            "duration_by_type": {
                "all_reduce": 0,
                "all_gather": 0,
                "broadcast": 0,
                "reduce": 0
            }
        }
    
    def update(self, operation, tensor_size_bytes, duration_ms):
        """
        Update communication metrics
        
        Args:
            operation: Type of operation ('all_reduce', 'all_gather', etc.)
            tensor_size_bytes: Size of transferred tensor in bytes
            duration_ms: Duration of operation in milliseconds
        """
        self.metrics["operations"] += 1
        self.metrics["bytes_transferred"] += tensor_size_bytes
        self.metrics["cumulative_duration_ms"] += duration_ms
        
        # Update operation-specific metrics
        if operation in self.metrics["op_count_by_type"]:
            self.metrics["op_count_by_type"][operation] += 1
            self.metrics["bytes_by_type"][operation] += tensor_size_bytes
            self.metrics["duration_by_type"][operation] += duration_ms
    
    def get_summary(self):
        """Get summary of communication metrics"""
        total_duration_s = self.metrics["cumulative_duration_ms"] / 1000
        total_bytes = self.metrics["bytes_transferred"]
        
        # Calculate average bandwidth in GB/s
        avg_bandwidth_gbps = 0
        if total_duration_s > 0:
            avg_bandwidth_gbps = (total_bytes * 8) / (total_duration_s * 1000000000)
        
        return {
            "total_operations": self.metrics["operations"],
            "total_bytes_transferred_mb": total_bytes / (1024**2),
            "total_duration_s": total_duration_s,
            "avg_bandwidth_gbps": avg_bandwidth_gbps,
            "operation_counts": self.metrics["op_count_by_type"],
            "bytes_by_operation_mb": {k: v / (1024**2) for k, v in self.metrics["bytes_by_type"].items()},
            "duration_by_operation_s": {k: v / 1000 for k, v in self.metrics["duration_by_type"].items()}
        }