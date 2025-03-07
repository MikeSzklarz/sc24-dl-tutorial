"""Training metrics collection for distributed training"""

import time
import numpy as np
import torch

class TrainingMetricsHook:
    """Hook for collecting training metrics during training process"""
    
    def __init__(self, logger, params, log_frequency=1):
        """
        Initialize training metrics hook
        
        Args:
            logger: DistributedMetricsLogger instance
            params: Training parameters
            log_frequency: Log every N iterations
        """
        self.logger = logger
        self.params = params
        self.log_frequency = log_frequency
        
        # Timers for different training phases
        self.timers = {
            "epoch_start": 0,
            "iter_start": 0,
            "data_start": 0,
            "forward_start": 0,
            "backward_start": 0,
            "optimizer_start": 0,
            "comm_start": 0
        }
        
        # Accumulated metrics
        self.metrics = {
            "epoch": 0,
            "iteration": 0,
            "data_time": [],
            "forward_time": [],
            "backward_time": [],
            "optimizer_time": [],
            "communication_time": [],
            "total_time": []
        }
        
        self.last_log_iter = -1
    
    def start_epoch(self, epoch):
        """Mark the start of an epoch"""
        self.timers["epoch_start"] = time.time()
        self.metrics["epoch"] = epoch
    
    def end_epoch(self, train_loss, val_loss=None, val_rmse=None):
        """Mark the end of an epoch and log summary"""
        epoch_time = time.time() - self.timers["epoch_start"]
        
        # Calculate samples per second
        steps_in_epoch = len(self.metrics["total_time"]) - self.last_log_iter
        if steps_in_epoch > 0:
            avg_iter_time = sum(self.metrics["total_time"][self.last_log_iter:]) / steps_in_epoch
            throughput = self.params.global_batch_size / avg_iter_time if avg_iter_time > 0 else 0
        else:
            throughput = 0
        
        # Log epoch summary
        self.logger.log_epoch_summary(
            epoch=self.metrics["epoch"],
            train_loss=train_loss,
            val_loss=val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss,
            val_rmse=val_rmse.cpu().numpy()[0] if isinstance(val_rmse, torch.Tensor) else val_rmse,
            epoch_time=epoch_time,
            throughput=throughput
        )
        
        # Perform bottleneck analysis
        self._analyze_bottlenecks()
        
        # Reset for next epoch
        self.last_log_iter = len(self.metrics["total_time"]) - 1
    
    def start_iteration(self, iteration):
        """Mark the start of an iteration"""
        self.timers["iter_start"] = time.time()
        self.metrics["iteration"] = iteration
    
    def start_data_loading(self):
        """Mark the start of data loading phase"""
        self.timers["data_start"] = time.time()
    
    def end_data_loading(self):
        """Mark the end of data loading phase"""
        if self.timers["data_start"] > 0:
            data_time = time.time() - self.timers["data_start"]
            self.metrics["data_time"].append(data_time)
    
    def start_forward(self):
        """Mark the start of forward pass"""
        self.timers["forward_start"] = time.time()
    
    def end_forward(self):
        """Mark the end of forward pass"""
        if self.timers["forward_start"] > 0:
            forward_time = time.time() - self.timers["forward_start"]
            self.metrics["forward_time"].append(forward_time)
    
    def start_backward(self):
        """Mark the start of backward pass"""
        self.timers["backward_start"] = time.time()
    
    def end_backward(self):
        """Mark the end of backward pass"""
        if self.timers["backward_start"] > 0:
            backward_time = time.time() - self.timers["backward_start"]
            self.metrics["backward_time"].append(backward_time)
    
    def start_optimizer(self):
        """Mark the start of optimizer step"""
        self.timers["optimizer_start"] = time.time()
    
    def end_optimizer(self):
        """Mark the end of optimizer step"""
        if self.timers["optimizer_start"] > 0:
            optimizer_time = time.time() - self.timers["optimizer_start"]
            self.metrics["optimizer_time"].append(optimizer_time)
    
    def start_communication(self):
        """Mark the start of communication operations"""
        self.timers["comm_start"] = time.time()
    
    def end_communication(self):
        """Mark the end of communication operations"""
        if self.timers["comm_start"] > 0:
            comm_time = time.time() - self.timers["comm_start"]
            self.metrics["communication_time"].append(comm_time)
    
    def end_iteration(self, loss=None):
        """Mark the end of an iteration and log metrics"""
        iter_time = time.time() - self.timers["iter_start"]
        self.metrics["total_time"].append(iter_time)
        
        # Only log every log_frequency iterations
        if self.metrics["iteration"] % self.log_frequency != 0:
            return
        
        # Get last value for each metric safely
        def get_last(metric_list, default=0):
            return metric_list[-1] if metric_list else default
        
        # Calculate time breakdown
        timings = {
            "total": iter_time,
            "data_loading": get_last(self.metrics["data_time"]),
            "forward_pass": get_last(self.metrics["forward_time"]),
            "backward_pass": get_last(self.metrics["backward_time"]),
            "optimizer": get_last(self.metrics["optimizer_time"]),
            "communication": get_last(self.metrics["communication_time"])
        }
        
        # Calculate throughput
        samples_per_sec = self.params.local_batch_size / iter_time if iter_time > 0 else 0
        
        # Create metrics dict
        metrics = {
            "iteration": self.metrics["iteration"],
            "epoch": self.metrics["epoch"],
            "timings": timings,
            "throughput": {
                "samples_per_second": samples_per_sec,
                "ms_per_sample": 1000 * iter_time / self.params.local_batch_size if self.params.local_batch_size > 0 else 0
            }
        }
        
        # Add loss if provided
        if loss is not None:
            metrics["loss"] = loss.item() if isinstance(loss, torch.Tensor) else loss
        
        # Log metrics
        self.logger.log_iteration_metrics(metrics)
    
    def _analyze_bottlenecks(self):
        """Analyze performance bottlenecks based on collected metrics"""
        # Only analyze if we have enough data and we're the main process
        if self.logger.world_rank != 0 or len(self.metrics["total_time"]) < 10:
            return
        
        bottlenecks = []
        
        # Get average times for each phase
        avg_total = np.mean(self.metrics["total_time"])
        
        if self.metrics["data_time"]:
            avg_data = np.mean(self.metrics["data_time"])
            data_pct = avg_data / avg_total if avg_total > 0 else 0
            if data_pct > 0.3:
                bottlenecks.append({
                    "type": "data_loading",
                    "severity": "high" if data_pct > 0.5 else "medium",
                    "percentage": round(data_pct * 100, 2),
                    "recommendation": "Increase num_workers, use DALI, or optimize data pipeline"
                })
        
        if self.metrics["communication_time"]:
            avg_comm = np.mean(self.metrics["communication_time"])
            comm_pct = avg_comm / avg_total if avg_total > 0 else 0
            if comm_pct > 0.3:
                bottlenecks.append({
                    "type": "communication",
                    "severity": "high" if comm_pct > 0.5 else "medium",
                    "percentage": round(comm_pct * 100, 2),
                    "recommendation": "Consider gradient accumulation, mixed precision, or optimized all-reduce"
                })
        
        if bottlenecks:
            self.logger.log_bottleneck_analysis(bottlenecks)