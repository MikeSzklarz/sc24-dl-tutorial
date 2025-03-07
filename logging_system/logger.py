"""Core logger class for distributed training metrics"""

import os
import json
import time
from datetime import datetime
import torch
import numpy as np

from .async_writer import AsyncMetricsWriter

def current_time_iso():
    """Return current time in ISO 8601 format"""
    return datetime.utcnow().isoformat() + 'Z'

class DistributedMetricsLogger:
    """Main logger class for distributed training metrics"""
    
    def __init__(self, log_dir, world_rank, world_size, config):
        """
        Initialize the distributed metrics logger
        
        Args:
            log_dir: Directory for log files
            world_rank: Rank of this process
            world_size: Total number of processes
            config: Logging configuration
        """
        self.log_dir = log_dir
        self.world_rank = world_rank
        self.world_size = world_size
        self.config = config
        
        # Create subdirectories structure
        self.dirs = {
            "system": os.path.join(log_dir, "system_metrics"),
            "training": os.path.join(log_dir, "training_metrics"),
            "model": os.path.join(log_dir, "model_metrics"),
            "communication": os.path.join(log_dir, "communication_metrics"),
            "visualization": os.path.join(log_dir, "visualization"),
            "raw": os.path.join(log_dir, "raw_logs")
        }
        
        # Create directories if master rank
        if world_rank == 0:
            for dir_path in self.dirs.values():
                os.makedirs(dir_path, exist_ok=True)
        
        # Initialize asynchronous writers for different metric types
        buffer_size = config.get("buffer_size", 100)
        flush_interval = config.get("flush_interval", 5)
        
        self.writers = {
            "system": AsyncMetricsWriter(
                os.path.join(self.dirs["system"], f"node_{world_rank}_system.jsonl"),
                buffer_size=buffer_size, flush_interval=flush_interval
            ),
            "gpu": AsyncMetricsWriter(
                os.path.join(self.dirs["system"], f"node_{world_rank}_gpu.jsonl"),
                buffer_size=buffer_size, flush_interval=flush_interval
            ),
            "iteration": AsyncMetricsWriter(
                os.path.join(self.dirs["training"], f"rank_{world_rank}_iterations.jsonl"),
                buffer_size=buffer_size, flush_interval=flush_interval
            ),
            "communication": AsyncMetricsWriter(
                os.path.join(self.dirs["communication"], f"rank_{world_rank}_comm.jsonl"),
                buffer_size=buffer_size, flush_interval=flush_interval
            ),
            "gradient": AsyncMetricsWriter(
                os.path.join(self.dirs["model"], f"rank_{world_rank}_gradients.jsonl"),
                buffer_size=buffer_size, flush_interval=flush_interval
            ),
            "parameter": AsyncMetricsWriter(
                os.path.join(self.dirs["model"], f"rank_{world_rank}_parameters.jsonl"),
                buffer_size=buffer_size, flush_interval=flush_interval
            ),
        }
        
        # Initialize log files with metadata if master rank
        if world_rank == 0:
            self._initialize_log_files()
        
        # Accumulated statistics for summary reports
        self.training_stats = {
            "start_time": time.time(),
            "iterations": [],
            "epoch_times": [],
            "train_losses": [],
            "val_losses": [],
            "val_metrics": [],
            "throughputs": [],
            "comm_overhead": [],
            "data_load_times": []
        }
    
    def _initialize_log_files(self):
        """Initialize log files with metadata header"""
        metadata = {
            "experiment_start_time": current_time_iso(),
            "world_size": self.world_size,
            "gpus_per_node": torch.cuda.device_count(),
            "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown",
            "config": self.config
        }
        
        metadata_file = os.path.join(self.log_dir, "experiment_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def convert_to_python_types(obj):
        """
        Convert numpy and torch values to Python native types to make them JSON serializable
        
        Args:
            obj: The object to convert
            
        Returns:
            The converted object
        """
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.cpu().detach().numpy().tolist()
        elif isinstance(obj, (list, tuple)):
            return [DistributedMetricsLogger.convert_to_python_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: DistributedMetricsLogger.convert_to_python_types(value) for key, value in obj.items()}
        else:
            return obj
    
    # ===== Logging methods for different metric types =====
    
    def log_system_metrics(self, metrics):
        """Log system-level metrics (CPU, memory, etc.)"""
        metrics["timestamp"] = current_time_iso()
        metrics["node_id"] = self.world_rank
        self.writers["system"].write(metrics)
    
    def log_gpu_metrics(self, metrics):
        """Log GPU-specific metrics"""
        metrics["timestamp"] = current_time_iso() 
        metrics["node_id"] = self.world_rank
        self.writers["gpu"].write(metrics)
    
    def log_iteration_metrics(self, metrics):
        """Log per-iteration training metrics"""
        # Update shared statistics
        if "timings" in metrics:
            self.training_stats["iterations"].append({
                "iteration": metrics.get("iteration", 0),
                "epoch": metrics.get("epoch", 0),
                "total_time": metrics["timings"].get("total", 0),
                "data_loading": metrics["timings"].get("data_loading", 0),
                "forward": metrics["timings"].get("forward_pass", 0),
                "backward": metrics["timings"].get("backward_pass", 0),
                "communication": metrics["timings"].get("communication", 0),
                "optimizer": metrics["timings"].get("optimizer", 0)
            })
            
            # Track data loading time for analysis
            self.training_stats["data_load_times"].append(
                metrics["timings"].get("data_loading", 0)
            )
            
            # Track communication overhead
            if metrics["timings"].get("total", 0) > 0:
                comm_pct = metrics["timings"].get("communication", 0) / metrics["timings"].get("total", 1)
                self.training_stats["comm_overhead"].append(comm_pct)
        
        if "loss" in metrics:
            self.training_stats["train_losses"].append(metrics["loss"])
        
        # Add timestamp
        metrics["timestamp"] = current_time_iso()
        metrics["node_id"] = self.world_rank
        
        # Write metrics
        self.writers["iteration"].write(metrics)
    
    def log_communication_metrics(self, metrics):
        """Log communication operations metrics"""
        metrics["timestamp"] = current_time_iso()
        metrics["node_id"] = self.world_rank
        self.writers["communication"].write(metrics)
    
    def log_gradient_metrics(self, metrics):
        """Log gradient statistics"""
        metrics["timestamp"] = current_time_iso()
        metrics["node_id"] = self.world_rank
        self.writers["gradient"].write(metrics)
    
    def log_parameter_metrics(self, metrics):
        """Log model parameter statistics"""
        metrics["timestamp"] = current_time_iso()
        metrics["node_id"] = self.world_rank
        self.writers["parameter"].write(metrics)
    
    def log_epoch_summary(self, epoch, train_loss, val_loss, val_rmse, epoch_time, throughput):
        """Log summary metrics for an epoch"""
        self.training_stats["epoch_times"].append(epoch_time)
        self.training_stats["throughputs"].append(throughput)
        
        if val_loss is not None:
            self.training_stats["val_losses"].append(val_loss)
        
        if val_rmse is not None:
            self.training_stats["val_metrics"].append(val_rmse)
        
        # Write summary to file if master rank
        if self.world_rank == 0:
            summary = {
                "timestamp": current_time_iso(),
                "epoch": epoch,
                "train_loss": self.convert_to_python_types(train_loss),
                "val_loss": self.convert_to_python_types(val_loss),
                "val_rmse": self.convert_to_python_types(val_rmse),
                "epoch_time_sec": self.convert_to_python_types(epoch_time),
                "throughput_samples_per_sec": self.convert_to_python_types(throughput)
            }
            
            summary_file = os.path.join(self.dirs["training"], "epoch_summaries.jsonl")
            with open(summary_file, "a") as f:
                f.write(json.dumps(summary) + "\n")
    
    def log_bottleneck_analysis(self, bottlenecks):
        """Log bottleneck analysis"""
        if self.world_rank == 0:
            bottleneck_file = os.path.join(self.dirs["visualization"], "bottlenecks.jsonl")
            with open(bottleneck_file, "a") as f:
                f.write(json.dumps({
                    "timestamp": current_time_iso(),
                    "bottlenecks": self.convert_to_python_types(bottlenecks)
                }) + "\n")
    
    def generate_summary_report(self):
        """Generate summary report at end of training"""
        try:
            if self.world_rank == 0:
                # Calculate overall statistics
                training_time = time.time() - self.training_stats["start_time"]
                
                # Compute averages from collected data
                iterations = self.training_stats["iterations"]
                if iterations:
                    avg_iteration_time = sum(it["total_time"] for it in iterations) / len(iterations)
                    avg_data_loading = sum(it["data_loading"] for it in iterations) / len(iterations)
                    avg_forward = sum(it["forward"] for it in iterations) / len(iterations)
                    avg_backward = sum(it["backward"] for it in iterations) / len(iterations)
                    avg_communication = sum(it["communication"] for it in iterations) / len(iterations)
                    avg_optimizer = sum(it["optimizer"] for it in iterations) / len(iterations)
                else:
                    avg_iteration_time = avg_data_loading = avg_forward = avg_backward = avg_communication = avg_optimizer = 0
                
                # Compute peak and latest metrics
                throughputs = self.training_stats["throughputs"]
                peak_throughput = max(throughputs) if throughputs else 0
                final_throughput = throughputs[-1] if throughputs else 0
                
                # Calculate average communication overhead percentage
                comm_overhead = self.training_stats["comm_overhead"]
                avg_comm_overhead = sum(comm_overhead) / len(comm_overhead) if comm_overhead else 0
                
                # Final losses and metrics
                train_losses = self.training_stats["train_losses"]
                val_losses = self.training_stats["val_losses"]
                val_metrics = self.training_stats["val_metrics"]
                
                final_train_loss = train_losses[-1] if train_losses else None
                final_val_loss = val_losses[-1] if val_losses else None
                final_val_metric = val_metrics[-1] if val_metrics else None
                
                # Create report with proper type conversion for JSON serialization
                report = {
                    "experiment": {
                        "end_time": current_time_iso(),
                        "duration_hours": training_time / 3600,
                        "world_size": self.world_size,
                        "gpus_per_node": torch.cuda.device_count() if torch.cuda.is_available() else 0
                    },
                    "performance": {
                        "final_throughput_samples_per_sec": self.convert_to_python_types(final_throughput),
                        "peak_throughput_samples_per_sec": self.convert_to_python_types(peak_throughput),
                        "avg_iteration_time_ms": self.convert_to_python_types(avg_iteration_time * 1000),
                        "communication_overhead_pct": self.convert_to_python_types(avg_comm_overhead * 100),
                        "time_breakdown_ms": {
                            "data_loading": self.convert_to_python_types(avg_data_loading * 1000),
                            "forward": self.convert_to_python_types(avg_forward * 1000),
                            "backward": self.convert_to_python_types(avg_backward * 1000),
                            "communication": self.convert_to_python_types(avg_communication * 1000),
                            "optimizer": self.convert_to_python_types(avg_optimizer * 1000)
                        }
                    },
                    "training": {
                        "final_train_loss": self.convert_to_python_types(final_train_loss),
                        "final_val_loss": self.convert_to_python_types(final_val_loss),
                        "final_val_rmse": self.convert_to_python_types(final_val_metric)
                    },
                    "recommendations": self._generate_recommendations()
                }
                
                # Write report
                report_file = os.path.join(self.log_dir, "summary_report.json")
                with open(report_file, "w") as f:
                    json.dump(report, f, indent=2)
                
                return report
        except Exception as e:
            import logging
            logging.error(f"Error generating summary report: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            # Still return some basic information even if there's an error
            return {
                "error": str(e),
                "timestamp": current_time_iso(),
                "note": "Summary report generation failed, partial data may be available"
            }
    
    def _generate_recommendations(self):
        """Generate recommendations based on collected metrics"""
        recommendations = []
        
        # Analyze data loading bottlenecks
        data_load_times = self.training_stats["data_load_times"]
        iterations = self.training_stats["iterations"]
        
        if data_load_times and iterations:
            avg_data_loading = sum(data_load_times) / len(data_load_times)
            avg_iteration_time = sum(it["total_time"] for it in iterations) / len(iterations)
            
            if avg_data_loading > 0.3 * avg_iteration_time:
                recommendations.append({
                    "issue": "data_loading_bottleneck",
                    "description": "Data loading is taking more than 30% of iteration time",
                    "recommendation": "Consider increasing num_workers, using DALI, or optimizing the data pipeline"
                })
        
        # Analyze communication overhead
        comm_overhead = self.training_stats["comm_overhead"]
        if comm_overhead and sum(comm_overhead) / len(comm_overhead) > 0.3:
            recommendations.append({
                "issue": "communication_bottleneck",
                "description": "Communication overhead is more than 30% of iteration time",
                "recommendation": "Consider gradient accumulation, mixed precision, or optimized all-reduce"
            })
        
        return recommendations
    
    def flush_all(self):
        """Flush all writers"""
        for writer in self.writers.values():
            writer.flush()
    
    def close(self):
        """Close all writers"""
        for writer in self.writers.values():
            writer.close()