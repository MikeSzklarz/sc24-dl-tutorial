import os
import time
import csv
import torch
import logging
import numpy as np
import shutil
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

class MetricsLogger:
    """Metrics logger with support for multi-node collection"""
    
    def __init__(self, log_dir, metrics_log_freq, world_size, world_rank):
        """Initialize the metrics logger
        
        Args:
            log_dir: Directory to save CSV files
            metrics_log_freq: How often to log metrics (in iterations)
            world_size: Total number of processes
            world_rank: Current process rank
        """
        self.log_dir = log_dir
        self.metrics_log_freq = metrics_log_freq
        self.world_size = world_size
        self.world_rank = world_rank
        self.is_main_process = (world_rank == 0)
        
        # Initialize NVML for GPU metrics on all nodes
        self.nvml_initialized = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            except Exception as e:
                logging.warning(f"Rank {world_rank}: Failed to initialize NVML: {e}")
        
        # Only main process creates directories and files
        if self.is_main_process:
            # Create log directory
            os.makedirs(log_dir, exist_ok=True)
            
            # Initialize CSV files
            self.perf_file = os.path.join(log_dir, "performance_metrics.csv")
            self.gpu_file = os.path.join(log_dir, "gpu_metrics.csv")
            
            # if the files exist, remove them
            if os.path.exists(self.perf_file):
                logging.info(f"Performance metrics exist... removing {self.perf_file}")
                os.remove(self.perf_file)
            if os.path.exists(self.gpu_file):
                logging.info(f"GPU metrics file exists... removing {self.gpu_file}")
                os.remove(self.gpu_file)
            
            # Initialize perf metrics file
            if not os.path.exists(self.perf_file):
                with open(self.perf_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "iteration", "epoch", "node_rank",
                        "world_size", "global_batch_size", "local_batch_size",
                        "time_per_batch", "data_loading_time", "forward_time", 
                        "backward_time", "optimizer_time", "all_reduce_time",
                        "comm_overhead_ratio", "train_loss", "samples_per_sec",
                        "learning_rate", "optimizer_buffer_mb"
                    ])
            
            # Initialize GPU metrics file if NVML is available
            if not os.path.exists(self.gpu_file):
                with open(self.gpu_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "iteration", "node_rank", "gpu_id", 
                        "utilization", "memory_used_mb", 
                        "temperature", "power_watts"
                    ])
            
            # Initialize image metrics
            self.predictions_dir = os.path.join(self.log_dir, "predictions")
            
            # if the directory exists, remove it
            if os.path.exists(self.predictions_dir):
                logging.info(f"Predictions directory exists... removing {self.predictions_dir}")
                shutil.rmtree(self.predictions_dir, ignore_errors=True)
            
            # Create organized subdirectories
            self.images_dir = os.path.join(self.predictions_dir, "images")
            self.data_dir = os.path.join(self.predictions_dir, "data")
            
            os.makedirs(self.predictions_dir, exist_ok=True)
            os.makedirs(self.images_dir, exist_ok=True)
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Create node-specific data directories for multi-node runs
            if self.world_size > 1:
                for rank in range(self.world_size):
                    os.makedirs(os.path.join(self.data_dir, f"node_{rank}"), exist_ok=True)
            
            # Initialize CSV file for image metrics
            self.img_metrics_file = os.path.join(self.log_dir, "image_metrics.csv")
            
            # Remove if it exists
            if os.path.exists(self.img_metrics_file):
                logging.info(f"Image metrics file exists... removing {self.img_metrics_file}")
                os.remove(self.img_metrics_file)
            
            # Create new file with headers
            with open(self.img_metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "iteration", "epoch", "node_rank",
                    "ssim", "rmse", "rmse_normalized"
                ])
            
            logging.info(f"Metrics Logger initialized. Logging every {metrics_log_freq} iterations to {log_dir}")
            logging.info(f"Predictions saved to: {self.predictions_dir}")
            logging.info(f"  - Images: {self.images_dir}")
            logging.info(f"  - Raw data: {self.data_dir}")
        
        # Initialize timing variables
        self.reset_timers()
    
    def reset_timers(self):
        """Reset all timing variables"""
        self.batch_start_time = 0
        self.data_load_start_time = 0
        self.forward_start_time = 0
        self.backward_start_time = 0
        self.optimizer_start_time = 0
        self.all_reduce_start_time = 0
        
        # Accumulated times for this iteration
        self.data_loading_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.optimizer_time = 0
        self.all_reduce_time = 0
    
    def should_log(self, iteration):
        """Check if we should log metrics for this iteration"""
        return iteration % self.metrics_log_freq == 0
    
    def on_batch_start(self):
        """Record the start of a batch"""
        self.batch_start_time = time.time()
    
    def on_data_load_start(self):
        """Record the start of data loading"""
        self.data_load_start_time = time.time()
    
    def on_data_load_end(self):
        """Record the end of data loading"""
        if self.data_load_start_time > 0:
            self.data_loading_time += time.time() - self.data_load_start_time
    
    def on_forward_start(self):
        """Record the start of forward pass"""
        self.forward_start_time = time.time()
    
    def on_forward_end(self):
        """Record the end of forward pass"""
        if self.forward_start_time > 0:
            self.forward_time += time.time() - self.forward_start_time
    
    def on_backward_start(self):
        """Record the start of backward pass"""
        self.backward_start_time = time.time()
    
    def on_backward_end(self):
        """Record the end of backward pass"""
        if self.backward_start_time > 0:
            self.backward_time += time.time() - self.backward_start_time
    
    def on_optimizer_start(self):
        """Record the start of optimizer step"""
        self.optimizer_start_time = time.time()
    
    def on_optimizer_end(self):
        """Record the end of optimizer step"""
        if self.optimizer_start_time > 0:
            self.optimizer_time += time.time() - self.optimizer_start_time
    
    def on_all_reduce_start(self):
        """Record the start of all-reduce operation"""
        self.all_reduce_start_time = time.time()
    
    def on_all_reduce_end(self):
        """Record the end of all-reduce operation"""
        if self.all_reduce_start_time > 0:
            self.all_reduce_time += time.time() - self.all_reduce_start_time
    
    def collect_gpu_metrics(self):
        """Collect GPU metrics if NVML is available"""
        if not self.nvml_initialized:
            return None
        
        try:
            gpu_metrics = []
            for i, handle in enumerate(self.gpu_handles):
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                
                # Memory usage
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem = mem_info.used / (1024 * 1024)  # MB
                
                # Temperature
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Power
                try:
                    gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
                except:
                    gpu_power = 0
                
                gpu_metrics.append({
                    'gpu_id': i,
                    'utilization': gpu_util,
                    'memory_used_mb': gpu_mem,
                    'temperature': gpu_temp,
                    'power_watts': gpu_power
                })
                
            return gpu_metrics
        except Exception as e:
            logging.warning(f"Rank {self.world_rank}: Failed to collect GPU metrics: {e}")
            return None
    
    def estimate_optimizer_buffer_size(self, optimizer):
        """Estimate the size of optimizer buffers in MB"""
        try:
            buffer_size = 0
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param.requires_grad:
                        # Count parameter size
                        buffer_size += param.nelement() * param.element_size()
                        
                        # Count optimizer state
                        if param in optimizer.state:
                            for state_value in optimizer.state[param].values():
                                if torch.is_tensor(state_value):
                                    buffer_size += state_value.nelement() * state_value.element_size()
            
            return buffer_size / (1024 * 1024)  # Convert to MB
        except:
            return 0
    
    def get_performance_metrics(self, iteration, epoch, loss, samples_per_sec, lr, optimizer=None,
                         global_batch_size=0, local_batch_size=0):
        """Collect performance metrics from this node
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate timing metrics
        now = time.time()
        batch_time = now - self.batch_start_time if self.batch_start_time > 0 else 0
        
        # Calculate communication overhead ratio
        if batch_time > 0:
            comm_overhead = self.all_reduce_time / batch_time
        else:
            comm_overhead = 0
        
        # Estimate optimizer buffer size if optimizer is provided
        optimizer_buffer_mb = 0
        if optimizer:
            optimizer_buffer_mb = self.estimate_optimizer_buffer_size(optimizer)
        
        # Prepare metrics dictionary
        metrics = {
            'timestamp': now,
            'iteration': iteration,
            'epoch': epoch,
            'node_rank': self.world_rank,
            'world_size': self.world_size,
            'global_batch_size': global_batch_size,
            'local_batch_size': local_batch_size,
            'time_per_batch': batch_time,
            'data_loading_time': self.data_loading_time,
            'forward_time': self.forward_time,
            'backward_time': self.backward_time,
            'optimizer_time': self.optimizer_time,
            'all_reduce_time': self.all_reduce_time,
            'comm_overhead_ratio': comm_overhead,
            'train_loss': loss,
            'samples_per_sec': samples_per_sec,
            'learning_rate': lr,
            'optimizer_buffer_mb': optimizer_buffer_mb
        }
        
        return metrics
    
    def gather_metrics(self, metrics):
        """Gather metrics from all processes to rank 0
        
        Args:
            metrics: Dictionary of metrics from current node
            
        Returns:
            List of metric dictionaries from all nodes (rank 0 only)
        """
        if not torch.distributed.is_initialized():
            return [metrics]  # Single-node case
            
        # Convert metrics dict to tensor format for gathering
        keys = list(metrics.keys())
        values = list(metrics.values())
        
        # Create a tensor to hold the values
        tensor_values = torch.zeros(len(values), dtype=torch.float64, device='cuda')
        for i, val in enumerate(values):
            tensor_values[i] = float(val) if val is not None else 0.0
            
        # Gather tensors from all processes
        gathered_tensors = [torch.zeros_like(tensor_values) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered_tensors, tensor_values)
        
        # Convert back to dictionaries
        all_metrics = []
        for tensor in gathered_tensors:
            node_metrics = {}
            for i, key in enumerate(keys):
                node_metrics[key] = tensor[i].item()
            all_metrics.append(node_metrics)
            
        return all_metrics
    
    def log_metrics(self, iteration, epoch, loss, samples_per_sec, lr, optimizer=None, 
                    global_batch_size=0, local_batch_size=0, image_fields=None, force=False):
        """Log metrics for current iteration from all nodes
        
        Args:
            iteration: Current iteration number
            epoch: Current epoch
            loss: Training loss
            samples_per_sec: Throughput in samples/second
            lr: Current learning rate
            optimizer: Optimizer instance (to estimate buffer size)
            global_batch_size: Global batch size
            local_batch_size: Local batch size
            image_fields: Optional list of tensors [inp, tar, gen] for image metrics
            force: Whether to force logging regardless of frequency
            
        Returns:
            img_metrics: Dictionary of image metrics if image_fields was provided (main process only), else None
        """
        if not (self.should_log(iteration) or force):
            return None
        
        # Get performance metrics from this node
        metrics = self.get_performance_metrics(
            iteration, epoch, loss, samples_per_sec, lr, 
            optimizer, global_batch_size, local_batch_size
        )
        
        # For multi-node setup, gather metrics from all nodes to rank 0
        if self.world_size > 1 and torch.distributed.is_initialized():
            try:
                # Use simple tensor-based approach for performance metrics
                all_metrics = self.gather_metrics(metrics)
                
                # Write all gathered metrics to CSV (only rank 0)
                if self.is_main_process:
                    with open(self.perf_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for node_metrics in all_metrics:
                            writer.writerow([
                                node_metrics['timestamp'],
                                node_metrics['iteration'],
                                node_metrics['epoch'],
                                node_metrics['node_rank'],
                                node_metrics['world_size'],
                                node_metrics['global_batch_size'],
                                node_metrics['local_batch_size'],
                                node_metrics['time_per_batch'],
                                node_metrics['data_loading_time'],
                                node_metrics['forward_time'],
                                node_metrics['backward_time'],
                                node_metrics['optimizer_time'],
                                node_metrics['all_reduce_time'],
                                node_metrics['comm_overhead_ratio'],
                                node_metrics['train_loss'],
                                node_metrics['samples_per_sec'],
                                node_metrics['learning_rate'],
                                node_metrics['optimizer_buffer_mb']
                            ])
            except Exception as e:
                logging.error(f"Error gathering metrics: {e}")
                # Fall back to logging only this node's metrics
                if self.is_main_process:
                    with open(self.perf_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            metrics['timestamp'], metrics['iteration'], metrics['epoch'], metrics['node_rank'],
                            metrics['world_size'], metrics['global_batch_size'], metrics['local_batch_size'],
                            metrics['time_per_batch'], metrics['data_loading_time'], metrics['forward_time'],
                            metrics['backward_time'], metrics['optimizer_time'], metrics['all_reduce_time'],
                            metrics['comm_overhead_ratio'], metrics['train_loss'], metrics['samples_per_sec'],
                            metrics['learning_rate'], metrics['optimizer_buffer_mb']
                        ])
        else:
            # Single node case - just log metrics directly
            if self.is_main_process:
                with open(self.perf_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        metrics['timestamp'], metrics['iteration'], metrics['epoch'], metrics['node_rank'],
                        metrics['world_size'], metrics['global_batch_size'], metrics['local_batch_size'],
                        metrics['time_per_batch'], metrics['data_loading_time'], metrics['forward_time'],
                        metrics['backward_time'], metrics['optimizer_time'], metrics['all_reduce_time'],
                        metrics['comm_overhead_ratio'], metrics['train_loss'], metrics['samples_per_sec'],
                        metrics['learning_rate'], metrics['optimizer_buffer_mb']
                    ])
        
        # Log GPU metrics (from each rank independently)
        gpu_metrics = self.collect_gpu_metrics()
        if gpu_metrics:
            # Gather GPU metrics from all nodes
            if self.world_size > 1 and torch.distributed.is_initialized():
                # Simplified approach: just signal main process to collect
                gpu_metrics_signal = torch.tensor([1.0] if gpu_metrics else [0.0], device='cuda')
                
                if self.is_main_process:
                    # Collect signals from all processes
                    gathered_signals = [torch.zeros_like(gpu_metrics_signal) for _ in range(self.world_size)]
                    torch.distributed.all_gather(gathered_signals, gpu_metrics_signal)
                    
                    # Write GPU metrics from this node
                    with open(self.gpu_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for gpu in gpu_metrics:
                            writer.writerow([
                                metrics['timestamp'], metrics['iteration'], metrics['node_rank'], gpu['gpu_id'],
                                gpu['utilization'], gpu['memory_used_mb'],
                                gpu['temperature'], gpu['power_watts']
                            ])
                    
                    # Signal each node to save their GPU metrics one by one
                    for other_rank in range(1, self.world_size):
                        if gathered_signals[other_rank][0] > 0:
                            # Signal this rank to send GPU metrics
                            signal_tensor = torch.tensor([other_rank], device='cuda')
                            torch.distributed.broadcast(signal_tensor, src=0)
                            
                            # Receive GPU metrics count
                            count_tensor = torch.zeros(1, dtype=torch.long, device='cuda')
                            torch.distributed.broadcast(count_tensor, src=other_rank)
                            gpu_count = count_tensor[0].item()
                            
                            # Receive and write GPU metrics
                            for _ in range(gpu_count):
                                # Receive metrics tensor (7 values per GPU)
                                metrics_tensor = torch.zeros(7, dtype=torch.float64, device='cuda')
                                torch.distributed.broadcast(metrics_tensor, src=other_rank)
                                
                                # Write to file
                                with open(self.gpu_file, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([
                                        metrics['timestamp'], metrics['iteration'], other_rank, 
                                        int(metrics_tensor[0].item()),  # gpu_id
                                        metrics_tensor[1].item(),       # utilization
                                        metrics_tensor[2].item(),       # memory_used_mb
                                        metrics_tensor[3].item(),       # temperature
                                        metrics_tensor[4].item()        # power_watts
                                    ])
                    
                    # Signal completion
                    signal_tensor = torch.tensor([-1], device='cuda')
                    torch.distributed.broadcast(signal_tensor, src=0)
                else:
                    # Wait for signal from main process
                    while True:
                        signal_tensor = torch.zeros(1, dtype=torch.long, device='cuda')
                        torch.distributed.broadcast(signal_tensor, src=0)
                        signal = signal_tensor[0].item()
                        
                        # Check if we should send our metrics
                        if signal == self.world_rank:
                            # Send GPU count
                            count_tensor = torch.tensor([len(gpu_metrics)], dtype=torch.long, device='cuda')
                            torch.distributed.broadcast(count_tensor, src=self.world_rank)
                            
                            # Send GPU metrics
                            for gpu in gpu_metrics:
                                metrics_tensor = torch.tensor([
                                    float(gpu['gpu_id']),
                                    float(gpu['utilization']),
                                    float(gpu['memory_used_mb']),
                                    float(gpu['temperature']),
                                    float(gpu['power_watts']),
                                    0.0,  # Padding
                                    0.0   # Padding
                                ], dtype=torch.float64, device='cuda')
                                torch.distributed.broadcast(metrics_tensor, src=self.world_rank)
                        elif signal < 0:
                            # End of collection
                            break
            else:
                # Single node case - just log directly
                if self.is_main_process:
                    with open(self.gpu_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for gpu in gpu_metrics:
                            writer.writerow([
                                metrics['timestamp'], metrics['iteration'], self.world_rank, gpu['gpu_id'],
                                gpu['utilization'], gpu['memory_used_mb'],
                                gpu['temperature'], gpu['power_watts']
                            ])
        
        # Handle image metrics if provided
        img_metrics = None
        if image_fields is not None:
            # For image metrics, let each node save its own data
            if self.is_main_process:
                img_metrics = self.log_image_metrics(image_fields, iteration, epoch, force=True)
                self.save_comparison_image(image_fields, iteration, epoch, force=True)
            else:
                # Save data for non-rank-0 nodes
                self.save_node_prediction_data(image_fields, iteration, epoch)
        
        # Reset timers for next iteration
        self.reset_timers()
        
        return img_metrics if self.is_main_process else None
    
    def save_node_prediction_data(self, fields, iteration, epoch):
        """Save prediction data from non-rank-0 nodes
        
        This function saves raw data (.npz files) from other nodes without
        calculating metrics or creating visualizations.
        
        Args:
            fields: List of tensors [input, target, prediction]
            iteration: Current iteration number
            epoch: Current epoch number
        """
        if self.is_main_process:
            return  # Main process handles this separately
        
        # Create base filename for this node
        base_filename = f"iter_{iteration}"
        if epoch is not None:
            base_filename = f"epoch_{epoch}_" + base_filename
        
        # Extract data
        inp, tar, gen = [x.detach().float().cpu().numpy() for x in fields]
        
        # Save raw data to temporary file
        temp_npz_path = f"/tmp/node_{self.world_rank}_{base_filename}.npz"
        np.savez(
            temp_npz_path,
            target=tar[0, 2, :, :],
            prediction=gen[0, 2, :, :],
            input=inp[0, 2, :, :] if inp.shape[1] > 2 else inp[0, 0, :, :]
        )
        
        # Signal main process that data is ready
        if torch.distributed.is_initialized():
            signal_tensor = torch.tensor([1.0], device='cuda')
            torch.distributed.all_reduce(signal_tensor)
            
            # Wait for main process to signal it's received the data
            while True:
                # Wait for signal to read data
                signal_tensor = torch.zeros(1, dtype=torch.long, device='cuda')
                torch.distributed.broadcast(signal_tensor, src=0)
                signal = signal_tensor[0].item()
                
                if signal == self.world_rank:
                    # Send file size first
                    file_size = os.path.getsize(temp_npz_path)
                    size_tensor = torch.tensor([file_size], dtype=torch.long, device='cuda')
                    torch.distributed.broadcast(size_tensor, src=self.world_rank)
                    
                    # Send file contents in chunks
                    with open(temp_npz_path, 'rb') as f:
                        file_data = f.read()
                    
                    # Convert bytes to tensor and send
                    chunk_size = 1024 * 1024  # 1MB chunks
                    for i in range(0, len(file_data), chunk_size):
                        chunk = file_data[i:i+chunk_size]
                        # Pad chunk if needed
                        if len(chunk) < chunk_size:
                            chunk = chunk + b'\0' * (chunk_size - len(chunk))
                        
                        # Create tensor from bytes
                        chunk_tensor = torch.tensor([ord(b) for b in chunk], dtype=torch.uint8, device='cuda')
                        torch.distributed.broadcast(chunk_tensor, src=self.world_rank)
                        
                        # Check if this was the last chunk
                        is_last = (i + chunk_size >= len(file_data))
                        last_tensor = torch.tensor([1 if is_last else 0], dtype=torch.uint8, device='cuda')
                        torch.distributed.broadcast(last_tensor, src=self.world_rank)
                        
                        if is_last:
                            break
                elif signal < 0:
                    # Collection completed
                    break
            
            # Clean up temporary file
            if os.path.exists(temp_npz_path):
                os.remove(temp_npz_path)
    
    def finalize(self):
        """Cleanup when training is complete"""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

    def log_image_metrics(self, fields, iteration, epoch, force=False):
        """
        Calculate, log, and save image metrics and data
        
        Args:
            fields: List of tensors [input, target, prediction]
            iteration: Current iteration number
            epoch: Current epoch number
            force: Whether to force logging regardless of frequency
        
        Returns:
            metrics: Dictionary of calculated metrics
        """
        if not (self.should_log(iteration) or force) or not self.is_main_process:
            return None
        
        inp, tar, gen = [x.detach().float().cpu().numpy() for x in fields]
        
        # Calculate image metrics
        metrics = {}
        
        # Calculate SSIM (structural similarity index)
        # Normalize data for SSIM calculation (requires data in range [0,1])
        tar_norm = tar[0, 2, :, :]
        gen_norm = gen[0, 2, :, :]
        
        # Check if normalization is needed
        if np.max(tar_norm) > 1.0 or np.min(tar_norm) < 0.0:
            # Simple min-max normalization
            tar_min, tar_max = np.min(tar_norm), np.max(tar_norm)
            if tar_max > tar_min:  # Avoid division by zero
                tar_norm = (tar_norm - tar_min) / (tar_max - tar_min)
            
            gen_min, gen_max = np.min(gen_norm), np.max(gen_norm)
            if gen_max > gen_min:  # Avoid division by zero
                gen_norm = (gen_norm - gen_min) / (gen_max - gen_min)
        
        # Calculate SSIM
        try:
            ssim_value = ssim(tar_norm, gen_norm, data_range=1.0)
            metrics['ssim'] = float(ssim_value)
        except Exception as e:
            metrics['ssim'] = -1.0
            metrics['ssim_error'] = str(e)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((tar[0, 2, :, :] - gen[0, 2, :, :])**2))
        metrics['rmse'] = float(rmse)
        
        # Add normalized RMSE (as percentage of data range)
        tar_range = np.max(tar[0, 2, :, :]) - np.min(tar[0, 2, :, :])
        if tar_range > 0:
            metrics['rmse_normalized'] = float(rmse / tar_range)
        else:
            metrics['rmse_normalized'] = 0.0
        
        # Create base filename for saving
        base_filename = f"iter_{iteration}"
        if epoch is not None:
            base_filename = f"epoch_{epoch}_" + base_filename
        
        # Save image metrics to CSV
        now = time.time()
        with open(self.img_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                now, iteration, epoch, self.world_rank,
                metrics.get('ssim', -1),
                metrics.get('rmse', -1),
                metrics.get('rmse_normalized', -1)
            ])
        
        # Save raw image data as NPZ in data directory
        np.savez(
            os.path.join(self.data_dir, f"node_{self.world_rank}_{base_filename}.npz"),
            target=tar[0, 2, :, :],
            prediction=gen[0, 2, :, :],
            input=inp[0, 2, :, :] if inp.shape[1] > 2 else inp[0, 0, :, :]
        )
        
        return metrics

    def save_comparison_image(self, fields, iteration, epoch, force=False):
        """
        Generate and save a comparison image of target vs prediction
        
        Args:
            fields: List of tensors [input, target, prediction]
            iteration: Current iteration number
            epoch: Current epoch number
            force: Whether to force logging regardless of frequency
        
        Returns:
            fig: Matplotlib figure object
        """
        if not (self.should_log(iteration) or force) or not self.is_main_process:
            return None
        
        inp, tar, gen = [x.detach().float().cpu().numpy() for x in fields]
        
        # Create visualization figure
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        plt.title('2m temperature')
        ax[0].imshow(tar[0, 2, :, :], cmap="turbo")
        ax[0].set_title("ERA5 target")
        ax[1].imshow(gen[0, 2, :, :], cmap="turbo")
        ax[1].set_title("ViT prediction")
        fig.tight_layout()
        
        # Create base filename
        base_filename = f"iter_{iteration}"
        if epoch is not None:
            base_filename = f"epoch_{epoch}_" + base_filename
        
        # Save figure as image in images directory
        fig_path = os.path.join(self.images_dir, f"node_{self.world_rank}_{base_filename}.png")
        fig.savefig(fig_path, dpi=100)
        plt.close(fig)  # Close the figure to free memory
        
        return fig