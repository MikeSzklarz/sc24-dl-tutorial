"""System metrics monitoring for distributed training"""

import time
import threading
import torch
import os

# Optional dependencies with graceful fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

class SystemMetricsMonitor:
    """Background thread for monitoring system metrics during training"""
    
    def __init__(self, logger, interval=5):
        """
        Initialize system metrics monitor
        
        Args:
            logger: DistributedMetricsLogger instance
            interval: Polling interval in seconds
        """
        self.logger = logger
        self.interval = interval
        self.running = False
        self.thread = None
        
        # Initialize GPU monitoring
        self.gpu_count = 0
        self.handles = []
        
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            
            if PYNVML_AVAILABLE:
                try:
                    device_count = pynvml.nvmlDeviceGetCount()
                    self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
                except Exception as e:
                    print(f"Warning: Error initializing NVML: {e}")
    
    def start(self):
        """Start monitoring thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        """Stop monitoring thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)  # Don't wait indefinitely
            self.thread = None
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            # Collect system metrics
            system_metrics = self._collect_system_metrics()
            self.logger.log_system_metrics(system_metrics)
            
            # Collect GPU metrics
            gpu_metrics = self._collect_gpu_metrics()
            if gpu_metrics:
                self.logger.log_gpu_metrics({"gpus": gpu_metrics})
            
            time.sleep(self.interval)
    
    def _collect_system_metrics(self):
        """Collect CPU and memory metrics"""
        metrics = {
            "cpu": self._collect_cpu_metrics(),
            "memory": self._collect_memory_metrics(),
            "network": self._collect_network_metrics()
        }
        return metrics
    
    def _collect_cpu_metrics(self):
        """Collect CPU metrics"""
        if PSUTIL_AVAILABLE:
            try:
                return {
                    "utilization_pct": psutil.cpu_percent(interval=0.1),
                    "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                }
            except Exception:
                pass
        
        # Fallback using /proc/stat on Linux
        try:
            with open('/proc/stat', 'r') as f:
                cpu_stats = f.readline().split()
                user = float(cpu_stats[1])
                nice = float(cpu_stats[2])
                system = float(cpu_stats[3])
                idle = float(cpu_stats[4])
                total = user + nice + system + idle
                return {
                    "utilization_pct": 100 * (1 - idle / total),
                    "load_avg": None
                }
        except Exception:
            return {
                "utilization_pct": None,
                "load_avg": None
            }
    
    def _collect_memory_metrics(self):
        """Collect system memory metrics"""
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                return {
                    "total_gb": mem.total / (1024**3),
                    "used_gb": mem.used / (1024**3),
                    "percent": mem.percent
                }
            except Exception:
                pass
        
        # Fallback using /proc/meminfo on Linux
        try:
            meminfo = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().split()[0]
                        meminfo[key] = int(value)
            
            total = meminfo.get('MemTotal', 0) / (1024**2)  # Convert to GB
            free = meminfo.get('MemFree', 0) / (1024**2)
            available = meminfo.get('MemAvailable', free) / (1024**2)
            used = total - available
            percent = (used / total) * 100 if total > 0 else 0
            
            return {
                "total_gb": total,
                "used_gb": used,
                "percent": percent
            }
        except Exception:
            return {
                "total_gb": None,
                "used_gb": None,
                "percent": None
            }
    
    def _collect_network_metrics(self):
        """Collect network metrics"""
        if PSUTIL_AVAILABLE:
            try:
                net_io = psutil.net_io_counters()
                return {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            except Exception:
                pass
        
        # Fallback using /proc/net/dev on Linux
        try:
            net_stats = {"bytes_sent": 0, "bytes_recv": 0}
            with open('/proc/net/dev', 'r') as f:
                lines = f.readlines()
                for line in lines[2:]:  # Skip headers
                    parts = line.strip().split()
                    if len(parts) >= 10 and ':' in parts[0]:
                        interface = parts[0].split(':')[0]
                        if interface != 'lo':  # Skip loopback
                            net_stats["bytes_recv"] += int(parts[1])
                            net_stats["bytes_sent"] += int(parts[9])
            return net_stats
        except Exception:
            return {
                "bytes_sent": None,
                "bytes_recv": None
            }
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics"""
        if not torch.cuda.is_available():
            return []
        
        gpu_metrics = []
        
        # Try using NVML first for detailed metrics
        if PYNVML_AVAILABLE and self.handles:
            for i, handle in enumerate(self.handles):
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert from mW to W
                    
                    gpu_metrics.append({
                        "index": i,
                        "memory_total_gb": mem_info.total / (1024**3),
                        "memory_used_gb": mem_info.used / (1024**3),
                        "memory_percent": (mem_info.used * 100 / mem_info.total) if mem_info.total > 0 else 0,
                        "utilization_gpu_percent": util_info.gpu,
                        "utilization_memory_percent": util_info.memory,
                        "temperature_c": temp,
                        "power_usage_w": power
                    })
                except Exception:
                    # If error occurs, try simpler PyTorch-based metrics for this GPU
                    gpu_metrics.append(self._get_pytorch_gpu_metrics(i))
        else:
            # Fallback to PyTorch for basic metrics
            for i in range(self.gpu_count):
                gpu_metrics.append(self._get_pytorch_gpu_metrics(i))
        
        return gpu_metrics
    
    def _get_pytorch_gpu_metrics(self, gpu_idx):
        """Get basic GPU metrics using PyTorch"""
        try:
            # Get device properties
            props = torch.cuda.get_device_properties(gpu_idx)
            
            # Get memory usage
            memory_allocated = torch.cuda.memory_allocated(gpu_idx) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(gpu_idx) / (1024**3)    # GB
            memory_total = props.total_memory / (1024**3)                        # GB
            
            return {
                "index": gpu_idx,
                "name": props.name,
                "memory_total_gb": memory_total,
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
                "memory_percent": ((memory_allocated + memory_reserved) * 100 / memory_total) if memory_total > 0 else 0
            }
        except Exception as e:
            return {"index": gpu_idx, "error": str(e)}