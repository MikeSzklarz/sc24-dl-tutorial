"""Asynchronous file writer for metrics data to avoid I/O bottlenecks during training"""

import os
import json
import time
import threading
import queue

class AsyncMetricsWriter:
    """Writes metrics to files asynchronously to avoid blocking the training process"""
    
    def __init__(self, file_path, buffer_size=100, flush_interval=5):
        """
        Initialize asynchronous writer
        
        Args:
            file_path: Path to output file
            buffer_size: Number of items to buffer before writing
            flush_interval: Time in seconds between forced flushes
        """
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        self.buffer = []
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.last_flush = time.time()
        
        # Start writer thread
        self.running = True
        self.thread = threading.Thread(target=self._writer_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def write(self, data):
        """Add data to buffer"""
        self.queue.put(data)
    
    def _writer_loop(self):
        """Background thread for writing data"""
        while self.running:
            # Check if it's time to flush based on interval
            if time.time() - self.last_flush > self.flush_interval:
                self._flush_buffer()
            
            # Get items from queue with timeout
            try:
                item = self.queue.get(timeout=0.5)
                with self.lock:
                    self.buffer.append(item)
                self.queue.task_done()
            except queue.Empty:
                pass
            
            # Check if buffer is full
            with self.lock:
                if len(self.buffer) >= self.buffer_size:
                    self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush buffer to file"""
        with self.lock:
            if not self.buffer:
                return
            
            try:
                # Append to file
                with open(self.file_path, "a") as f:
                    for item in self.buffer:
                        f.write(json.dumps(item) + "\n")
                
                self.buffer = []
                self.last_flush = time.time()
            except Exception as e:
                print(f"Error writing to {self.file_path}: {e}")
    
    def flush(self):
        """Manually flush buffer"""
        self._flush_buffer()
    
    def close(self):
        """Close writer and flush remaining data"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)  # Don't wait indefinitely
        self._flush_buffer()