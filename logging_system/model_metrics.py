"""Model parameter and gradient metrics collection"""

import torch
import torch.nn as nn

class ModelMetricsHook:
    """Hook for collecting model parameter and gradient statistics"""
    
    def __init__(self, logger, model, log_frequency=100):
        """
        Initialize model metrics hook
        
        Args:
            logger: DistributedMetricsLogger instance
            model: PyTorch model to monitor
            log_frequency: Log metrics every N iterations
        """
        self.logger = logger
        self.model = model
        self.log_frequency = log_frequency
        self.iteration = 0
        
        # Extract the actual model from DDP wrapper if necessary
        self.unwrapped_model = model.module if hasattr(model, 'module') else model
        
        # Register hooks
        self.handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks on model parameters"""
        for name, module in self.unwrapped_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.LayerNorm, nn.BatchNorm2d)):
                # Add backward hook to capture gradients
                self.handles.append(
                    module.register_full_backward_hook(
                        lambda mod, grad_in, grad_out, module_name=name:
                            self._backward_hook(mod, grad_in, grad_out, module_name)
                    )
                )
    
    def _backward_hook(self, module, grad_input, grad_output, module_name):
        """Collect gradient statistics during backward pass"""
        # Only log every log_frequency iterations
        if self.iteration % self.log_frequency != 0:
            return
        
        # Process the output gradient
        if grad_output and isinstance(grad_output, tuple) and len(grad_output) > 0:
            grad = grad_output[0]
            if grad is not None:
                # Calculate gradient statistics
                if torch.is_tensor(grad) and grad.numel() > 0:
                    try:
                        grad_norm = torch.norm(grad).item()
                        grad_abs_mean = torch.mean(torch.abs(grad)).item()
                        grad_std = torch.std(grad).item()
                        grad_max = torch.max(grad).item()
                        grad_min = torch.min(grad).item()
                        
                        # Log gradient metrics
                        self.logger.log_gradient_metrics({
                            "iteration": self.iteration,
                            "module_name": module_name,
                            "grad_norm": grad_norm,
                            "grad_abs_mean": grad_abs_mean,
                            "grad_std": grad_std,
                            "grad_max": grad_max,
                            "grad_min": grad_min
                        })
                    except Exception as e:
                        print(f"Error calculating gradient stats for {module_name}: {e}")
    
    def log_parameter_stats(self):
        """Log parameter statistics for the current iteration"""
        # Only log every log_frequency iterations
        if self.iteration % self.log_frequency != 0:
            return
        
        # Collect statistics for each parameter group
        for name, param in self.unwrapped_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Parameter statistics
                param_norm = torch.norm(param.data).item()
                param_mean = torch.mean(param.data).item()
                param_std = torch.std(param.data).item()
                
                # Gradient statistics
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                else:
                    grad_norm = 0.0
                
                # Log parameter metrics
                self.logger.log_parameter_metrics({
                    "iteration": self.iteration,
                    "parameter_name": name,
                    "param_norm": param_norm,
                    "param_mean": param_mean,
                    "param_std": param_std,
                    "grad_norm": grad_norm
                })
    
    def update_iteration(self, iteration):
        """Update current iteration counter"""
        self.iteration = iteration
        
    def remove_hooks(self):
        """Remove registered hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []