"""Utilities for visualizing distributed training metrics"""

import os
import json
import math
import numpy as np

class MetricsVisualizer:
    """Generates visualization-friendly data formats for metrics"""
    
    def __init__(self, log_dir, rank=0):
        """
        Initialize metrics visualizer
        
        Args:
            log_dir: Directory containing log files
            rank: Process rank (only rank 0 generates visualizations)
        """
        self.log_dir = log_dir
        self.rank = rank
        self.viz_dir = os.path.join(log_dir, "visualization")
        
        # Create visualization directory if needed
        if rank == 0:
            os.makedirs(self.viz_dir, exist_ok=True)
    
    def generate_scaling_efficiency_data(self, world_size, samples_per_sec):
        """
        Generate scaling efficiency visualization data
        
        Args:
            world_size: Number of processes
            samples_per_sec: Samples processed per second
        """
        if self.rank != 0:
            return
        
        # Calculate ideal linear scaling (assuming perfect scaling from 1 node)
        base_throughput = samples_per_sec / world_size  # Estimated single-node performance
        ideal_scaling = [base_throughput * i for i in range(1, world_size + 1)]
        
        # Calculate actual scaling efficiency
        efficiency = (samples_per_sec / ideal_scaling[-1]) * 100 if ideal_scaling[-1] > 0 else 0
        
        # Create visualization data
        scaling_data = {
            "world_size": world_size,
            "actual_throughput": samples_per_sec,
            "ideal_throughput": ideal_scaling[-1],
            "scaling_efficiency_pct": efficiency,
            "chart_data": {
                "nodes": list(range(1, world_size + 1)),
                "ideal_scaling": ideal_scaling,
                "actual_performance": [samples_per_sec]
            }
        }
        
        # Save to file
        with open(os.path.join(self.viz_dir, "scaling_efficiency.json"), "w") as f:
            json.dump(scaling_data, f, indent=2)
        
        return scaling_data
    
    def generate_performance_breakdown(self, training_metrics_file, n_samples=100):
        """
        Generate performance breakdown visualization data
        
        Args:
            training_metrics_file: Path to training metrics file
            n_samples: Number of samples to include (uses last n_samples)
        """
        if self.rank != 0:
            return
        
        # Load training metrics
        metrics = []
        try:
            with open(training_metrics_file, "r") as f:
                for line in f:
                    metrics.append(json.loads(line.strip()))
        except Exception as e:
            print(f"Error loading training metrics: {e}")
            return None
        
        # Use last n_samples
        metrics = metrics[-n_samples:] if len(metrics) > n_samples else metrics
        
        # Extract timing data
        iterations = [m.get("iteration", i) for i, m in enumerate(metrics)]
        
        time_components = []
        for m in metrics:
            if "timings" in m:
                component = {
                    "data_loading": m["timings"].get("data_loading", 0),
                    "forward_pass": m["timings"].get("forward_pass", 0),
                    "backward_pass": m["timings"].get("backward_pass", 0),
                    "communication": m["timings"].get("communication", 0),
                    "optimizer": m["timings"].get("optimizer", 0)
                }
                
                # Calculate "other" time
                total = m["timings"].get("total", 0)
                component_sum = sum(component.values())
                if total > component_sum:
                    component["other"] = total - component_sum
                
                time_components.append(component)
            else:
                time_components.append({})
        
        # Create stacked time series data
        stacked_data = {
            "iterations": iterations,
            "time_components": time_components
        }
        
        # Calculate average breakdown
        if time_components:
            avg_breakdown = {}
            for key in time_components[0].keys():
                values = [tc.get(key, 0) for tc in time_components]
                avg_breakdown[key] = sum(values) / len(values) if values else 0
            
            # Calculate percentages
            total_time = sum(avg_breakdown.values())
            if total_time > 0:
                avg_breakdown_pct = {k: (v / total_time) * 100 for k, v in avg_breakdown.items()}
            else:
                avg_breakdown_pct = {k: 0 for k in avg_breakdown.keys()}
            
            stacked_data["average_breakdown"] = avg_breakdown
            stacked_data["average_breakdown_pct"] = avg_breakdown_pct
        
        # Save to file
        with open(os.path.join(self.viz_dir, "performance_breakdown.json"), "w") as f:
            json.dump(stacked_data, f, indent=2)
        
        return stacked_data
    
    def generate_convergence_plot_data(self, epoch_summaries_file):
        """
        Generate convergence plot visualization data
        
        Args:
            epoch_summaries_file: Path to epoch summaries file
        """
        if self.rank != 0:
            return
        
        # Load epoch summaries
        summaries = []
        try:
            with open(epoch_summaries_file, "r") as f:
                for line in f:
                    summaries.append(json.loads(line.strip()))
        except Exception as e:
            print(f"Error loading epoch summaries: {e}")
            return None
        
        # Extract convergence data
        epochs = [s.get("epoch", i) for i, s in enumerate(summaries)]
        train_losses = [s.get("train_loss", None) for s in summaries]
        val_losses = [s.get("val_loss", None) for s in summaries]
        val_rmse = [s.get("val_rmse", None) for s in summaries]
        
        # Create convergence data
        convergence_data = {
            "epochs": epochs,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_rmse": val_rmse
        }
        
        # Save to file
        with open(os.path.join(self.viz_dir, "convergence_plot.json"), "w") as f:
            json.dump(convergence_data, f, indent=2)
        
        return convergence_data
    
    def generate_bottleneck_summary(self, bottlenecks_file):
        """
        Generate bottleneck summary visualization data
        
        Args:
            bottlenecks_file: Path to bottlenecks file
        """
        if self.rank != 0:
            return
        
        # Load bottleneck data
        bottlenecks = []
        try:
            with open(bottlenecks_file, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if "bottlenecks" in entry:
                        bottlenecks.extend(entry["bottlenecks"])
        except Exception as e:
            print(f"Error loading bottlenecks: {e}")
            return None
        
        # Count bottleneck types
        bottleneck_types = {}
        for b in bottlenecks:
            b_type = b.get("type", "unknown")
            if b_type not in bottleneck_types:
                bottleneck_types[b_type] = {
                    "count": 0,
                    "high_severity": 0,
                    "medium_severity": 0,
                    "low_severity": 0,
                    "avg_percentage": 0,
                    "recommendations": set()
                }
            
            bottleneck_types[b_type]["count"] += 1
            severity = b.get("severity", "low")
            bottleneck_types[b_type][f"{severity}_severity"] += 1
            
            if "percentage" in b:
                current_avg = bottleneck_types[b_type]["avg_percentage"]
                current_count = bottleneck_types[b_type]["count"]
                new_avg = ((current_avg * (current_count - 1)) + b["percentage"]) / current_count
                bottleneck_types[b_type]["avg_percentage"] = new_avg
            
            if "recommendation" in b:
                bottleneck_types[b_type]["recommendations"].add(b["recommendation"])
        
        # Convert sets to lists for JSON serialization
        for b_type in bottleneck_types:
            bottleneck_types[b_type]["recommendations"] = list(bottleneck_types[b_type]["recommendations"])
        
        # Create summary
        bottleneck_summary = {
            "total_bottlenecks": len(bottlenecks),
            "bottleneck_types": bottleneck_types
        }
        
        # Save to file
        with open(os.path.join(self.viz_dir, "bottleneck_summary.json"), "w") as f:
            json.dump(bottleneck_summary, f, indent=2)
        
        return bottleneck_summary
    
    def generate_all_visualizations(self):
        """Generate all visualization data from log files"""
        if self.rank != 0:
            return
        
        # Define common file paths
        training_dir = os.path.join(self.log_dir, "training_metrics")
        system_dir = os.path.join(self.log_dir, "system_metrics")
        
        # Generate convergence plot
        epoch_summaries_file = os.path.join(training_dir, "epoch_summaries.jsonl")
        if os.path.exists(epoch_summaries_file):
            self.generate_convergence_plot_data(epoch_summaries_file)
        
        # Generate performance breakdown
        rank0_iterations_file = os.path.join(training_dir, "rank_0_iterations.jsonl")
        if os.path.exists(rank0_iterations_file):
            self.generate_performance_breakdown(rank0_iterations_file)
        
        # Generate bottleneck summary
        bottlenecks_file = os.path.join(self.viz_dir, "bottlenecks.jsonl")
        if os.path.exists(bottlenecks_file):
            self.generate_bottleneck_summary(bottlenecks_file)