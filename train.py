import sys
import os
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils import get_data_loader_distributed
from utils.loss import l2_loss, l2_loss_opt
from utils.metrics import weighted_rmse
from utils.plots import generate_images
from networks import vit

import psutil
import platform
import pynvml
import socket
from datetime import timedelta, datetime

def log_system_info(params, world_rank, local_rank, tboard_writer=None):
    """Log detailed system and environment information"""
    if world_rank == 0:
        system_info = {
            "System": platform.system(),
            "Node": socket.gethostname(),
            "Release": platform.release(),
            "Version": platform.version(),
            "Machine": platform.machine(),
            "Processor": platform.processor(),
            "Python": platform.python_version(),
            "PyTorch": torch.__version__,
            "CUDA Available": torch.cuda.is_available(),
            "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "CUDA Devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "CPU Count": os.cpu_count(),
            "RAM Total (GB)": round(psutil.virtual_memory().total / (1024**3), 2)
        }
        
        # Log CUDA device properties for each GPU
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                system_info[f"GPU {i}"] = f"{props.name}, {props.total_memory / (1024**3):.2f} GB"
        
        # Log to file
        logging.info("==================== SYSTEM INFORMATION ====================")
        for key, value in system_info.items():
            logging.info(f"{key}: {value}")
        logging.info("============================================================")
        
        # Log to TensorBoard
        if tboard_writer:
            # Create text summary for system info
            system_info_text = "\n".join([f"**{k}**: {v}" for k, v in system_info.items()])
            tboard_writer.add_text("System Information", system_info_text, 0)
            
    # Log the current GPU info for this rank
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(local_rank)
        logging.info(f"Rank {world_rank} using GPU {local_rank}: {device_props.name} with {device_props.total_memory / (1024**3):.2f} GB")

def init_gpu_monitoring():
    """Initialize GPU monitoring using pynvml"""
    try:
        pynvml.nvmlInit()
        return True
    except:
        logging.warning("Failed to initialize NVML for GPU monitoring")
        return False
    
def get_gpu_metrics(device_id):
    """Get detailed GPU metrics using pynvml"""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        
        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used_gb = mem_info.used / (1024**3)
        memory_utilization = 100 * mem_info.used / mem_info.total
        
        # Utilization rates
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = utilization.gpu
        memory_io_utilization = utilization.memory
        
        # Temperature
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        # Power usage
        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # convert from mW to W
        
        return {
            "memory_used_gb": memory_used_gb,
            "memory_utilization": memory_utilization,
            "gpu_utilization": gpu_utilization,
            "memory_io_utilization": memory_io_utilization,
            "temperature": temperature,
            "power_usage": power_usage
        }
    except:
        return None

def log_gpu_metrics(metrics, epoch, iteration, tboard_writer, prefix="GPU"):
    """Log GPU metrics to TensorBoard"""
    if metrics:
        for key, value in metrics.items():
            tboard_writer.add_scalar(f"{prefix}/{key}", value, iteration)

def log_cpu_metrics(epoch, iteration, tboard_writer):
    """Log CPU and memory utilization metrics"""
    cpu_percent = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024**3)
    memory_percent = memory.percent
    
    tboard_writer.add_scalar("System/CPU/utilization_percent", cpu_percent, iteration)
    tboard_writer.add_scalar("System/Memory/used_gb", memory_used_gb, iteration)
    tboard_writer.add_scalar("System/Memory/percent", memory_percent, iteration)

def log_distributed_metrics(comm_start_time, comm_end_time, iteration, world_size, tboard_writer):
    """Log communication overhead metrics for distributed training"""
    comm_time = comm_end_time - comm_start_time
    tboard_writer.add_scalar("Distributed/communication_time_ms", comm_time * 1000, iteration)
    tboard_writer.add_scalar("Distributed/communication_percent", 
                           (comm_time / (comm_time + 1e-6)) * 100, iteration)
    
    # Calculate approximate bandwidth if we know the message size (simplified)
    # This assumes we know the size of the model parameters
    # param_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)  # Assuming float32
    # bandwidth_mb_s = param_size_mb / (comm_time + 1e-6)
    # tboard_writer.add_scalar("Distributed/estimated_bandwidth_mb_s", bandwidth_mb_s, iteration)

def train(params, args, local_rank, world_rank, world_size):
    # set device and benchmark mode
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:%d'%local_rank)

    # initialize gpu monitoring
    nvml_available = init_gpu_monitoring()
    
    # log basic system information
    log_system_info(params, world_rank, local_rank, args.tboard_writer if world_rank == 0 else None)

    # get data loader
    logging.info('rank %d, begin data loader init'%world_rank)
    train_data_loader, train_dataset, train_sampler = get_data_loader_distributed(params, params.train_data_path, params.distributed, train=True)
    val_data_loader, valid_dataset = get_data_loader_distributed(params, params.valid_data_path, params.distributed, train=False)
    logging.info('rank %d, data loader initialized'%(world_rank))

    # create model
    model = vit.ViT(params).to(device)

    if params.enable_jit:
        model = torch.compile(model)
    
    if params.amp_dtype == torch.float16: 
        scaler = GradScaler('cuda')
    if params.distributed and not args.noddp:
        if args.disable_broadcast_buffers: 
            model = DistributedDataParallel(model, device_ids=[local_rank],
                                            bucket_cap_mb=args.bucket_cap_mb,
                                            broadcast_buffers=False,
                                            gradient_as_bucket_view=True)
        else:
            model = DistributedDataParallel(model, device_ids=[local_rank],
                                            bucket_cap_mb=args.bucket_cap_mb)

    if params.enable_fused:
        optimizer = optim.Adam(model.parameters(), lr = params.lr, fused=True, betas=(0.9, 0.95))
    else:
        optimizer = optim.Adam(model.parameters(), lr = params.lr,  betas=(0.9, 0.95))

    if world_rank == 0:
        logging.info(model)

    iters = 0
    startEpoch = 0

    if params.lr_schedule == 'cosine':
        if params.warmup > 0:
            lr_scale = lambda x: min((x+1)/params.warmup, 0.5*(1 + np.cos(np.pi*x/params.num_iters)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.num_iters)
    else:
        scheduler = None

    # select loss function
    if params.enable_jit:
        loss_func = l2_loss_opt
    else:
        loss_func = l2_loss

    if world_rank==0: 
        logging.info("Starting Training Loop...")

    # Log initial loss on train and validation to tensorboard
    with torch.no_grad():
        inp, tar = map(lambda x: x.to(device), next(iter(train_data_loader)))
        gen = model(inp)
        tr_loss = loss_func(gen, tar)
        inp, tar = map(lambda x: x.to(device), next(iter(val_data_loader)))
        gen = model(inp)
        val_loss = loss_func(gen, tar)
        val_rmse = weighted_rmse(gen, tar)
        if params.distributed:
            torch.distributed.all_reduce(tr_loss)
            torch.distributed.all_reduce(val_loss)
            torch.distributed.all_reduce(val_rmse)
        if world_rank==0:
            args.tboard_writer.add_scalar('Loss/train', tr_loss.item()/world_size, 0)
            args.tboard_writer.add_scalar('Loss/valid', val_loss.item()/world_size, 0)
            args.tboard_writer.add_scalar('RMSE(u10m)/valid', val_rmse.cpu().numpy()[0]/world_size, 0)

    params.num_epochs = params.num_iters//len(train_data_loader)
    
    iters = 0
    total_training_start = time.time()
    
    # Initialize cumulative metrics
    cumulative_metrics = {
        "training_time": 0,
        "data_loading_time": 0,
        "forward_time": 0,
        "backward_time": 0,
        "optimizer_time": 0,
        "communication_time": 0,
        "validation_time": 0,
        "total_samples_processed": 0
    }
    
    for epoch in range(startEpoch, startEpoch + params.num_epochs):
        epoch_start = time.time()
        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        if params.distributed and (train_sampler is not None):
            train_sampler.set_epoch(epoch)
        
        tr_loss = []
        # Reset timing variables for each epoch
        dat_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        optimizer_time = 0.0
        communication_time = 0.0
        tr_time = 0.0

        model.train()
        step_count = 0
        for i, data in enumerate(train_data_loader, 0):
            if world_rank == 0:
                if (epoch == 3 and i == 0):
                    torch.cuda.profiler.start()
                if (epoch == 3 and i == len(train_data_loader) - 1):
                    torch.cuda.profiler.stop()

            torch.cuda.nvtx.range_push(f"step {i}")
            iters += 1
            step_start = time.time()
            dat_start = time.time()
            torch.cuda.nvtx.range_push(f"data copy in {i}")

            inp, tar = map(lambda x: x.to(device), data)
            torch.cuda.nvtx.range_pop()  # copy in
            dat_end = time.time()
            dat_time += dat_end - dat_start

            tr_start = time.time()
            b_size = inp.size(0)
            cumulative_metrics["total_samples_processed"] += b_size * world_size
            
            optimizer.zero_grad()

            # Forward pass timing
            torch.cuda.nvtx.range_push(f"forward")
            forward_start = time.time()
            with autocast(device_type='cuda', enabled=params.amp_enabled, dtype=params.amp_dtype):
                gen = model(inp)
                loss = loss_func(gen, tar)
            torch.cuda.synchronize()
            forward_end = time.time()
            forward_time += forward_end - forward_start
            torch.cuda.nvtx.range_pop()  # forward

            # Backward pass timing
            torch.cuda.nvtx.range_push(f"backward")
            backward_start = time.time()
            if params.amp_dtype == torch.float16: 
                scaler.scale(loss).backward()
            else:
                loss.backward()
            torch.cuda.synchronize()
            backward_end = time.time()
            backward_time += backward_end - backward_start
            torch.cuda.nvtx.range_pop()  # backward

            # Optimizer step timing
            torch.cuda.nvtx.range_push(f"optimizer")
            optimizer_start = time.time()
            if params.amp_dtype == torch.float16: 
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            torch.cuda.synchronize()
            optimizer_end = time.time()
            optimizer_time += optimizer_end - optimizer_start
            torch.cuda.nvtx.range_pop()  # optimizer

            # Communication overhead timing (for distributed training)
            comm_start = time.time()
            if params.distributed:
                torch.distributed.all_reduce(loss)
                torch.cuda.synchronize()
            comm_end = time.time()
            communication_time += comm_end - comm_start
            
            tr_loss.append(loss.item()/world_size)

            torch.cuda.nvtx.range_pop()  # step
            
            # lr step
            scheduler.step()

            tr_end = time.time()
            tr_time += tr_end - tr_start
            step_time = tr_end - step_start
            step_count += 1
            
            # Periodically log detailed step information (every 10 steps or user-defined interval)
            log_frequency = getattr(params, 'log_frequency', 10)
            if i % log_frequency == 0:
                if world_rank == 0:
                    # Calculate step time parts
                    step_time = tr_end - step_start
                    data_time = dat_end - dat_start
                    compute_time = tr_end - tr_start
                    
                    logging.info(f'Epoch {epoch+1}, Step {i}: loss={loss.item()/world_size:.6f}, '
                                f'step_time={step_time:.4f}s, '
                                f'data_time={data_time:.4f}s, '
                                f'compute_time={compute_time:.4f}s')
                    
                    # Log train loss per step for more frequent updates
                    args.tboard_writer.add_scalar('Loss/train_step', loss.item()/world_size, iters)
                    
                    # Log GPU metrics
                    if nvml_available:
                        gpu_metrics = get_gpu_metrics(local_rank)
                        if gpu_metrics:
                            # Log to console
                            logging.info(f"GPU metrics: util={gpu_metrics['gpu_utilization']}%, "
                                        f"mem={gpu_metrics['memory_utilization']:.1f}%, "
                                        f"temp={gpu_metrics['temperature']}°C, "
                                        f"power={gpu_metrics['power_usage']:.2f}W")
                            
                            # Log to TensorBoard
                            log_gpu_metrics(gpu_metrics, epoch, iters, args.tboard_writer)
                    
                    # Log CPU metrics
                    log_cpu_metrics(epoch, iters, args.tboard_writer)
                    
                    # Log batch timing details to TensorBoard
                    args.tboard_writer.add_scalar('Time/step_total', step_time, iters)
                    args.tboard_writer.add_scalar('Time/data_loading', dat_end - dat_start, iters)
                    args.tboard_writer.add_scalar('Time/forward', forward_end - forward_start, iters)
                    args.tboard_writer.add_scalar('Time/backward', backward_end - backward_start, iters)
                    args.tboard_writer.add_scalar('Time/optimizer', optimizer_end - optimizer_start, iters)
                    
                    if params.distributed:
                        args.tboard_writer.add_scalar('Time/communication', comm_end - comm_start, iters)
                        log_distributed_metrics(comm_start, comm_end, iters, world_size, args.tboard_writer)
            
            # More frequent simple progress logging
            if i % 50 == 0:
                logging.info(f'Rank {world_rank}, Epoch {epoch+1}, Step {i} completed')
            
        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start

        # Update cumulative metrics
        cumulative_metrics["training_time"] += tr_time
        cumulative_metrics["data_loading_time"] += dat_time
        cumulative_metrics["forward_time"] += forward_time
        cumulative_metrics["backward_time"] += backward_time
        cumulative_metrics["optimizer_time"] += optimizer_time
        cumulative_metrics["communication_time"] += communication_time

        if world_rank==0:
            # Calculate performance metrics
            iters_per_sec = step_count / epoch_duration
            samples_per_sec = params["global_batch_size"] * iters_per_sec
            
            # Calculate summary statistics for time breakdown
            total_step_time = dat_time + forward_time + backward_time + optimizer_time + communication_time
            
            # Avoid division by zero
            if total_step_time > 0:
                data_percent = 100 * dat_time / total_step_time
                forward_percent = 100 * forward_time / total_step_time
                backward_percent = 100 * backward_time / total_step_time
                optimizer_percent = 100 * optimizer_time / total_step_time
                communication_percent = 100 * communication_time / total_step_time
            else:
                data_percent = forward_percent = backward_percent = optimizer_percent = communication_percent = 0
                
            # Log detailed epoch summary
            logging.info(f"======== Epoch {epoch+1} Summary ========")
            logging.info(f'Time for epoch {epoch+1}: {epoch_duration:.2f} sec')
            logging.info(f'  Avg train loss: {np.mean(tr_loss):.6f}')
            logging.info(f'  Performance: {iters_per_sec:.2f} iter/sec, {samples_per_sec:.2f} samples/sec')
            logging.info(f'  Time breakdown: data={data_percent:.1f}%, forward={forward_percent:.1f}%, '
                        f'backward={backward_percent:.1f}%, optimizer={optimizer_percent:.1f}%, '
                        f'communication={communication_percent:.1f}%')
            
            # TensorBoard logging for epoch metrics
            args.tboard_writer.add_scalar('Loss/train', np.mean(tr_loss), iters)
            args.tboard_writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], iters)
            args.tboard_writer.add_scalar('Performance/iters_per_sec', iters_per_sec, iters)
            args.tboard_writer.add_scalar('Performance/samples_per_sec', samples_per_sec, iters)
            args.tboard_writer.add_scalar('Performance/epoch_time', epoch_duration, epoch)
            
            # Time breakdown visualization
            args.tboard_writer.add_scalar('Time Breakdown/data_loading_percent', data_percent, epoch)
            args.tboard_writer.add_scalar('Time Breakdown/forward_percent', forward_percent, epoch)
            args.tboard_writer.add_scalar('Time Breakdown/backward_percent', backward_percent, epoch)
            args.tboard_writer.add_scalar('Time Breakdown/optimizer_percent', optimizer_percent, epoch)
            args.tboard_writer.add_scalar('Time Breakdown/communication_percent', communication_percent, epoch)
            
            # Create training visualizations
            fig = generate_images([inp, tar, gen])
            args.tboard_writer.add_figure('Visualization, t2m', fig, iters, close=True)
            
            # Log scaling efficiency (if world_size > 1)
            if world_size > 1:
                # This is an approximation assuming linear scaling with GPUs
                # In a real implementation, you'd have baseline single-GPU numbers to compare
                theoretical_max_samples_per_sec = samples_per_sec / world_size * world_size  # Ideal linear scaling
                scaling_efficiency = (samples_per_sec / theoretical_max_samples_per_sec) * 100
                args.tboard_writer.add_scalar('Scaling/efficiency_percent', scaling_efficiency, epoch)
                args.tboard_writer.add_scalar('Scaling/samples_per_gpu', samples_per_sec / world_size, epoch)

        # Validation
        val_start = time.time()
        val_loss = torch.zeros(1, device=device)
        val_rmse = torch.zeros((params.n_out_channels), dtype=torch.float32, device=device)
        valid_steps = 0
        model.eval()

        with torch.inference_mode():
            with torch.no_grad():
                for i, data in enumerate(val_data_loader, 0):
                    with autocast(device_type='cuda', enabled=params.amp_enabled, dtype=params.amp_dtype):
                        inp, tar = map(lambda x: x.to(device), data)
                        gen = model(inp)
                        val_loss += loss_func(gen, tar)
                        val_rmse += weighted_rmse(gen, tar)
                    valid_steps += 1

                if params.distributed:
                    val_comm_start = time.time()
                    torch.distributed.all_reduce(val_loss)
                    val_loss /= world_size
                    torch.distributed.all_reduce(val_rmse)
                    val_rmse /= world_size
                    val_comm_end = time.time()
                    val_comm_time = val_comm_end - val_comm_start
                    if world_rank == 0:
                        args.tboard_writer.add_scalar('Time/validation_communication', val_comm_time, epoch)

        val_rmse /= valid_steps  # Avg validation rmse
        val_loss /= valid_steps
        val_end = time.time()
        val_time = val_end - val_start
        cumulative_metrics["validation_time"] += val_time
        
        if world_rank==0:
            logging.info(f'  Validation loss: {val_loss.item():.6f}')
            logging.info(f'  Validation RMSE: {val_rmse.cpu().numpy()[0]:.6f}')
            logging.info(f'  Validation time: {val_time:.2f} sec')
            
            args.tboard_writer.add_scalar('Loss/valid', val_loss, iters)
            args.tboard_writer.add_scalar('RMSE(u10m)/valid', val_rmse.cpu().numpy()[0], iters)
            args.tboard_writer.add_scalar('Time/validation_epoch', val_time, epoch)
            
    # Log final overall performance metrics
    total_training_end = time.time()
    total_time = total_training_end - total_training_start
    total_samples = cumulative_metrics["total_samples_processed"]
    total_steps = sum(min(params.steps_per_epoch, len(train_data_loader)) for _ in range(params.num_epochs))
    
    if total_time > 0:
        overall_iters_per_sec = total_steps / total_time
        overall_samples_per_sec = total_samples / total_time
    else:
        overall_iters_per_sec = overall_samples_per_sec = 0
    total_training_time = total_time
    
    # Add additional final summary metrics to TensorBoard
    args.tboard_writer.add_scalar('Final_Summary/total_training_time', total_training_time, 0)
    args.tboard_writer.flush()
        
    if world_rank == 0:
        # Calculate overall time breakdown
        total_active_time = (cumulative_metrics["data_loading_time"] + 
                          cumulative_metrics["forward_time"] + 
                          cumulative_metrics["backward_time"] + 
                          cumulative_metrics["optimizer_time"] + 
                          cumulative_metrics["communication_time"] +
                          cumulative_metrics["validation_time"])
        
        overhead_time = total_training_time - total_active_time
        
        # Log overall training summary
        logging.info("\n" + "="*50)
        logging.info("TRAINING COMPLETE - SUMMARY")
        logging.info("="*50)
        logging.info(f"Total training time: {total_training_time:.2f} seconds")
        logging.info(f"Total epochs: {params.num_epochs}")
        logging.info(f"Total iterations: {total_steps}")
        logging.info(f"Total samples processed: {cumulative_metrics['total_samples_processed']:,}")
        logging.info(f"Average samples/second: {overall_samples_per_sec:.2f}")
        logging.info(f"Average iterations/second: {overall_iters_per_sec:.2f}")
        
        # Time breakdown
        logging.info("\nTime Breakdown:")
        logging.info(f"  Data loading:   {cumulative_metrics['data_loading_time']:.2f}s ({100*cumulative_metrics['data_loading_time']/total_active_time:.1f}%)")
        logging.info(f"  Forward pass:   {cumulative_metrics['forward_time']:.2f}s ({100*cumulative_metrics['forward_time']/total_active_time:.1f}%)")
        logging.info(f"  Backward pass:  {cumulative_metrics['backward_time']:.2f}s ({100*cumulative_metrics['backward_time']/total_active_time:.1f}%)")
        logging.info(f"  Optimizer step: {cumulative_metrics['optimizer_time']:.2f}s ({100*cumulative_metrics['optimizer_time']/total_active_time:.1f}%)")
        logging.info(f"  Communication:  {cumulative_metrics['communication_time']:.2f}s ({100*cumulative_metrics['communication_time']/total_active_time:.1f}%)")
        logging.info(f"  Validation:     {cumulative_metrics['validation_time']:.2f}s ({100*cumulative_metrics['validation_time']/total_active_time:.1f}%)")
        logging.info(f"  Other overhead: {overhead_time:.2f}s ({100*overhead_time/total_training_time:.1f}%)")
        
        # Add summary to TensorBoard as text
        summary_text = f"""
## Training Summary
- **Total Time**: {total_training_time:.2f} seconds
- **Epochs**: {params.num_epochs}
- **Total Iterations**: {total_steps}
- **Samples Processed**: {cumulative_metrics['total_samples_processed']:,}
- **Avg Throughput**: {overall_samples_per_sec:.2f} samples/sec
- **Avg Iteration Rate**: {overall_iters_per_sec:.2f} iter/sec

## Time Breakdown
- Data loading: {cumulative_metrics['data_loading_time']:.2f}s ({100*cumulative_metrics['data_loading_time']/total_active_time:.1f}%)
- Forward pass: {cumulative_metrics['forward_time']:.2f}s ({100*cumulative_metrics['forward_time']/total_active_time:.1f}%)
- Backward pass: {cumulative_metrics['backward_time']:.2f}s ({100*cumulative_metrics['backward_time']/total_active_time:.1f}%)
- Optimizer step: {cumulative_metrics['optimizer_time']:.2f}s ({100*cumulative_metrics['optimizer_time']/total_active_time:.1f}%)
- Communication: {cumulative_metrics['communication_time']:.2f}s ({100*cumulative_metrics['communication_time']/total_active_time:.1f}%)
- Validation: {cumulative_metrics['validation_time']:.2f}s ({100*cumulative_metrics['validation_time']/total_active_time:.1f}%)
- Other overhead: {overhead_time:.2f}s ({100*overhead_time/total_training_time:.1f}%)
"""
        args.tboard_writer.add_text("Training Summary", summary_text, 0)
        
        # Create pie chart of time breakdown
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = ['Data Loading', 'Forward Pass', 'Backward Pass', 'Optimizer', 'Communication', 'Validation', 'Overhead']
        
        # Ensure all times are non-negative for the pie chart
        sizes = [
            max(0, cumulative_metrics['data_loading_time']),
            max(0, cumulative_metrics['forward_time']),
            max(0, cumulative_metrics['backward_time']),
            max(0, cumulative_metrics['optimizer_time']),
            max(0, cumulative_metrics['communication_time']),
            max(0, cumulative_metrics['validation_time']),
            max(0, overhead_time)
        ]
        
        args.tboard_writer.flush()

    if params.distributed:
        torch.distributed.barrier()
    logging.info('DONE ---- rank %d'%world_rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str, help='tag for indexing the current experiment')
    parser.add_argument("--yaml_config", default='./config/ViT.yaml', type=str, help='path to yaml file containing training configs')
    parser.add_argument("--config", default='base', type=str, help='name of desired config in yaml file')
    parser.add_argument("--amp_mode", default='none', type=str, choices=['none', 'fp16', 'bf16'], help='select automatic mixed precision mode')  
    parser.add_argument("--enable_fused", action='store_true', help='enable fused Adam optimizer')
    parser.add_argument("--enable_jit", action='store_true', help='enable JIT compilation')
    parser.add_argument("--local_batch_size", default=None, type=int, help='local batchsize (manually override global_batch_size config setting)')
    parser.add_argument("--num_iters", default=None, type=int, help='number of iters to run')
    parser.add_argument("--num_data_workers", default=None, type=int, help='number of data workers for data loader')
    parser.add_argument("--data_loader_config", default=None, type=str, choices=['pytorch', 'dali'], help="dataloader configuration. choices: 'pytorch', 'dali'")
    parser.add_argument("--bucket_cap_mb", default=25, type=int, help='max message bucket size in mb')
    parser.add_argument("--disable_broadcast_buffers", action='store_true', help='disable syncing broadcasting buffers')
    parser.add_argument("--noddp", action='store_true', help='disable DDP communication')
    parser.add_argument("--log_frequency", default=10, type=int, help='frequency for detailed logging (steps)')
    
    # debugging
    parser.add_argument("--debug_distributed", action='store_true', help='enable debugging for NCCL and Pytorch Distributed training')
    
    args = parser.parse_args()
 
    run_num = args.run_num

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    
    if args.debug_distributed:
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
        os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
        
        nccl_dir = os.path.join(os.getcwd(), 'logs', 'nccl') # setting dir to cwd/logs/nccl
        os.makedirs(nccl_dir, exist_ok=True) # create nccl logs dir if it doesnt exist

        # set NCCL logging to file
        nccl_file = os.path.join(nccl_dir, 'nccl.log')
        os.environ['NCCL_LOG_FILE'] = nccl_file
        logging.info(f"NCCL logs will be saved to {nccl_file}")
        
        logging.info("Debugging enabled for NCCL and Pytorch Distributed training")
        
        logging.info("Environment Variables Set:")
        for var in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_RANK", "TORCH_DISTRIBUTED_DEBUG", "NCCL_DEBUG"]:
            logging.info(f"{var}: {os.environ.get(var, 'Not Set')}")
        logging.info(f"NCCL Version: {torch.cuda.nccl.version()}")

    # Update config with modified args
    # set up amp
    if args.amp_mode != 'none':
        params.update({"amp_mode": args.amp_mode})
    amp_dtype = torch.float32
    if params.amp_mode == "fp16":
        amp_dtype = torch.float16
    elif params.amp_mode == "bf16":
        amp_dtype = torch.bfloat16    
    params.update({"amp_enabled": amp_dtype is not torch.float32,
                    "amp_dtype" : amp_dtype, 
                    "enable_fused" : args.enable_fused,
                    "enable_jit" : args.enable_jit
                    })

    if args.data_loader_config:
        params.update({"data_loader_config" : args.data_loader_config})
    
    if args.num_iters:
        params.update({"num_iters" : args.num_iters})

    if args.num_data_workers:
        params.update({"num_data_workers" : args.num_data_workers})
        
    params.update({"log_frequency": args.log_frequency})

    params.distributed = False
    if 'WORLD_SIZE' in os.environ:
        params.distributed = int(os.environ['WORLD_SIZE']) > 1
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = 1

    world_rank = 0
    local_rank = 0
    if params.distributed:
        torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
        world_rank = torch.distributed.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])

    if args.local_batch_size:
        # Manually override batch size
        params.local_batch_size = args.local_batch_size
        params.update({"global_batch_size" : world_size*args.local_batch_size})
    else:
        # Compute local batch size based on number of ranks
        params.local_batch_size = params.global_batch_size//world_size

    # for dali data loader, set the actual number of data shards and id
    params.data_num_shards = world_size
    params.data_shard_id = world_rank

    # Set up directory
    baseDir = params.expdir
    expDir = os.path.join(baseDir, args.config + '/%dGPU/'%(world_size) + str(run_num) + '/')
    if world_rank==0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        params.log()
        args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, 'logs/'))

    params.experiment_dir = os.path.abspath(expDir)

    train(params, args, local_rank, world_rank, world_size)

    if params.distributed:
        torch.distributed.barrier()
    logging.info('DONE ---- rank %d'%world_rank)
