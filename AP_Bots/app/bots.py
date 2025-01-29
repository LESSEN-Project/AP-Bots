from AP_Bots.models import LLM

import GPUtil
import pynvml
import os

def get_available_GPUmem():
    
    gpus = GPUtil.getGPUs()

    if not gpus:
        return 0

    pynvml.nvmlInit()
    all_free_mem = 0
    current_process_mem = 0
    
    for gpu in gpus:
        # Add available free memory
        all_free_mem += gpu.memoryFree / 1024
        
        # Get handle for the GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.id)
        
        # Get current process memory usage
        current_pid = os.getpid()
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for process in processes:
            if process.pid == current_pid:
                current_process_mem += process.usedGpuMemory / (1024**3)  # Convert to GB

    # Shutdown NVML
    pynvml.nvmlShutdown()
    
    # Return the sum of both rounded
    return round(all_free_mem + current_process_mem, 1)

def get_llm_config():
    return LLM.get_cfg()

def get_model_gpu_req():

    cfg = get_llm_config()
    return {model: int(cfg[model]["min_GPU_RAM"]) for model in cfg.sections() if model != "DEFAULT"}

def get_avail_bots():

    free_gpu_mem = get_available_GPUmem()
    model_gpu_req = get_model_gpu_req()
    available_models = [model for model in model_gpu_req.keys() if model_gpu_req[model] <= free_gpu_mem]

    return available_models, model_gpu_req, free_gpu_mem