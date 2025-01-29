import os
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetComputeRunningProcesses, nvmlShutdown, nvmlDeviceGetCount

from AP_Bots.models import LLM

def get_available_GPUmem():

    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
    except Exception as e:
        print(e)
        return 0

    all_free_mem = 0
    current_process_mem = 0
    current_pid = os.getpid()

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        all_free_mem += mem_info.free / 1024**3

        processes = nvmlDeviceGetComputeRunningProcesses(handle)
        for process in processes:
            if process.pid == current_pid:
                current_process_mem += process.usedGpuMemory / 1024**3

    nvmlShutdown()
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