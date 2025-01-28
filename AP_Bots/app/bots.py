import GPUtil

from AP_Bots.models import LLM


def get_available_GPUmem():

    gpus = GPUtil.getGPUs()
    all_free_mem = 0
    for gpu in gpus:
        all_free_mem += gpu.memoryFree / 1024
    
    return all_free_mem

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