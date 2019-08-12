


from ..Utils.other_utils import *

def main(config, save_name, model_config, model, model_tokenizer):

    if not os.path.exists(config.output_dir + save_name):
        os.makedirs(config.output_dir + save_name)

    if not os.path.exists(config.cache_dir + save_name):
        os.makedirs(config.cache_dir + save_name)

    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
    device, n_gpu = get_device(gpu_ids[0])
    if n_gpu > 1:
        n_gpu = len(gpu_ids)
    
    set_seed(config.seed)




    
