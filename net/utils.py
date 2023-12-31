import yaml
from argparse import Namespace
from torch import nn

def read_config(config_file):
    with open(config_file, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def add_config(config_path, args):
    config = read_config(config_path)
    # add configs to args.model_params
    args.model_params = Namespace(**config)
    args.batch_size = args.model_params.batch_size
    args.data = args.model_params.data
    args.brats = args.model_params.brats
    args.fold = args.model_params.fold
    args.gpus = args.model_params.gpus
    args.task = args.model_params.task
    args.num_workers = args.model_params.num_workers
    args.nfolds = args.model_params.nfolds
    return args
    

def get_activation_function(activation_name, *args, **kwargs):
    try:
        activation_function = getattr(nn, activation_name)(*args, **kwargs)
        if isinstance(activation_function, nn.Module):
            return activation_function
        else:
            raise ValueError(f"Invalid activation function: {activation_name}")
    except AttributeError:
        raise ValueError(f"Activation function '{activation_name}' not found in torch.nn module.")

def print_activation_functions():
    activation_functions = [name for name in dir(nn) if isinstance(getattr(nn, name), nn.Module) and "activation" in name.lower()]
    print("Available activation functions in torch.nn:")
    print(activation_functions)
