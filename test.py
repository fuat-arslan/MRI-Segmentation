import torch
import torch.nn as nn
import os
from utils.args import get_main_args

import warnings
warnings.filterwarnings("ignore")

from data_loading.data_module import DataModule

from net.loss import compute_loss, LossBraTS
from net.wrappers import tester
from net.networks import U_Net, AttU_Net
from net.utils import add_config


def test(args, model_path):
    data_module = DataModule(args)
    data_module.setup()

    #Data loaders
    # test_dataloader of DALI didnt work with on our servers due to CUDA compatibility problems.
    # Therefore, at the beginning of the project, we split the test data to another folder
    # (which is given in the config file seperately). so, there is no data leakage. This only 
    # loads the test data using train_dataloader function. Folders are completely different.
    test_loader = data_module.train_dataloader() # do not use in this way

    #Extract model parameters from args
    device = torch.device(args.model_params.device)
    deep_supervision=args.model_params.deep_supervision
    # activation = generate_activation_function(args.model_params.activation)
    activation = nn.ReLU() if args.model_params.activation == "relu" else nn.LeakyReLU(0.02, inplace=False)

    #Get model from args
    model_class = U_Net if args.model_params.model_name == "unet" else AttU_Net
    model = model_class(args.model_params.dimension,args.model_params.in_channels,
                args.model_params.out_channels,
                    args.model_params.kernels,
                    args.model_params.instance_norm,
                    activation,
                    args.model_params.deep_supervision,
                    args.model_params.deep_supervision_head,
                    args.model_params.n_bottleneck
                ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    criterion = LossBraTS(focal=args.model_params.focal)
    output_dict = tester(test_loader, model, criterion=criterion, device=device, dim3d=True if args.model_params.dimension == 3 else False, deep_supervision=deep_supervision, save_path=args.model_params.experiment_dir)
    save_dir = args.model_params.experiment_dir
    torch.save(output_dict, os.path.join(save_dir, "test_results.pth"))
    return output_dict

def main():
    args = get_main_args()
    args = add_config(args.model_config, args)
    args.data = args.model_params.test_data
    #get the file from experimen_dir starts with best_model but rest can be different
    model_path = os.path.join(args.model_params.experiment_dir, 
                              [file for file in os.listdir(args.model_params.experiment_dir) if file.startswith("best_model")][-1])
    test(args, model_path)

if __name__ == "__main__":
    main()
    