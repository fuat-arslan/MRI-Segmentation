
import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils.args import get_main_args

import warnings
warnings.filterwarnings("ignore")

from data_loading.data_module import DataModule

from net.loss import compute_loss, LossBraTS
from net.wrappers import trainer
from net.networks import U_Net, AttU_Net
from net.utils import add_config

def train(args,model_path):
    data_module = DataModule(args)
    data_module.setup()

    #Data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    # test_loader = data_module.test_dataloader()

    #Extract model parameters from args
    device = torch.device(args.model_params.device)
    deep_supervision=args.model_params.deep_supervision
    # activation = generate_activation_function(args.model_params.activation)
    activation = nn.ReLU() if args.model_params.activation == "relu" else nn.LeakyReLU(0.02, inplace=False)

    #Get model from args
    model_class = U_Net if args.model_params.model_type == "unet" else AttU_Net
    print(args.model_params.model_type == "unet")
    model = model_class(args.model_params.dimension,args.model_params.in_channels,
                args.model_params.out_channels,
                    args.model_params.kernels,
                    args.model_params.instance_norm,
                    activation,
                    args.model_params.deep_supervision,
                    args.model_params.deep_supervision_head,
                    args.model_params.n_bottleneck
                ).to(device)


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params:,}")
    print(f"Model Filters: {args.model_params.kernels}, Number of BottleNecks: {args.model_params.n_bottleneck}, Model Name: {args.model_params.model_name}")
    num_epochs = args.model_params.epochs

    # criterion = DiceLoss()
    criterion = LossBraTS(focal=args.model_params.focal)
    optimizer = optim.NAdam(model.parameters(), lr=args.model_params.learning_rate, weight_decay=args.model_params.weight_decay)

    output_dict = trainer(num_epochs, train_loader, val_loader, model, optimizer, criterion,
                        dim3d=True if args.model_params.dimension == 3 else False, deep_supervision=deep_supervision, device=device,
                        best_model_path=model_path, amp=args.model_params.amp)

    #save output dict
    torch.save(output_dict, os.path.join(model_path, "output_dict.pt"))

def __main__():
    args = get_main_args()

    args = add_config(args.model_config, args)

    os.makedirs(args.model_params.results_dir, exist_ok=True)
    model_path = os.path.join(args.model_params.results_dir, args.model_params.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path,exist_ok=True)
    else:
        print("Model already exists")
        if args.model_params.overwrite:
            print("Overwriting model")
            os.system(f"rm -rf {model_path}")
            os.makedirs(model_path,exist_ok=True)
        else:
            # model_path = model_path + "_new"
            # os.makedirs(model_path,exist_ok=True)
            suffix = 1
            while os.path.exists(f"{model_path}{suffix}"):
                suffix += 1
            model_path = f"{model_path}{suffix}"
            os.makedirs(model_path, exist_ok=True)

    #copy config file to model directory
    os.system(f"cp {args.model_config} {model_path}")
    with open(os.path.join(model_path, os.path.basename(args.model_config)), "a") as f:
        f.write(f"\nexperiment_dir: {model_path}")
    train(args,model_path)

if __name__ == "__main__":
    __main__()