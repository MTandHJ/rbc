






import torch
import torchvision
import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
import foolbox as fb
import argparse
from utils.dict2object import Config
from utils.dealinfo import save_checkpoint, load_checkpoint, \
                            mkdirs, valid_attack
from utils.loadopts import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "model", type=str,
    choices=(
        "mnist", "mnistrbc", 
        "cifar10", "cifar10rbc", 
        "wideresnet34", "wideresnet34rbc",
        "wideresnet34-100", "wideresnet34rbc-100"
    )
)
parser.add_argument(
    "attack", type=str,
    choices=("pgd", "cwl2", "deepfool", "bb")
)
parser.add_argument(
    "active", type=str,
    choices=(
        "relu", "ReLU",
        "tanh", "Tanh",
        "prelu", "PReLU",
        "leakyrelu", "LeakyReLU"
    )
)
parser.add_argument(
    "--loss_func", type=str,
    choices=(
        "rbc", "mmc",
        "rbcmmc"
    ),
    default="rbc"
)
parser.add_argument("--scale", type=float, default=0.1, help="to adjust the weight")
parser.add_argument("--leverage", type=float, default=100, help="to adjust the rbc")
parser.add_argument("--epsilon", type=float, default=8/255)
parser.add_argument("--stepsize", type=float, default=0.2, 
                    help="pgd:rel_stepsize, cwl2:step_size, deepfool:overshoot, bb:lr")
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--valid_stepsize", type=float, default=0.1, 
                    help="pgd:rel_stepsize, cwl2:step_size, deepfool:overshoot, bb:lr")
parser.add_argument("--valid_steps", type=int, default=20)
parser.add_argument('--bounds', default=(0, 1), help="images' bounds")
parser.add_argument("--epoches", type=int, default=400)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=3)
parser.add_argument("--pin_memory", action="store_false", default=True),
parser.add_argument("--shuffle", action="store_false", default=True)
parser.add_argument("--optim", type=str, choices=("adam", "sgd"), default="adam")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.9, help="if SGD used")
parser.add_argument("--betas", default=(0.9, 0.999), help="if Adam used")
parser.add_argument("--is_load_checkpoint", action="store_true", default=False)
parser.add_argument("--description", type=str, default="adv")
opts = parser.parse_args()


VALID_EPOCHES = 20


def load_cfg():
    cfg = Config()
    load_models(cfg, opts)
    load_trainset(cfg, opts)
    load_optims(cfg, opts)
    load_loss_func(cfg, opts)
    if opts.is_load_checkpoint:
        cfg['start_epoch'] = load_checkpoint(
            cfg.info_path, cfg.model, 
            cfg.optim, cfg.learning_policy
        )
    else:
        cfg['start_epoch'] = -1
    # attacker
    attacker = Config()
    load_attacks(attacker, opts)
    # valider
    valider = Config()
    load_valid(valider, opts)
    return cfg, attacker, valider


def train(
    model, device, attacker, valider, path,
    trainloader, train_size, normalizer,
    start_epoch, epoches, optim, loss_func,
    learning_policy=None
    ):
    fmodel = fb.PyTorchModel(
        model, 
        bounds=attacker.bounds,
        preprocessing=attacker.preprocessing,
        device=device
    )
    for epoch in range(start_epoch+1, epoches+1):
        # valid, Acc.(Adv.) == 1 - Success
        if epoch % VALID_EPOCHES == 0:
            save_checkpoint(path, model, optim, learning_policy, epoch)
            model.eval()
            valid_accuracy, valid_success, valid_distance \
                = valid_attack(model, device, valider)
            train_accuracy, train_success, train_distance \
                = valid_attack(model, device, valider, dataloader=trainloader, data_size=train_size)
            model.train()
            writter.add_scalars("Accuracy", {"valid":valid_accuracy, "train":train_accuracy}, epoch)
            writter.add_scalars("Success", {"valid":valid_success, "train":train_success}, epoch)
            writter.add_scalars("Distance", {"valid":valid_distance, "train":train_distance}, epoch)

        for i, data in enumerate(trainloader):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)

            model.eval()
            raw, clipped, is_adv = attacker.attack(fmodel, imgs, labels, epsilons=attacker.epsilon)
            model.train()
            
            # use clipped to train
            outs, features = model(normalizer(clipped))
            loss = loss_func(
                model=model, 
                outs=outs, 
                features=features,
                labels=labels
            )

            optim.zero_grad()
            loss.backward()
            optim.step()

        # rescale learning rate
        learning_policy.step()


        


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    cfg, attacker, valider = load_cfg()

    # make path
    mkdirs(cfg.log_path, cfg.info_path)
    
    # set writter
    writter = SummaryWriter(log_dir=cfg.log_path, filename_suffix="adv")

    # training
    train(attacker=attacker, valider=valider, path=cfg.info_path, **cfg)
    torch.save(cfg.model.state_dict(), cfg.info_path+"/paras.pt")
    writter.close()

    








    






















