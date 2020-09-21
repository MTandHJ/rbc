




import torch
import torchvision
import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
import foolbox as fb
import argparse
from collections.abc import Iterable
from utils.dict2object import Config
from utils.dealinfo import load, accuracy
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
parser.add_argument("file", type=str)
parser.add_argument(
    "--active", type=str,
    choices=(
        "relu", "ReLU",
        "tanh", "Tanh",
        "prelu", "PReLU",
        "leakyrelu", "LeakyReLU"
    )
)
parser.add_argument("--epsilon", type=float, default=8/255)
parser.add_argument("--epsilon_times", type=int, default=1)
parser.add_argument("--stepsize", type=float, default=0.1, 
                    help="pgd:rel_stepsize, cwl2:step_size, deepfool:overshoot, bb:lr")
parser.add_argument("--steps", type=int, default=20)
parser.add_argument('--bounds', default=(0, 1), help="images' bounds")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=3)
parser.add_argument("--pin_memory", action="store_false", default=True),
parser.add_argument("--shuffle", action="store_true", default=False)
opts = parser.parse_args()


def load_cfg():
    cfg = Config()
    load_models(cfg, opts)
    load_testset(cfg, opts)
    filename = cfg.info_path + "/paras.pt"
    load(cfg.model, filename)
    return cfg


def attack(
    model, device, attacker,
    testloader, test_size
    ):
    fmodel = fb.PyTorchModel(
        model, 
        bounds=attacker.bounds,
        preprocessing=attacker.preprocessing,
        device=device
    )
    assert isinstance(attacker.epsilon, Iterable)
    running_accuracy = 0.
    running_success = 0.
    running_distance = 0.
    for i, data in enumerate(testloader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        accuracy_count = accuracy(fmodel, imgs, labels)
        raw, clipped, is_adv = attacker.attack(fmodel, imgs, labels, epsilons=attacker.epsilon)
        success_count = is_adv.sum(dim=1)
        distance = fb.distances.linf(imgs, clipped[0])[is_adv[0]].sum().item()

        running_accuracy += accuracy_count
        running_success += success_count
        running_distance += distance

    running_accuracy = running_accuracy / test_size
    running_distance = running_distance / running_success[0]
    running_success = running_success / test_size 
    return running_accuracy, running_success.tolist(), running_distance


def main(cfg):
    if opts.epsilon_times is 1:
        opts.epsilon = [opts.epsilon]
        attacker = Config()
        load_attacks(attacker, opts)
        results = attack(attacker=attacker, **cfg)
        s = "Accuracy: {0[0]:<.6f}, Success: {0[1][0]:<.6f}," \
             " Distance: {0[2]:<.6f}".format(results)
        writter.add_text("Attack", s)
    else:
        epsilons = (torch.linspace(8, 255, opts.epsilon_times) / 255.).tolist()
        opts.epsilon = epsilons
        attacker = Config()
        load_attacks(attacker, opts)
        accuracy_rate, success_rates, distance = attack(attacker=attacker, **cfg)
        writter.add_text("Test/Accuracy", str(accuracy_rate))
        writter.add_text("Attack/Distance", str(distance))
        for i, epsilon in enumerate(epsilons):
            writter.add_scalar("Attack/Success", success_rates[i], i)


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    cfg = load_cfg()

    # set writter
    writter = SummaryWriter(log_dir=cfg.log_path, filename_suffix="attack")

    # attack
    main(cfg)
    writter.close()



















