





import torch
import torchvision
import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
import foolbox as fb
import argparse
from utils.dict2object import Config
from utils.dealinfo import load, accuracy
from utils.loadopts import *



parser = argparse.ArgumentParser()
parser.add_argument(
    "model", type=str,
    choices=(
        "mnist", "mnistrbc", 
        "cifar10", "cifar10rbc", 
        "wideresnet34", "wideresnet34rbc"
    )
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
parser.add_argument(
    "--attack", type=str,
    choices=("pgd",),
    default="pgd"
)
parser.add_argument("--epsilon", type=float, default=8/255)
parser.add_argument("--epsilon_times", type=int, default=1)
parser.add_argument("--stepsize", type=float, default=0.1, 
                    help="if the attack is pgd, the stepsize is rel_stepsize")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument('--bounds', default=(0, 1), help="images' bounds")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=3)
parser.add_argument("--pin_memory", action="store_false", default=True),
parser.add_argument("--shuffle", action="store_true", default=False)
parser.add_argument("--num_classes", type=int, default=10)
opts = parser.parse_args()


def load_cfg():
    cfg = Config()
    load_models(cfg, opts)
    load_testset(cfg, opts)
    filename = cfg.info_path + "/paras.pt"
    load(cfg.model, filename)
    cfg.model.addition = True
    return cfg

def load_attacker():

    return Ada_pgd(opts.bounds, opts.epsilon, opts.stepsize, opts.steps)


class Ada_pgd:

    def __init__(
        self, bounds, epsilon, 
        rel_stepsize=0.1, steps=50, 
        num_classes=10
    ):
        self.bounds = bounds
        self.epsilon = epsilon
        self.stepsize = rel_stepsize * epsilon
        self.steps = steps
        self.num_classes = num_classes

    def get_targets(self, label):
        targets = list(range(self.num_classes))
        targets.remove(label)
        return targets

    def loss_fn(self, model, features, target):
        if hasattr(model, "fc"):
            centers = model.fc.weight
        else:
            centers = model.rbc.weight
        return (features - centers[[target]]).pow(2).sum() 

    def clip(self, img):
        return torch.clamp(img, self.bounds[0], self.bounds[1])

    def update(self, prime, adv, grad):
        temp = adv + self.stepsize * grad.sign()
        temp = torch.clamp(temp - prime, -self.epsilon, self.epsilon) + prime
        return self.clip(temp).clone().detach()

    def get_random_start(self, prime):
        perturbation = torch.rand_like(prime) * self.epsilon * 2 - self.epsilon
        return prime + perturbation

    def is_adv(self, out, label):
        if out[0].argmax() != label:
            return True
        else:
            return False

    def run(
        self, model, 
        input, label
    ):
        device = input.device
        targets = self.get_targets(label)

        for target in targets:
            adv_input = self.clip(self.get_random_start(input)).requires_grad_(True).to(device)
            for step in range(self.steps):
                out, features = model(adv_input)
                if self.is_adv(out, label):
                    return adv_input.detach().data, True
                loss = self.loss_fn(model, features, target)
                grad = torch.autograd.grad(loss, adv_input)[0]
                adv_input = self.update(input, adv_input, grad).requires_grad_(True).to(device)
        return adv_input.detach().data, False

    def __call__(self, model, inputs, labels):
        adv_results = []
        is_adv = []
        for i in range(len(inputs)):
            item = self.run(model, inputs[[i]], labels[i])
            adv_results.append(item[0][0])
            is_adv.append(item[1])
        return torch.stack(adv_results), torch.tensor(is_adv)


def attack(
    model, device, attacker,
    testloader, test_size
    ):
    running_success = 0.
    running_distance = 0.
    for i, data in enumerate(testloader):
        imgs, labels = data
        imgs = imgs.to(device)

        clipped, is_adv = attacker(model, imgs, labels)
        success_count = is_adv.sum().item()
        distance = fb.distances.linf(imgs, clipped)[is_adv].sum().item()

        running_success += success_count
        running_distance += distance

    running_distance = running_distance / running_success
    running_success = running_success / test_size 
    return running_success, running_distance

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    cfg = load_cfg()
    attacker = load_attacker()

    # set writter
    writter = SummaryWriter(log_dir=cfg.log_path, filename_suffix="attack")

    # attack
    results = attack(attacker=attacker, **cfg)
    s = "Success: {0[0]:<.6f}, Distance: {0[1]:<.6f}".format(results)
    writter.add_text("Attack", s)
    writter.close()























