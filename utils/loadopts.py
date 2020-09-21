

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
import foolbox as fb
from typing import TypeVar
from .dict2object import Config
from .tools import gpu
from .loss import *
from .config import *



Argparse = TypeVar("Argparse")


class ModelNotDefineError(Exception): pass
class OptimNotDefineError(Exception): pass
class AttackNotIncludeError(Exception): pass
class ActiveNotIncludeError(Exception): pass
class LossNotDefineError(Exception): pass

# set the active function
def _set_active(model, active):
    if active in ("relu", "ReLU"):
        model.active = nn.ReLU(inplace=True)
    elif active in ("tanh", "Tanh"):
        model.active = nn.Tanh()
    elif active in ("prelu", "PReLU"):
        model.active = nn.PReLU()
    elif active in ("leakyrelu", "LeakyReLU"):
        model.active = nn.LeakyReLU()
    elif active is None:
        pass
    else:
        raise ActiveNotIncludeError

# load the model
def load_models(cfg: Config, opts: Argparse) -> None:
    if opts.model == "cifar10":
        from models import cifar10
        model = cifar10.CIFAR10()
    elif opts.model == "cifar10rbc":
        from models import cifar10
        model = cifar10.CIFAR10rbc()
    elif opts.model == "wideresnet34":
        from models import wide_resnet
        model = wide_resnet.WideResnet(34, 10, 10)
    elif opts.model == "wideresnet34rbc":
        from models import wide_resnet
        model = wide_resnet.WideResnetrbc(34, 10, 10)
    elif opts.model == "wideresnet34-100":
        from models import wide_resnet
        model = wide_resnet.WideResnet(34, 10, 100)
    elif opts.model == "wideresnet34rbc-100":
        from models import wide_resnet
        model = wide_resnet.WideResnetrbc(34, 10, 100)
    elif opts.model == "mnist":
        from models import mnist
        model = mnist.MNIST()
    elif opts.model == "mnistrbc":
        from models import mnist
        model = mnist.MNISTrbc()
    else:
        raise ModelNotDefineError("Model {0} is not defined.".format(opts.model))
    _set_active(model, opts.active) # Set the active function of the linear classifier
    cfg['model'], cfg['device'] = gpu(model)
    # Make the path for logging paras and data
    cfg.info_path = INFO_PATH.format(modelname=opts.model) 
    cfg.log_path = LOG_PATH.format(modelname=opts.model)
    if hasattr(opts, "file"):
        cfg.info_path += opts.file
        cfg.log_path += opts.file
        attackname = ATTACKNAME.format(
            attack=opts.attack,
            epsilon=opts.epsilon,
            epsilon_times=opts.epsilon_times,
            stepsize=opts.stepsize,
            steps=opts.steps
        )
        cfg.log_path += attackname
    else:
        typename = TYPENAME.format(
            description=opts.description,
            loss_func=opts.loss_func,
            active=opts.active,
            scale=opts.scale,
            lr=opts.lr,
            leverage=opts.leverage
        )
        cfg.info_path += typename
        cfg.log_path += typename
    
# normalizing during training
class _Normalize:

    def __init__(self, mean=None, std=None):
        self.set_normalizer(mean, std)

    def set_normalizer(self, mean, std):
        if mean is None or std is None:
            self.flag = False
            return 0
        self.flag = True
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.nat_normalize = T.Normalize(
            mean=mean, std=std
        )
        self.inv_normalize = T.Normalize(
            mean=-mean/std, std=1/std
        )

    def _normalize(self, imgs, inv):
        if not self.flag:
            return imgs
        if inv:
            normalizer = self.inv_normalize
        else:
            normalizer = self.nat_normalize
        new_imgs = [normalizer(img) for img in imgs]
        return torch.stack(new_imgs)

    def __call__(self, imgs, inv=False):
        return self._normalize(imgs, inv)


def get_normalizer(model):
    if model in TRANSFORMS['CIFAR10']:
        normalizer = _Normalize(
            mean=MEANS['CIFAR10'],
            std=STDS['CIFAR10']
        )
    elif model in TRANSFORMS['CIFAR100']:
        normalizer = _Normalize(
            mean=MEANS['CIFAR100'],
            std=STDS['CIFAR100']
        )
    else:
        normalizer = _Normalize()
    return normalizer

# some transformation on inputs
def _get_transform(model, train):
    transform = T.ToTensor()
    if model in TRANSFORMS['CIFAR10'] and train:
        transform = T.Compose((
                T.Pad(4, padding_mode='reflect'),
                T.RandomHorizontalFlip(),
                T.RandomCrop(32),
                T.ToTensor()
            ))
    elif model in TRANSFORMS['CIFAR100'] and train:
        transform = T.Compose((
                T.Pad(4, padding_mode='reflect'),
                T.RandomHorizontalFlip(),
                T.RandomCrop(32),
                T.ToTensor()
            )) 
    return transform

# basic dataset
def _dataset(opts: Argparse, train=True):
    """
    If you rewrite DATASETS in config.py, don't forget 
    to check here.
    """
    transform = _get_transform(opts.model, train=train)
    if opts.model in DATASETS['CIFAR10']:
        dataset = torchvision.datasets.CIFAR10(
            root=ROOT, train=train, download=False,
            transform=transform
        )
    elif opts.model in DATASETS['CIFAR100']:
        dataset = torchvision.datasets.CIFAR100(
            root=ROOT, train=train, download=False,
            transform=transform
        )
    elif opts.model in DATASETS['MNIST']:
        dataset = torchvision.datasets.MNIST(
            root=ROOT, train=train, download=False,
            transform=transform
        )
    return dataset

# load the trainset
def load_trainset(cfg: Config, opts: Argparse) -> None:
    # set trainset
    cfg['normalizer'] = get_normalizer(opts.model)
    dataset = _dataset(opts, train=True)
    cfg['train_size'] = len(dataset)
    cfg['trainloader'] = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size,
                                            shuffle=opts.shuffle, num_workers=opts.num_workers,
                                            pin_memory=opts.pin_memory)
    cfg['epoches'] = opts.epoches

# load the testset
def load_testset(cfg: Config, opts: Argparse) -> None:
    # set testset
    dataset = _dataset(opts, train=False)
    cfg['test_size'] = len(dataset)
    cfg['testloader'] = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size,
                                            shuffle=False, num_workers=opts.num_workers,
                                            pin_memory=opts.pin_memory)

# load the validset, informally
def load_validset(valider: Config, opts: Argparse) -> None:
    # set validset
    # Note that a suitable way to set validation may
    # split the training set into  training set and 
    # validation set. However, since we just want to
    # observe the exprimental progress but not for advoiding
    # overfitting, we simply set the validation as testing set.
    # If you need, you can overwrite this function for validation,
    # meanwhile the load_trainset should be overwriten.
    dataset = _dataset(opts, train=False)
    valider['valid_size'] = len(dataset)
    valider['validloader'] = torch.utils.data.DataLoader(dataset, batch_size=16,
                                            shuffle=False, num_workers=opts.num_workers,
                                            pin_memory=opts.pin_memory)

# load the optimizer
def load_optims(cfg: Config, opts: Argparse) -> None:
    # set optim
    if opts.optim == "sgd":
        cfg['optim'] = torch.optim.SGD(cfg.model.parameters(), 
                            lr=opts.lr, momentum=opts.momentum)
    elif opts.optim == "adam":
        cfg['optim'] = torch.optim.Adam(cfg.model.parameters(),
                            lr=opts.lr, betas=opts.betas)
    else:
        raise OptimNotDefineError("Optim {0} is not included.".format(opts.optim))
    policy = LEARNING_POLICY[opts.model]
    cfg['learning_policy'] = getattr(
        torch.optim.lr_scheduler, 
        policy[0]
    )(cfg.optim, **policy[1])

# the preprocessing used on pytorchmodel
def _get_preprocessing(model):
    preprocessing = None
    if model in TRANSFORMS['CIFAR10']:
        preprocessing = dict(
            mean=MEANS['CIFAR10'],
            std=STDS['CIFAR10'],
            axis=-3
        )
    elif model in TRANSFORMS['CIFAR100']:
        preprocessing = dict(
            mean=MEANS['CIFAR100'],
            std=STDS['CIFAR100'],
            axis=-3
        )
    return preprocessing

# basic attacks
def _attack(attack_type, stepsize, steps):
    if attack_type == "pgd":
        attack = fb.attacks.LinfPGD(
            rel_stepsize=stepsize,
            steps=steps
        )
    elif attack_type == "cwl2":
        attack = fb.attacks.L2CarliniWagnerAttack(
            stepsize=stepsize,
            steps=steps
        )
    elif attack_type == "deepfool":
        attack = fb.attacks.LinfDeepFoolAttack(
            overshoot=stepsize,
            steps=steps
        )
    elif attack_type == "bb":
        attack = fb.attacks.LinfinityBrendelBethgeAttack(
            overshoot=1.1,
            lr=stepsize,
            steps=steps
        )
    else:
        raise AttackNotIncludeError("Attack {0} is not included.".format(attack_type))
    return attack

# load the attack
def load_attacks(attacker: Config, opts: Argparse) -> None:
    # set attack
    attacker['attack'] = _attack(opts.attack, opts.stepsize, opts.steps)
    attacker['epsilon'] = opts.epsilon
    attacker['bounds'] = opts.bounds
    attacker['preprocessing'] = _get_preprocessing(opts.model)

# load the valider
def load_valid(valider: Config, opts: Argparse) -> None:
    load_validset(valider, opts)
    # set attack
    valider['attack'] = _attack(opts.attack, opts.valid_stepsize, opts.valid_steps)
    valider['epsilon'] = opts.epsilon
    valider['bounds'] = opts.bounds
    valider['preprocessing'] = _get_preprocessing(opts.model)

# load the loss function
def load_loss_func(cfg: Config, opts: Argparse) -> None:
    # set the loss function
    if opts.model in DATASETS['CIFAR100']:
        num_classes = 100
    else:
        num_classes = 10
    if opts.loss_func == "rbc":
        cfg['loss_func'] = RbcLoss(
            cfg.device, 
            scale=opts.scale, 
            leverage=opts.leverage,
            num_classes=num_classes
        )
    elif opts.loss_func == "mmc":
        cfg['loss_func'] = MMCLoss(
            cfg.model,
            opts.scale
        )
    elif opts.loss_func == "rbcmmc":
        loss1 = MMCLoss(cfg.model, opts.scale)
        cfg['loss_func'] = RbcLoss(
            cfg.device,
            loss1=loss1,
            scale=opts.scale,
            leverage=opts.leverage,
            num_classes=num_classes
        )
    else:
        raise LossNotDefineError()
        






