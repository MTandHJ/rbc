



# Here are some basic settings.
# It could be overwritten if you want to specify
# special configs. However, please check the correspoding
# codes in loadopts.py.


from .dict2object import Config



ROOT = "../data"

DATASETS = {
    "CIFAR10": ("cifar10", "cifar10rbc", "wideresnet34", "wideresnet34rbc"),
    "MNIST": ("mnist", "mnistrbc"),
    "CIFAR100": ("wideresnet34-100", "wideresnet34rbc-100")
}

TRANSFORMS = {
    "CIFAR10": ("wideresnet34", "wideresnet34rbc"),
    "CIFAR100": ("wideresnet34-100", "wideresnet34rbc-100")
}

MEANS = {
    "CIFAR10": [0.4914, 0.4824, 0.4467],
    "CIFAR100": [0.5071, 0.4867, 0.4408]
}

STDS = {
    "CIFAR10": [0.2471, 0.2435, 0.2617],
    "CIFAR100": [0.2675, 0.2565, 0.2761]
}

LEARNING_POLICY = {
    "wideresnet34":(
        "MultiStepLR",
        Config(
            milestones=[60, 120, 180],
            gamma=0.2
        )
    ),
    "wideresnet34-100":(
        "MultiStepLR",
        Config(
            milestones=[60, 120, 180],
            gamma=0.2
        )
    ),
    "wideresnet34rbc":(
        "MultiStepLR",
        Config(
            milestones=[60, 120, 180, 240, 300, 360],
            gamma=0.9
        )
    ),
    "wideresnet34rbc-100":(
        "MultiStepLR",
        Config(
            milestones=[60, 120, 180, 240, 300, 360],
            gamma=0.9
        )
    ),
    "cifar10":(
        "StepLR",
        Config(
            step_size=40,
            gamma=1.
        )
    ),
    "cifar10rbc":(
        "StepLR",
        Config(
            step_size=40,
            gamma=1.
        )
    ),
    "mnist":(
        "StepLR",
        Config(
            step_size=40,
            gamma=1.
        )
    ),
    "mnistrbc":(
        "StepLR",
        Config(
            step_size=40,
            gamma=1.
        )
    )
}

INFO_PATH = "./infos/{modelname}/"
LOG_PATH = "./logs/{modelname}/"
TYPENAME = "{description}-{loss_func}-{active}-{scale:.3f}-{lr:.3f}-{leverage:.1f}"
ATTACKNAME = "/{attack}-{epsilon:.3f}-{epsilon_times}-{stepsize:.3f}-{steps}"




