


import torch
import os
import sys
import foolbox as fb





def mkdirs(*paths):
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


# load model's parameters
def load(model, filename, cpu=False):
    """
    :param model:
    :param filename:
    :param cpu: if trained by gpu but you want to test by cpu,
                set cpu=True
    :return:
    """
    if cpu:
        model.load_state_dict(torch.load(filename, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(filename))
    model.eval()

# save the checkpoint
def save_checkpoint(path, model, optim, lr_scheduler, epoch):
    path = path + "/model-optim-lr_sch-epoch.tar"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        },
        path
    )

# load the checkpoint
def load_checkpoint(path, model, optim, lr_scheduler):
    path = path + "/model-optim-lr_sch-epoch.tar"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    epoch = checkpoint['epoch']
    return epoch



# caculate accuracy
def accuracy(model, imgs, labels):
    out = model(imgs)
    pre = out.argmax(dim=1)
    return (pre == labels).sum().item()


def valid_attack(model, device, valider, dataloader=None, data_size=None):
    """
    valider: Config
        -- attack
        -- bounds
        -- preprocessing
        -- epsilon
        -- validloader
        -- valid_size
    """
    if not dataloader:
        dataloader = valider.validloader
        data_size = valider.valid_size
    fmodel = fb.PyTorchModel(
        model, 
        bounds=valider.bounds, 
        preprocessing=valider.preprocessing,
        device=device
    )
    running_valid_accuracy = 0.
    running_valid_success = 0.
    running_valid_distance = 0.
    for i, data in enumerate(dataloader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        accuracy_count = accuracy(fmodel, imgs, labels)
        raw, clipped, is_adv = valider.attack(fmodel, imgs, labels, epsilons=valider.epsilon)
        success_count = is_adv.sum().item()
        distance = fb.distances.linf(imgs, clipped)[is_adv].sum().item()

        running_valid_accuracy += accuracy_count
        running_valid_success += success_count
        running_valid_distance += distance

    running_valid_accuracy = running_valid_accuracy / data_size
    running_valid_distance = running_valid_distance / (running_valid_success + 1e-5)
    running_valid_success = running_valid_success / data_size
    return running_valid_accuracy, running_valid_success, running_valid_distance










