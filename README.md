


## Improve Adversarial Robustness via Weight Penalization on Classification Layer

### Libraries

This project is tested under the following environment settings:


- Python: 3.7.7
- foolbox: 3.1.1
- pytorch: 1.4.0


### File Structure

- data: This file stroes datasets, you could overwrite the config.py in utils to specify the path.
- rbc
    - models: some nerual network structures
    - utils: ...
    - adv_train_model.py: for adversarial training
    - train_model.py: for standard training
    - attack_model.py: to attack the trained models
    - mta_pgd.py: the multi-target attack

In the following, we first give the codes for training in different nerual network structures and on different datasets. 
Then, how to evaluate these models (e.g. AdvTraing, MMC or Ours) by attack_model.py and mta_pgd.py.

### Training

We first introduce some options given below:

- model: choose the nerual network structure and corresponding dataset
- attack: choose the attack for robustness evaluation
- active: choose the activation of the last layer of encoder part
- loss_func: choose the loss function for training
- scale: the hyperparameter to adjust the scale of weight's norm
- leverage: the hyperparameter to adjust the ratio of penalty
- epsilon: Typically, it is 0.3 for MNIST and 8/255 for CIFAR-10 and CIFAR-100 with a default setting 8/255 and therefore, you should reset it if you want to train on MNIST or other settings.
- valid_stepsize|stepsize: the step size for attacks;specially, it becomes rel_stepsize for pgd, overshoot for deepfool and lr for bb.
- valid_steps|steps: the number of iterations of attacks
- bounds: the inputs' bounds

The rest is some baisc options for training.
- epoches
- batch_size
- num_workers
- pin_memory
- shuffle
- optim: sgd or adam
- lr
- momentum: for sgd
- betas: for adam
- is_load_checkpoint: default false
  

Ours:

    python train_model.py mnisrbc pgd tanh --epsilon 0.1 --valid_stepsize 0.03333333 --valid_steps 100

    python train_model.py cifar10rbc pgd tanh

    python train_model.py wideresnet34rbc pgd tanh

    python train_model.py wideresnet34rbc-100 pgd tanh


MMC:

    python train_model.py mnist pgd relu --loss_func mmc --epsilon 0.1 --valid_stepsize 0.0333333 --valid_steps 100

    python train_model.py cifar10 pgd relu --loss_func mmc

AdvTraining

    python adv_train_model.py mnist pgd relu --epsilon 0.3 --valid_stepsize 0.0333333 --valid_steps 100

    python adv_train_model.py cifar10 pgd relu

### Evaluation

- file: the first parent directory including paras.pt; in the following suppose this file is named train-0.1-0.01-100
- epsilon_times: Sometimes you need to attack a model with different epsilons together.

In the following, we fix the neural network as cifar10, similar evaluation can be extend to other models.

Ours:

    python train_model.py cifar10rbc pgd train-0.1-0.01-100 tanh

    python train_model.py cifar10rbc cwl2 train-0.1-0.01-100 tanh --epsilon 1 --stepsize 0.01 --steps 1000

    python train_model.py cifar10rbc deepfool train-0.1-0.01-100 tanh --stepsize 0.02 --steps 50

    python mta_pgd.py cifar10rbc train-0.1-0.01-100



MMC:

    python train_model.py cifar10 pgd train-0.1-0.01-100 tanh

    python train_model.py cifar10 cwl2 train-0.1-0.01-100 tanh --epsilon 1 --stepsize 0.01 --steps 1000

    python train_model.py cifar10 deepfool train-0.1-0.01-100 tanh --stepsize 0.02 --steps 50

    python mta_pgd.py cifar10 train-0.1-0.01-100

AdvTraining

    python train_model.py cifar10 pgd train-0.1-0.01-100 tanh

    python train_model.py cifar10 cwl2 train-0.1-0.01-100 tanh --epsilon 1 --stepsize 0.01 --steps 1000

    python train_model.py cifar10 deepfool train-0.1-0.01-100 tanh --stepsize 0.02 --steps 50

    python mta_pgd.py cifar10 train-0.1-0.01-100


