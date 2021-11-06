import os
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import torch
import random 
import numpy as np
import learn2learn as l2l
import mamlS

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy

def testing(params):
    ways=params["ways"]
    shots=params["shots"]
    meta_lr=params["meta_lr"]
    fast_lr=params["fast_lr"]
    meta_batch_size=params["meta_batch_size"]
    adaptation_steps_train=params["adaptation_steps_train"]
    adaptation_steps_test=params["adaptation_steps_test"]
    num_iterations=params["num_iterations"]
    cuda=params["cuda"]
    seed=params["seed"]
    save_interval = params["save_interval"]
    results_dir = params["results_dir"]
    method = params["method"]
    model_name = "miniImagenet_" + method + "_lr_"+str(fast_lr)+"_ways_"+str(ways)+\
                "_shots_"+str(shots)+"_steps_"+str(adaptation_steps_train)+\
                "_seed_"+str(seed)+"_iter_"+str(num_iterations)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("seed: ",seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:0')

    # Create Tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets('mini-imagenet',
                                                  train_samples=2*shots,
                                                  train_ways=ways,
                                                  test_samples=2*shots,
                                                  test_ways=ways,
                                                  root='~/data',
    )

    # Create model
    model = l2l.vision.models.MiniImagenetCNN(ways)
    PATH = results_dir + model_name +".pt"
    if method == "MAML":
        maml = mamlS.MAML(model, lr=fast_lr, first_order=False, sign=False).to(device)
    elif method == "FO-MAML":
        maml = mamlS.MAML(model, lr=fast_lr, first_order=True, sign=False).to(device)
    elif method == "SIGN-MAML":
        maml = mamlS.MAML(model, lr=fast_lr, first_order=True, sign=True).to(device)
    maml.module.load_state_dict(torch.load(PATH))
    loss = nn.CrossEntropyLoss(reduction='mean')
    
    meta_test_error = []
    meta_test_accuracy = []
    for task in range(1000):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets.test.sample()
        #print(batch[0].shape)
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps_test,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error.append(evaluation_error.item())
        meta_test_accuracy.append(evaluation_accuracy.item())
        print("Processed task: ",task)
    
    loss_avg = np.mean(np.array(meta_test_error))
    loss_std = np.std(np.array(meta_test_error))
    acc_avg = np.mean(np.array(meta_test_accuracy))
    acc_std = np.std(np.array(meta_test_accuracy))
    print('Meta Test Error', loss_avg)
    print('Meta Test Accuracy', acc_avg)
    print('Meta Test Error std', loss_std)
    print('Meta Test Accuracy std', acc_std)
    return loss_avg, acc_avg, loss_std, acc_std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)