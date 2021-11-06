#!/usr/bin/env python3
import random
import numpy as np
import argparse
import torch
from torch import nn, optim
import mamlS
import pickle
import learn2learn as l2l
from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels)
import time

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


def main():

    ways=args.ways
    shots=args.shots
    meta_lr=args.meta_lr
    fast_lr=args.fast_lr
    meta_batch_size=args.meta_batch_size
    adaptation_steps=args.steps
    num_iterations=args.num_iterations
    cuda=args.cuda
    seed=args.seed

    save_interval = args.save_interval
    results_dir = args.results_dir
    model_name = "miniImagenet_" + args.method + "_lr_"+str(fast_lr)+"_ways_"+str(ways)+\
                "_shots_"+str(shots)+"_steps_"+str(adaptation_steps)+\
                "_seed_"+str(seed)+"_iter_" +str(args.num_iterations)

    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("seed: ",seed)
    print("Number of iterations: ",num_iterations)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:0')
   
    print("Running on device: ",device)
    tasksets = l2l.vision.benchmarks.get_tasksets('mini-imagenet',
                                                  train_samples=2*shots,
                                                  train_ways=ways,
                                                  test_samples=2*shots,
                                                  test_ways=ways,
                                                  root='~/data',
    )

    # Create model
    model = l2l.vision.models.MiniImagenetCNN(ways)
    model.to(device)
    
    if args.method == "MAML":
        maml = mamlS.MAML(model, lr=fast_lr, first_order=False, sign=False)
    elif args.method == "FO-MAML":
        maml = mamlS.MAML(model, lr=fast_lr, first_order=True, sign=False)
    elif args.method == "SIGN-MAML":
        maml = mamlS.MAML(model, lr=fast_lr, first_order=True, sign=True)

    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    times = []
    overall_start_time = time.time()

    for iteration in range(1,num_iterations+1):
        start_time = time.time()
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()
        end_time = time.time()

        # save results
        # if iteration % save_interval == 0:
        #     model_name_temp = model_name + "_iter_" + str(iteration)
        #     torch.save(maml.module.state_dict(), results_dir+model_name_temp+".pt")

        # Print some metrics
        train_loss.append(meta_train_error / meta_batch_size)
        train_acc.append(meta_train_accuracy / meta_batch_size)
        times.append(end_time - start_time)
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print("Time elapsed: ",end_time - start_time)

    overall_finish_time = time.time()
    print("Total time: ",overall_finish_time - overall_start_time)
    
    results_summary = {
        "train_acc": train_acc,
        "train_loss": train_loss,
        "time": times
    }
    pickle.dump(results_summary, open(results_dir+model_name+".p", "wb" ))
    torch.save(maml.module.state_dict(), results_dir+model_name+".pt")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--method",type=str,help="method",default="SIGN-MAML") 
    argparser.add_argument("--ways",type=int,help="nways",default=5)
    argparser.add_argument("--shots",type=int,help="k_shot",default=1)
    argparser.add_argument("--meta_lr",type=float,help="meta_lr",default=0.001)
    argparser.add_argument("--fast_lr",type=float,help="fast_lr",default=0.01)
    argparser.add_argument("--meta_batch_size",type=int,help="meta batch size",default=8)
    argparser.add_argument("--steps",type=int,help="steps",default=5)
    argparser.add_argument("--num_iterations",type=int,help="num_iterations",default=1)
    argparser.add_argument("--cuda",type=bool,help="True/False",default=True)
    argparser.add_argument("--seed",type=int,help="seed",default=1) 
    argparser.add_argument("--save_interval",type=int,help="save interval",default=10000)
    argparser.add_argument("--results_dir",type=str,help="results dir",default="trainResults/") 
    args=argparser.parse_args()
    main()
