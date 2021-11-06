import numpy as np
import pickle
import utils

params = {}
params["ways"] = 5
params["shots"] = 1
params["meta_lr"] = 0.001
params["meta_batch_size"] = 8
params["num_iterations"] = 60000
params["cuda"] = True
params["seed"] = 1
params["save_interval"] = 10000
params["adaptation_steps_train"] = 5
params["adaptation_steps_test"] = 10
dataset = "miniImagenet"
params["results_dir"] = "trainResults/"
params["method"] = "FO-MAML"
params["fast_lr"] = 0.1

loss_avg, acc_avg, loss_std, acc_std = utils.testing(params)
results = (loss_avg, acc_avg, loss_std, acc_std)
results_name = "testResults/miniImagenet_5way_1shot_" + params["method"]
results_name = results_name
pickle.dump(results,open(results_name+".p","wb"))
print((acc_avg, acc_std))