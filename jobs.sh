#!/bin/sh
python main.py --method SIGN-MAML --meta_batch_size 8 --fast_lr 0.0035 --steps 5 --num_iterations 60000 --results_dir trainResults/
python main.py --method FO-MAML --meta_batch_size 8 --fast_lr 0.1 --steps 5 --num_iterations 60000 --results_dir trainResults/
python main.py --method MAML --meta_batch_size 8 --fast_lr 0.14 --steps 5 --num_iterations 60000 --results_dir trainResults/
