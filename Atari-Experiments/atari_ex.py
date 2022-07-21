
#### general imports
import argparse
import warnings
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname('./')))
#### model imports
from online_learning.trainer import train_il_agent, evaluate_il_algorithm
from online_learning.testing import train_bc_agent
sys.path.append(os.path.dirname(os.path.dirname('../../')))
sys.path.append(os.path.dirname(os.path.dirname('./')))
###
def main():

    # args and the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_algo_flag', type=int, default=1, metavar='N')
    parser.add_argument('--run_tests', type=int, default=0, metavar='N')
    args, _ = parser.parse_known_args()

    # pretrain expert of train il policy
    if args.run_tests:
        train_bc_agent()
    elif args.eval_algo_flag:
        evaluate_il_algorithm()
    else:
        train_il_agent()

if __name__ == "__main__":
    main()
