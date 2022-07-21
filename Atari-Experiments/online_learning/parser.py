
import argparse
import numpy as np

def get_args():

    # grab parse.
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

    # optimization args
    parser.add_argument('--algo', type=str, default='OGD', help='OGD,AdaOGD,AdamOGD,FTL,FTRL,AdaFTRL,AdaOFTRL')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--log_lr', type=float, default=-4)
    parser.add_argument('--samples_per_update', type=int, default=1000)
    parser.add_argument('--loss_type', type=str, default='mle')
    parser.add_argument('--max_procs', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=1, metavar='N', help='maximum number of steps (default: 1000000)')
    parser.add_argument('--mini_batch_size', type=int, default=2048, metavar='N', help='maximum number of steps (default: 1000000)')
    parser.add_argument('--offline_wandb', type=int, default=0, metavar='N', help='maximum number of steps (default: 1000000)')

    # FTRL + variant algorithms
    parser.add_argument('--inner_policy_optim', type=str, default='Adagrad', metavar='N')
    parser.add_argument('--epochs_per_update', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--log_inner_lr', type=float, default=-4)
    parser.add_argument('--log_outer_lr', type=float, default=-4)
    parser.add_argument('--trust_region_type', type=str, default='l2')
    parser.add_argument('--log_lambda', type=float, default=-5.5)
    parser.add_argument('--ftrl_clip', type=float, default=0.15)
    parser.add_argument('--use_sgd', type=int, default=1)
    parser.add_argument('--ftrl_variant', type=str, default='Strongly-Convex')
    parser.add_argument('--add_l2_reg', type=int, default=0)
    parser.add_argument('--l2_reg', type=float, default=1e1)
    parser.add_argument('--early_stop_crit', type=float, default=1e-3)

    # general project stuff
    parser.add_argument('--cuda', type=int, default=1, metavar='N', help='run on CUDA (default: False)')
    parser.add_argument('--seed', type=int, default=np.random.randint(1000000), metavar='N', help='random seed (default: 123456)')
    parser.add_argument('--multi_seed_run', type=int, default=3, metavar='N', help='random seed (default: 123456)')
    parser.add_argument('--group', type=str, default='static-args', metavar='N')
    parser.add_argument("--env_name", help="environment ID", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument('--expert_type', type=str, default='sac')
    parser.add_argument('--project', type=str, default='AdaptiveFTRL_Exp')

    # online imitation learning stuff
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--beta_update', type=float, default=0.0)
    parser.add_argument('--stochastic_interaction', type=int, default=1, metavar='N', help='random seed (default: 123456)')
    parser.add_argument('--replay_size', type=int, default=1000, metavar='N', help='size of replay buffer (default: 10000000)')
    parser.add_argument('--expert_mode', type=int, default=0)

    # policy model info
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N', help='hidden size (default: 256)')
    parser.add_argument('--model_type', type=str, default='nn', metavar='N')
    parser.add_argument('--bandwidth', type=float, default=0., metavar='N')
    parser.add_argument('--transform_dist', type=int, default=1, metavar='N')
    parser.add_argument('--nonlin', type=str, default='relu', metavar='N')
    parser.add_argument('--clamp', type=int, default=0, metavar='N')
    parser.add_argument('--kernel_type', type=str, default='rbf')
    parser.add_argument('--static_cov', type=int, default=0)

    # parse
    args, knk = parser.parse_known_args()

    # print a quick warning
    print('the following args were ignored...', knk)

    #
    return args, parser
