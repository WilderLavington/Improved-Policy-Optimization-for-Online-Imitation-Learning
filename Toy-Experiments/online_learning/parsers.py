
import argparse

def grid_world_args(parser):

    # grab parse.
    parser.add_argument('--START_STATE_x', type=int, default=0)
    parser.add_argument('--START_STATE_y', type=int, default=5)
    parser.add_argument('--ENV_DIMS_x', type=int, default=10)
    parser.add_argument('--ENV_DIMS_y', type=int, default=10)
    info, _ = parser.parse_known_args()

    return info, parser


def general_args():

    # grab parse.
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--SAMPLE_SIZE', type=int, default=1)
    parser.add_argument('--STATE_DIM', type=int, default=12)
    parser.add_argument('--ACTION_DIM', type=int, default=3)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--total_rounds', type=int, default=50)
    parser.add_argument('--max_inner_steps', type=int, default=100)
    parser.add_argument('--log_lr', type=float, default=-1.)
    parser.add_argument('--problem_type', type=str, default='norm', help='norm,adv,stoch')
    parser.add_argument('--exp_id', type=str, default='01')
    parser.add_argument('--algo', type=str, default='OGD_step')
    parser.add_argument('--exp_sweep', type=int, default=1)
    parser.add_argument('--log_param', type=float, default=-1.)
    parser.add_argument('--project', type=str, default="Toy-Experiments")
    parser.add_argument('--eval_model', type=str, default="policy", help='policy, expert, random')
    info, _ = parser.parse_known_args()

    return info, parser
