
import argparse

def get_args(outer_parser=None):

    # args that should not be changed
    if outer_parser is None:
        parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    else:
        parser = outer_parser
    #
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env_name', default="HalfCheetah-v2", help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian", help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True, help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G', help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G', help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G', help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G', help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N', help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N', help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N', help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N', help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N', help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N', help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N', help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', type=int, default=1, metavar='N', help='run on CUDA (default: False)')

    #
    parser.add_argument('--replay_filter_coeff', type=float, default=0.5, metavar='N', help='0. to 1.')
    parser.add_argument('--iter_filter_coeff', type=float, default=0.5, metavar='N', help='0. to inf')
    parser.add_argument('--sampler_type', type=str, default='uniform', metavar='N', help='stratified,uniform,filtered')
    parser.add_argument('--algo', type=str, default='sac', metavar='N', help='stratified,uniform,filtered')
    parser.add_argument('--num_particles', type=int, default=256, metavar='N', help='stratified,uniform,filtered')
    parser.add_argument('--num_clusters', type=int, default=256, metavar='N', help='stratified,uniform,filtered')

    # sampler type
    parser.add_argument('--policy_sampler_type', type=str, default='uniform', metavar='N', help='stratified,uniform,filtered')
    parser.add_argument('--critic_sampler_type', type=str, default='uniform', metavar='N', help='stratified,uniform,filtered')
    parser.add_argument('--combined_update', type=int, default=0, metavar='N', help='stratified, uniform, filtered')

    # optimizers stuff
    parser.add_argument('--critic_optim', type=str, default='Adam', metavar='N')
    parser.add_argument('--policy_optim', type=str, default='Adam', metavar='N')
    parser.add_argument('--n_batches_per_epoch', type=int, default=1, metavar='N')

    # project
    parser.add_argument('--project', type=str, default='optimizers-in-rl', metavar='N')
    parser.add_argument('--group', type=str, default='static-args', metavar='N')

    # parse
    args = parser.parse_args()
    return args
