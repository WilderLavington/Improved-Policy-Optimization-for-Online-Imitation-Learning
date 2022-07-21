
import argparse

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env_name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=int, default=0, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=100000, metavar='N', help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', type=int, default=1, metavar='N', help='run on CUDA (default: False)')
    #
    parser.add_argument('--replay_filter_coeff', type=float, default=0.5, metavar='N', help='0. to 1.')
    parser.add_argument('--iter_filter_coeff', type=float, default=0.5, metavar='N', help='0. to inf')
    parser.add_argument('--policy_sampler_type', type=str, default='uniform', metavar='N', help='stratified,uniform,filtered')
    parser.add_argument('--algo', type=str, default='BC', metavar='N', help='stratified,uniform,filtered')
    parser.add_argument('--num_particles', type=int, default=256, metavar='N', help='stratified,uniform,filtered')
    parser.add_argument('--num_clusters', type=int, default=256, metavar='N', help='stratified,uniform,filtered')

    # optimizers stuff
    parser.add_argument('--expert_params_path', type=str, default='models/', metavar='N')
    parser.add_argument('--policy_optim', type=str, default='Adam', metavar='N')
    parser.add_argument('--critic_optim', type=str, default='Adam', metavar='N')

    # sls args
    parser.add_argument('--sls_beta_b', type=float, default=0.9, metavar='N')
    parser.add_argument('--sls_c', type=float, default=0.5, metavar='N')
    parser.add_argument('--sls_gamma', type=float, default=2.0, metavar='N')
    parser.add_argument('--sls_beta_f', type=float, default=2.0, metavar='N')
    parser.add_argument('--log_init_step_size', type=float, default=4., metavar='N')
    parser.add_argument('--sls_eta_max', type=float, default=100., metavar='N')

    # project
    parser.add_argument('--project', type=str, default='optimizers-in-bc', metavar='N')
    parser.add_argument('--group', type=str, default='static-args', metavar='N')
    parser.add_argument('--data_file_location', type=str, default='./data/examples/', metavar='N')


    # BC stuff
    parser.add_argument('--behavior_type', type=str, default='static-args', metavar='N')
    parser.add_argument('--expert_mode', type=int, default=0, metavar='N')
    parser.add_argument('--model_type', type=str, default='nn', metavar='N')
    parser.add_argument('--bandwidth', type=float, default=0., metavar='N')

    # better grid-search
    parser.add_argument('--log_lr', type=float, default=-3., metavar='N')
    parser.add_argument('--log_eps', type=float, default=-5., metavar='N')
    parser.add_argument('--use_log_lr', type=int, default=1, metavar='N')
    parser.add_argument('--transform_dist', type=int, default=1, metavar='N')
    parser.add_argument('--nonlin', type=str, default='relu', metavar='N')
    parser.add_argument('--clamp', type=int, default=0, metavar='N')

    # sps stuff
    parser.add_argument('--use_torch_dataloader', type=int, default=1, metavar='N')
    parser.add_argument('--sps_eps', type=int, default=0, metavar='N')
    parser.add_argument('--batch_in_step', type=int, default=0, metavar='N')

    # world models stuff
    parser.add_argument('--train_world_model', type=int, default=0, metavar='N')
    parser.add_argument('--mpc_algo', type=str, default='WM', metavar='N')
    parser.add_argument('--dyna_optim', type=str, default='Adam', metavar='N')
    parser.add_argument('--log_gan_scale', type=float, default=0., metavar='N')
    parser.add_argument('--marginal_steps', type=int, default=15, metavar='N')
    parser.add_argument('--policy_updates', type=int, default=1, metavar='N')
    parser.add_argument('--dyna_model_type', type=str, default='nn', metavar='N')
    parser.add_argument('--dyna_bandwidth', type=float, default=0., metavar='N')

    # parse
    args, _ = parser.parse_known_args()

    return args
