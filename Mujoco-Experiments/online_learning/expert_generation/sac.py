
#
import os
import torch
import torch.nn.functional as F
import wandb
import time

#
from online_learning.expert_generation.utils import select_optimizer, timer
from online_learning.expert_generation.utils import soft_update, hard_update, select_optimizers
from online_learning.expert_generation.models.policies import GaussianPolicy
from online_learning.expert_generation.models.critics import QNetwork


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.args = args
        self.algo = args.algo
        self.num_particles = args.num_particles
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.combined_update = args.combined_update
        self.device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        # self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # set expert policy type and load expert parameters
        # self.expert = GaussianPolicy(num_inputs, action_space.shape[0], 256, action_space).to(self.device)
        # self.expert.load_state_dict(torch.load(args.expert_params_path,map_location=self.device))

        # set policy type
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], 256, action_space).to(self.device)

        #
        self.policy_optim, self.critic_optim = select_optimizers(self.policy, self.critic, args)

        try:
            self.policy.load_state_dict(torch.load('./models/sac_actor_'+args.env_name+'_expert'))
            self.critic.load_state_dict(torch.load('./models/sac_critic_'+args.env_name+'_expert'))
        except:
            print('no data is available')
        #
        self.call_backwards_critic = True if args.critic_optim not in ['Sls','Sps','Ssn','SlsEg','SlsAcc'] else False
        self.call_backwards_policy = True if args.policy_optim not in ['Sls','Sps','Ssn','SlsEg','SlsAcc'] else False
        #
        self.update_stored_lr_critic = True if args.critic_optim in ['Sls','Sps','Ssn','SlsEg','SlsAcc'] else False
        self.update_stored_lr_policy = True if args.policy_optim in ['Sls','Sps','Ssn','SlsEg','SlsAcc'] else False

        self.policy_step_size = args.lr
        self.critic_step_size = args.lr

        self.lr = args.lr
        self.policy_updates = 0
        self.start = time.time()

        # include batch in step
        self.include_batch = True if args.policy_optim in ['Sps'] else False

        # include batch in step
        self.updates = 0
        self.start = time.time()
        self.avg_return = None
        self.expert_return = None
        self.beta = 0

    # create stored trajectory statistics
    def generate_example_stats(self, memory):
        #
        self.k_lag_coeffs = compute_buffer_autocorr(memory, self.k_lag).to(self.device)
        #
        self.marginal_kdes = compute_buffer_marginal_kdes(memory)
        #
        return self.k_lag_coeffs, self.marginal_kdes

    def display(self):
        print("=========================================")
        print("Algorithm: OSAC, Policy Loss Type: {}".format(self.algo))
        print("mask logprob: {}, dyna loss: {}".format(None, None))
        print("reward loss: {}, total examples: {}".format(None, self.args.replay_size))
        print("discriminator loss: {}, value function loss: {}".format(None, None))
        print("Q-function-1 loss: {}, Q-function-2 loss: {}".format(self.qf1_loss_val,self.qf2_loss_val))
        print("dynamics updates: {}, policy updates: {}".format(0, self.policy_updates))
        print("policy loss: {}, Time-elapsed: {} ".format(self.policy_loss_val, timer(self.start,time.time())))
        print("Policy Return: {}, Expert Return: {} ".format(self.avg_return,  self.expert_return))
        print("Dual Loss: {}, Dual Parameters: {} ".format(self.alpha_loss_val, self.alpha_val))
        print("=========================================")

    def log(self):
        # log all info we have in wandb
        wandb.log({'total_examples':self.args.replay_size,
                   'dual_loss': self.alpha_loss_val, 'log_alpha': self.alpha_val,
                   'avg_return': self.avg_return, 'expert_return': self.expert_return,
                   'q_loss_1': self.qf1_loss_val, 'q_loss_1': self.qf2_loss_val,
                   'policy_updates': self.policy_updates, 'policy_loss':self.policy_loss_val},
                 step=self.policy_updates)


    # model interactions
    def select_action(self, state, evaluate=False, eval_expert=False):

        assert (evaluate and not eval_expert) or (not evaluate and eval_expert) or (not evaluate and not eval_expert)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        # sample policy data
        if evaluate is False:
            action, log_prob, _ = self.policy.sample(state)
            return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()
        else:
            _, log_prob, action = self.policy.sample(state)
            return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()

    # fuck python so hard rn, it wont let me create locally defined functions
    def sac_policy_update(self, batch, backwards):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = batch
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        self.policy_optim.zero_grad()
        if backwards:
            policy_loss.backward()
        return policy_loss

    def krac_policy_update(self, batch, backwards):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = batch
        state_batch = state_batch.unsqueeze(1).expand(state_batch.shape[0], self.num_particles, state_batch.shape[1])\
                        .contiguous()\
                    .view(state_batch.shape[0] * self.num_particles, state_batch.shape[1])
        action_batch, log_pi, _ = self.policy.sample(state_batch, reparam=False)
        with torch.no_grad():
            qf1, qf2 = self.critic(state_batch, action_batch)
            log_weight = torch.min(qf1, qf2) - self.alpha * (log_pi)
            log_weight = log_weight.view(-1, self.num_particles)
            max_log_weight = torch.max(log_weight, dim=1, keepdim=True).values
            log_weight = log_weight - max_log_weight
            lse = torch.log(torch.sum(torch.exp(log_weight), dim=1, keepdim=True))
            normalized_weight = log_weight.detach() - lse.detach()
            exp_weight = torch.exp(normalized_weight).detach() + 1e-8
        loss = torch.sum(exp_weight * (log_pi.view(-1, self.num_particles)), dim=1)
        policy_loss = -1 * torch.mean(loss)
        # print(policy_loss,  self.policy_optim.state['step_size'])
        self.policy_optim.zero_grad()
        if backwards:
            policy_loss.backward()
        return policy_loss

    def update_parameters(self, memory, batch_size, updates):

        # check if we need to resample for critic
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        critic_sample = state_batch, action_batch, reward_batch, next_state_batch, mask_batch
        policy_sample = state_batch, action_batch, reward_batch, next_state_batch, mask_batch

        # create a closure# step
        def critic_closure(return_info=False, backwards=self.call_backwards_critic):
            # set sample
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = critic_sample
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
            # compute them updates
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss
            self.critic_optim.zero_grad()
            if backwards:
                qf_loss.backward()
            if not return_info:
                return qf_loss
            else:
                return qf1_loss, qf2_loss

        # now step the Q function
        self.critic_optim.step(critic_closure)

        # set sample for policy
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = policy_sample
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # create closure
        if self.algo == 'sac':
            batch = (state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
            policy_closure = lambda : self.sac_policy_update(batch,self.call_backwards_policy)
        elif self.algo == 'krac':
            batch = (state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
            policy_closure = lambda : self.krac_policy_update(batch,self.call_backwards_policy)
        else:
            raise Exception('')

        # step policy
        self.policy_optim.step(policy_closure)

        # update entropy
        if self.automatic_entropy_tuning:
            action_batch, log_pi, _ = self.policy.sample(state_batch, reparam=False)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        qf1_loss, qf2_loss = critic_closure(return_info=True, backwards=False)
        policy_loss = self.krac_policy_update(batch,False)

        # check if we need batch info
        if self.update_stored_lr_policy:
            self.policy_step_size = self.policy_optim.state['step_size']
        else:
            self.policy_step_size = self.lr
        # check if we need batch info
        if self.update_stored_lr_critic:
            self.critic_step_size = self.policy_optim.state['step_size']
        else:
            self.critic_step_size = self.lr

        self.qf1_loss_val = qf1_loss.item()
        self.qf2_loss_val = qf2_loss.item()
        self.policy_loss_val = policy_loss.item()
        self.alpha_loss_val = alpha_loss.item()
        self.alpha_val = alpha_tlogs.item()

        self.policy_updates += 1

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", prefix=None, actor_path=None, critic_path=None):

        if prefix is not None:
            if not os.path.exists(prefix):
                os.makedirs(prefix)
        else:
            if not os.path.exists('models/'):
                os.makedirs('models/')
            prefix = 'models/'

        if actor_path is None:
            actor_path = prefix+"sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = prefix+"sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
