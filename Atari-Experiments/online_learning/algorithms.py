
# general imports
import os
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from copy import deepcopy
import wandb
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from torch.distributions import Normal
import gym
#
from .utils import timer, select_optimizer
from .utils import parameters_to_vector, avg_dicts#, secant_algorithm, steffensen_algorithm
from .policies import select_policy
from .replay_memory import ReplayMemory
from .load_expert import pretrain_args, get_expert
from .evaluation import AlgoEvaluator

AVG_EXPERT_RETURNS = {"BreakoutNoFrameskip-v4":372.00, "PongNoFrameskip-v4":21.00, "SeaquestNoFrameskip-v4":1840.00}
#
class OnlineLearningAlgo():
    #
    def __init__(self, env, args):
        super(OnlineLearningAlgo, self).__init__()

        # hyper-parameters / policy info
        self.args = args
        self.device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

        # environment info
        self.expert = get_expert(self.args)
        self.evaluator = AlgoEvaluator(env, args)
        self.env = self.expert.get_env()
        self.avg_return, self.regret = None, 0.
        self.current_state = self.env.reset()
        self.memory = ReplayMemory(deepcopy(args.replay_size), args.seed, args)
        # set expert policy type and load expert parameters
        self.features_extractor = torch.nn.Sequential(\
                deepcopy(self.expert.model.policy.features_extractor),
                deepcopy(self.expert.model.policy.mlp_extractor))

        # grab the wrapper
        obs = self.env.reset()
        self.obs_size = obs.shape
        obs = self.expert.model.policy.obs_to_tensor(obs)[0].float()
        self.num_inputs = self.features_extractor(obs)[0].shape[-1]
        self.num_outputs = self.env.action_space.n

        # set policy type
        self.policy = select_policy(self.num_inputs, self.num_outputs,
                self.features_extractor, args.hidden_size, \
                model_type=args.model_type, bandwidth=args.bandwidth, \
                nonlin=args.nonlin).to(self.device)
        self.policy.device = self.device
        self.policy.state_transform = lambda obs: self.expert.model.policy.obs_to_tensor(obs)[0].float()

        # control mixing
        self.beta = self.args.beta
        self.beta_update = self.args.beta_update
        self.stochastic_interaction = self.args.stochastic_interaction
        self.episode = 1
        self.early_stop_crit = args.early_stop_crit
        self.samples = int(args.samples_per_update)

        # logging info
        self.updates, self.interactions = 0, 0
        self.policy_return, self.policy_loss = None, None
        self.start = time.time()

        # evaluate expert
        with torch.no_grad():
            self.policy_loss, self.expert_loss = None, None
            print('policy loss', self.policy_loss, 'expert loss', self.expert_loss)
            self.expert_return = AVG_EXPERT_RETURNS[self.args.env_name]
            # self.expert_return = self.evaluate_return(self.expert.get_env(), use_random=False, use_expert=True, use_agent=False, duplicates=3)
            self.random_return = self.evaluate_return(self.expert.get_env(), use_random=True, use_expert=False, use_agent=False, duplicates=3)
            self.policy_return = self.evaluate_return(self.expert.get_env(), use_random=False, use_expert=False, use_agent=True, duplicates=3)
            print('expert_return', self.expert_return, 'random_return', self.random_return, 'policy_return', self.policy_return)

        self.info = {}
        self.inclass_regret = 0
        self.optimal_regret = 0
        self.inclass_return = None
        self.optimal_return = None

        # quick checks
        assert args.samples_per_update <= args.replay_size

    # display training information
    def display(self):
        print("=========================================")
        print("Algorithm: {}, Policy Loss Type: {}".format(self.algo, self.loss_type))
        print("Total Updates: {}, Total Examples: {}".format(self.updates, self.interactions))
        print("Beta: {}, Replay Size: {}".format(self.beta, self.replay_size))
        print("Environment: {}, Model Type: {}".format(self.args.env_name, self.args.model_type))
        print("Policy Loss: {}, Time Elapsed: {} ".format(self.policy_loss, timer(self.start,time.time())))
        print("Policy Return: {}, Expert Return: {} ".format(self.policy_return,  self.expert_return))
        # print("Behavioral Policy Return: {}, Average Regret: {}".format(self.avg_return, self.regret / max(self.replay_size,1)))
        print("General info....")
        self.print_info()
        print("=========================================")

    # log to wandb or csv
    def log(self):
        #
        assert not np.isnan(self.policy_loss)
        # log all info we have in wandb
        info = {'total_examples':self.interactions, 'beta': self.beta,
                   'policy_return': self.policy_return, 'expert_return': self.expert_return,
                   'policy_loss':self.policy_loss, 'expert_loss':self.expert_loss,
                   'update': self.updates,
                   'log_policy_loss': np.log(self.policy_loss) if self.policy_loss is not None else None,
                   'avg_inclass_regret': self.inclass_regret / max( self.interactions,1.),
                   'avg_optimal_regret': self.optimal_regret / max( self.interactions,1.),
                   'inclass_regret': self.inclass_regret,
                   'optimal_regret': self.optimal_regret,
                   'inclass_return': self.inclass_return,
                   'optimal_return': self.optimal_return}
        # additional info from sub-module
        info.update(self.info)
        for key in info.keys():
            if torch.is_tensor(info[key]):
                info[key] = info[key].cpu().detach().numpy()
        for key in info.keys():
            if ('regret' in key) and (info[key] > 0):
                info.update({'log_'+key: np.log(info[key])})
        # pass it back to wandb
        wandb.log(info)

    # update evaluation information
    def evaluate(self, new_examples, max_samples=1000):

        # subsample Examples
        memory, info, avg_reward = new_examples
        states, expert_actions, r, d = info
        rs = torch.randperm(states.size()[0])[:min(states.size()[0],max_samples)]
        states, expert_actions = states[rs,...], expert_actions[rs,...]
        info = states, expert_actions, r, d
        new_examples = memory, info, avg_reward
        # evaluate performance vs expert
        with torch.no_grad():
            self.policy_return = self.evaluate_return(self.expert.get_env(), use_expert=False, use_agent=True, duplicates=3)
            self.policy_loss, self.expert_loss = self.evaluate_loss(new_examples)

    # gather examples for learning and evaluation
    def gather_examples(self, current_state, env, memory, examples=100, use_expert=False, evaluate=False):
        # init
        states = torch.zeros( examples, *self.obs_size[1:] )
        expert_actions = torch.zeros( examples, 1 )
        rewards = torch.zeros( examples, 1 )
        dones = torch.zeros( examples, 1 )
        avg_reward, current_return, done_counter = 0, 0, 0
        e, done = 0, 0
        # do we want to interact unde the mode?
        det_policy = (not self.stochastic_interaction) or evaluate
        reset_flag = False
        # set loop
        while e < examples:
            # grab whats needed from model
            expert_action = self.select_action(current_state, use_mean=True, use_expert=True)
            sample_action = self.select_action(current_state, use_mean=det_policy, use_expert=use_expert)
            # store what matters
            states[e,...] = torch.tensor(current_state).squeeze(0)
            expert_actions[e,:] = torch.tensor(expert_action)
            # step the simulator
            next_state, reward, done, infos = env.step(sample_action)
            # push to larger memory
            if memory is not None:
                memory.push(current_state, expert_action, reward, done)
            # grab flags and rewards
            dones[e,:] = torch.tensor(done)
            rewards[e,:] = torch.tensor(reward)
            current_return += reward
            # update the current state
            current_state = deepcopy(next_state)
            e += 1
            # real reset flag check
            if infos is not None:
                episode_infos = infos[0].get("episode")
                if episode_infos is not None:
                    reset_flag = True
            # check for reset
            if reset_flag:
                current_state = env.reset()
                avg_reward += current_return
                current_return = 0
                done_counter += 1
                reset_flag = False
        avg_reward = avg_reward / max(1, done_counter)
        # return the tensors
        return memory, (states, expert_actions, rewards, dones), avg_reward, current_state

    # environment interactions
    def select_action(self, state, use_mean=False, use_expert=False, use_agent=False):
        # use expert
        if use_expert:
            if use_mean:
                return self.expert.sample(state)[0]
            else:
                return self.expert.sample(state)[2]
        if use_agent:
            if use_mean:
                return self.policy.sample(state)[0]
            else:
                return self.policy.sample(state)[2]
        # sample expert
        if torch.rand(1)[0] <= self.beta:
            if use_mean:
                return self.expert.sample(state)[0]
            else:
                return self.expert.sample(state)[2]
        # sample policy data
        else:
            if use_mean:
                return self.policy.sample(state)[0]
            else:
                return self.policy.sample(state)[2]

        # return it all
        return action

    # evaluation
    def evaluate_return(self, env, use_expert=False, duplicates=1, use_random=False, use_agent=False):
        avg_reward = 0.
        assert not (use_random and use_expert)
        with torch.no_grad():
            for _  in range(duplicates):
                state = env.reset()
                episode_reward = 0
                t = 0
                reset_flag = False
                while (not reset_flag):
                    action = self.select_action(state, use_mean=True, use_expert=use_expert, use_agent=use_agent)
                    if use_random:
                        action = [env.action_space.sample()]
                    next_state, reward, done, infos = env.step(action)
                    state = next_state
                    episode_reward += reward
                    t += 1
                    if infos is not None:
                        episode_infos = infos[0].get("episode")
                        if episode_infos is not None:
                            # print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                            # print("Atari Episode Length", episode_infos["l"])
                            episode_reward = episode_infos['r']
                            reset_flag = True
                            avg_reward += episode_reward
                # print(t, episode_reward)
            avg_reward /= duplicates
            return avg_reward

    # compute loss of policy and expert
    def evaluate_loss(self, info, use_expert=False, examples=5000):
        # grab examples and compute loss
        states, expert_actions, _, _ = info[1]
        states = states.numpy()
        expert_actions = expert_actions.numpy()
        expert_loss = self.compute_loss(states, expert_actions, True).item()
        policy_loss = self.compute_loss(states, expert_actions, False).item()
        #
        return policy_loss, expert_loss

    # different losses for use in model update
    def compute_loss(self, state_batch, expert_action_batch, use_expert=False):
        if self.args.loss_type == 'mle':
            if use_expert:
                log_liklihood = self.expert.log_prob(state_batch, expert_action_batch.reshape(-1))
            else:
                log_liklihood = self.policy.log_prob(state_batch, expert_action_batch.reshape(-1))
            return -1 * log_liklihood.sum()
        else:
            raise Exception()

    def compute_grad(self):
        # compute grad norm for posterity
        grad = torch.cat([p.grad.view(-1) if p.grad is not None\
                               else p.new_zeros(p.size()).view(-1)\
                               for p in self.policy.parameters()]).data
        return grad

    # general training loop script
    def train_agent(self):

        # intial log and display
        new_examples = self.gather_examples(self.current_state, self.env, self.memory, examples=self.samples, use_expert=False)
        memory, info, avg_reward, self.current_state = new_examples
        self.evaluate((memory, info, avg_reward))
        self.log()
        self.display()

        # check that the model will not diverge
        with torch.autograd.set_detect_anomaly(True):
            self.update_parameters((memory, info, avg_reward))

        # with torch.autograd.detect_anomaly():
        for episode in range(self.args.episodes):

            # gather new loss and increment
            self.episode = episode + 1
            new_examples = self.gather_examples(self.current_state, self.env, self.memory, examples=self.samples, use_expert=False)
            memory, info, avg_reward, self.current_state = new_examples

            # evaluate + display
            if (self.episode % self.args.log_interval == 0):
                # evalaute current performance
                self.evaluate((memory, info, avg_reward))
                # log evaluation info to wandb
                self.log()
                # also display everything
                self.display()

            # update model parameters
            self.update_parameters((memory, info, avg_reward))

            # update beta
            self.beta *= self.beta_update

        # final log and display
        new_examples = self.gather_examples(self.current_state, self.env, self.memory, examples=self.samples, use_expert=False)
        memory, info, avg_reward, self.current_state = new_examples
        self.evaluate((memory, info, avg_reward))
        self.log()
        self.display()

    # multi-runner funcions
    def agent_update(self):

        # grab new losses
        new_examples = self.gather_examples(self.current_state, self.env, self.memory, self.samples, False)
        memory, info, avg_reward, self.current_state = new_examples

        # evaluate + display
        if (self.episode % self.args.log_interval == 0):
            self.evaluate((memory, info, avg_reward))

        # algorithm update
        self.update_parameters((memory, info, avg_reward))

        # update beta
        self.episode = self.episode + 1
        self.beta *= self.beta_update

    def agent_info(self):
        info = {'total_examples':self.interactions, 'beta': self.beta,
                   'policy_return': self.policy_return, 'expert_return': self.expert_return,
                   'policy_loss':self.policy_loss, 'expert_loss':self.expert_loss,
                   'update': self.updates,
                   'log_policy_loss': np.log(self.policy_loss) if self.policy_loss is not None else None,
                   'avg_inclass_regret': self.inclass_regret / max( self.interactions,1.),
                   'avg_optimal_regret': self.optimal_regret / max( self.interactions,1.),
                   'inclass_regret': self.inclass_regret,
                   'optimal_regret': self.optimal_regret,
                   'inclass_return': self.inclass_return,
                   'optimal_return': self.optimal_return, 'buffer_size': self.memory.__len__(),
                   }
        info.update(self.info)
        return info

    # save and load existing models
    def save_model(self, env_name, suffix="", actor_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)

    def load_model(self, actor_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))

    def update_parameters(self):
        raise Exception('Not Implimented')

    def print_info(self):
        print(self.info)

class OGD(OnlineLearningAlgo):

    def __init__(self, env, args):
        super(OGD,self).__init__(env, args)
        self.lr = 10**args.log_lr
        self.algo = 'OGD'
        self.loss_type = args.loss_type
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=self.lr)
        self.replay_size = 0

    def update_parameters(self, new_examples, max_samples=5000):
        # grab examples
        _, (states, expert_actions, _, _), self.avg_return = new_examples
        self.replay_size = int(self.args.samples_per_update)
        self.interactions += int(self.args.samples_per_update)
        #
        self.optimizer.zero_grad()
        self.policy.to(self.device)
        dataset = torch.utils.data.TensorDataset(states, expert_actions)
        data_generator = torch.utils.data.DataLoader(dataset, batch_size=max_samples, shuffle=True)
        loss = torch.tensor(0.).to(self.device)
        # iterate over the entire memory and grab ftl loss
        for batch_states, batch_expert_actions in data_generator:
            ftl_loss = self.compute_loss(batch_states.to(self.device), batch_expert_actions.to(self.device))
            ftl_loss = ftl_loss / states.size()[0]
            (ftl_loss).backward()
            loss += (ftl_loss).detach()
        # compute loss
        grad_norm = self.compute_grad().detach().pow(2).mean()
        # step optimizer
        self.optimizer.step()
        self.updates += 1
        # store
        self.info = {'ogd_loss':  loss,
                     'grad_norm': grad_norm}
        # return computed loss
        return loss

class AdaOGD(OGD):
    def __init__(self, env, args):
        super(AdaOGD, self).__init__(env, args)
        self.optimizer = torch.optim.Adagrad(self.policy.parameters(), lr=self.lr)
        self.algo = 'AdaOGD'

class AdamOGD(OGD):
    def __init__(self, env, args):
        super(AdaOGD, self).__init__(env, args)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.algo = 'AdamOGD'

class FTL(OnlineLearningAlgo):
    def __init__(self, env, args):
        super(FTL,self).__init__(env, args)
        self.epochs_per_update = args.epochs_per_update
        self.lr = 10**args.log_lr
        self.algo = 'FTL'
        self.loss_type = args.loss_type
        self.samples = int(args.samples_per_update)
        self.replay_size = 0
        self.early_stop_crit = 1e-8
        self.lambda_reg = 1e-8

    def gd_step(self, tensor_states, tensor_expert_actions):
        # create closure
        def closure(verbose=False, call_backward=False,
                s=tensor_states, a=tensor_expert_actions):
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # either accumulate the gradient or use mini-batch
            cum_ftl_loss = 0
            # take single step if SGD
            if self.args.use_sgd:
                # accumulate gradients
                ftl_loss = self.compute_loss(tensor_states.to(self.device), tensor_expert_actions.to(self.device))
                # for averaging
                ftl_loss = ftl_loss / tensor_states.size()[0]
                if call_backward:
                    (ftl_loss).backward()
                cum_ftl_loss += (ftl_loss).detach()
            # iterate over the entire memory and grab ftl loss if GD
            else:
                for batch_states, batch_expert_actions in self.data_generator:
                    # accumulate gradients
                    ftl_loss = self.compute_loss(batch_states.to(self.device), batch_expert_actions.to(self.device))
                    # for averaging
                    ftl_loss = ftl_loss / self.replay_size
                    if call_backward:
                        (ftl_loss).backward()
                    cum_ftl_loss += (ftl_loss).detach()
            # do we want other info
            return cum_ftl_loss
        # step optimizer (do we need a closure?)
        if self.args.inner_policy_optim == 'LSOpt':
            # direct step
            loss = self.optimizer.step(closure)
        else:
            # backprop through computation graph
            self.optimizer.zero_grad()
            loss = closure(call_backward=True)
            self.optimizer.step()
        # compute grad norm for posterity
        loss = closure(call_backward=True)
        grad_norm = self.compute_grad().detach().pow(2).mean()
        # return
        return loss.detach(), grad_norm.detach()

    def iterative_step(self, tensor_states, tensor_expert_actions):
        # split into full-batch and SGD
        if self.args.use_sgd:
            # iterate over the entire memory
            for states_batch, expert_actions_batch in self.data_generator:
                # take gradient over data subset
                loss, grad_norm = self.gd_step(states_batch.to(self.device), expert_actions_batch.to(self.device))
        else:
            loss, grad_norm = self.gd_step(None, None)
        # return it
        return loss.detach(), grad_norm.detach()

    def update_parameters(self, new_examples):
        # set optimization info
        self.args.lr = self.lr
        self.optimizer = select_optimizer(self.policy, self.args, self.args.inner_policy_optim)

        # get sampled trajectories and push to memory
        self.memory, _, self.avg_return = new_examples
        self.replay_size = self.memory.__len__()
        self.interactions += self.args.samples_per_update
        # gather examples from memory
        states, expert_actions, _, _ = map(np.stack, zip(*self.memory.buffer))
        tensor_states = torch.stack([torch.tensor(s) for s in states]).float().detach()
        tensor_expert_actions = torch.stack([torch.tensor(s) for s in expert_actions]).float().detach()
        # move to device
        tensor_states = tensor_states.reshape(-1, *self.obs_size[1:])
        tensor_expert_actions = tensor_expert_actions.reshape(-1, 1)
        ftl_loss = torch.tensor(0.0).to(self.device)
        # make a generater
        dataset = torch.utils.data.TensorDataset(tensor_states, tensor_expert_actions)
        self.data_generator = torch.utils.data.DataLoader(dataset, batch_size=self.args.mini_batch_size, shuffle=True)
        # otherwise run iterative
        for epoch in range(self.epochs_per_update): 
            loss, grad_norm = self.iterative_step(tensor_states, tensor_expert_actions)
            self.updates += 1
            ftl_loss += loss.detach()
            # early stopping crit
            if grad_norm.item() < self.early_stop_crit:
                break
        # store logging info
        self.info = {'ftl_loss': ftl_loss,
                     'grad_norm': grad_norm, 'epochs': epoch}
        # return loss
        return loss

class FTRL(OnlineLearningAlgo):

    def __init__(self, env, args):
        super(FTRL,self).__init__(env, args)
        self.epochs_per_update = args.epochs_per_update
        self.inner_lr = 10**args.log_inner_lr
        self.outer_lr = 10**args.log_outer_lr
        self.algo = 'FTRL'
        self.loss_type = args.loss_type
        self.old_memory = None
        self.samples = int(args.samples_per_update)
        self.replay_size = 0
        self.prev_grad_sum = None
        self.eta = torch.tensor([self.outer_lr]).to(self.device)
        self.early_stop_crit = 1e-12
        self.epsilon = 1e-12
        # args to use classic version
        self.sigma = torch.tensor([1/self.outer_lr]).to(self.device)
        # projected grad decent params
        self.ftrl_clip = args.ftrl_clip
        self.ftrl_variant = args.ftrl_variant
        self.squared_gradsum = 0.
        self.params_average = 0 * parameters_to_vector(self.policy.parameters()).detach()

    def update_gradsum(self, tensor_states, tensor_expert_actions, max_batch_size=3000):
        # gather examples from memory
        states, expert_actions, _, _ = map(np.stack, zip(*self.memory.buffer))
        tensor_states = torch.stack([torch.tensor(s) for s in states]).float().detach()
        tensor_expert_actions = torch.stack([torch.tensor(s) for s in expert_actions]).float().detach()
        tensor_states = tensor_states.reshape(-1, *self.obs_size[1:])
        tensor_expert_actions = tensor_expert_actions.reshape(-1, 1)
        #
        dataset = torch.utils.data.TensorDataset(tensor_states, tensor_expert_actions)
        data_generator = torch.utils.data.DataLoader(dataset, batch_size=max_batch_size, shuffle=True)
        #
        self.policy.zero_grad()
        self.prev_grad_sum = 0. * self.compute_grad()
        # iterate over the entire memory
        for batch_states, batch_expert_actions in data_generator:
            #
            self.policy.zero_grad()
            loss = self.compute_loss(batch_states.to(self.device), batch_expert_actions.to(self.device))
            # compute gradient for the sum of those function evals
            loss.backward()
            # new
            self.prev_grad_sum += self.compute_grad()
        # return it
        return self.prev_grad_sum

    def update_squared_gradsum(self, states, expert_actions, max_batch_size=3000):
        #
        dataset = torch.utils.data.TensorDataset(states, expert_actions)
        data_generator = torch.utils.data.DataLoader(dataset, batch_size=max_batch_size, shuffle=True)
        #
        squared_grad = 0. * self.compute_grad()
        # iterate over the entire memory
        for batch_states, batch_expert_actions in data_generator:
            #
            self.policy.zero_grad()
            loss = self.compute_loss(batch_states.to(self.device), batch_expert_actions.to(self.device))
            # compute gradient for the sum of those function evals
            loss.backward()
            # new
            squared_grad += self.compute_grad()
        # update to new sum of gradients
        self.squared_gradsum += squared_grad.detach().pow(2).sum()
        # return for use in eta update
        return self.squared_gradsum

    def step(self):

        #
        def closure(verbose=False, call_backward=False):
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # either accumulate the gradient or use mini-batch
            cum_ftl_loss = 0
            # iterate over the entire memory and grab ftl loss
            for batch_states, batch_expert_actions in self.data_generator:
                # accumulate gradients
                ftl_loss = self.compute_loss(batch_states.to(self.device), batch_expert_actions.to(self.device))
                # for averaging
                ftl_loss = ftl_loss / self.replay_size
                if call_backward:
                    (ftl_loss).backward()
                cum_ftl_loss += (ftl_loss).detach()
            # compute linearized term
            if self.prev_grad_sum is not None:
                lin_loss = -1 * torch.dot(self.prev_grad_sum.detach(), parameters_to_vector(self.policy.parameters()))
            else:
                lin_loss =  0 * parameters_to_vector(self.policy.parameters()).mean()
            # divide this by length of all data
            lin_loss = lin_loss / self.replay_size
            # compute trust region
            prev_parameter_vec = deepcopy(parameters_to_vector(self.prev_policy.parameters()).detach())
            parameter_vec = parameters_to_vector(self.policy.parameters())
            diff = (prev_parameter_vec-parameter_vec).pow(2)
            tr_loss = diff.sum() / self.replay_size
            # pick ftrl_variant for scaling
            scaling = self.outer_lr * np.sqrt(self.episode)
            # compute
            loss = (lin_loss + scaling * tr_loss)
            #
            if call_backward:
                loss.backward()
            # return it
            if not verbose:
                return loss + cum_ftl_loss
            else:
                return cum_ftl_loss, lin_loss, tr_loss

        # step optimizer (do we need a closure?)
        if self.args.inner_policy_optim == 'LSOpt':
            # direct step
            loss = self.optimizer.step(closure)
            ftl_loss, lin_loss, tr_loss = closure(verbose=True, call_backward=True)
            grad_norm = self.compute_grad().detach().pow(2).mean()
        else:
            # backprop through computation graph
            loss = closure(call_backward=True)
            grad_norm =  self.compute_grad().detach().pow(2).mean()
            self.optimizer.step()
            ftl_loss, lin_loss, tr_loss = closure(verbose=True)

        # info return
        return loss.detach(), grad_norm.detach(), (ftl_loss.detach(), lin_loss.detach(), tr_loss.detach())

    def sgd_step(self, batch_states, batch_expert_actions):

        #
        def closure(verbose=False, call_backward=False):
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # either accumulate the gradient or use mini-batch
            cum_ftl_loss = torch.tensor(0.0).to(self.device)
            # accumulate gradients
            ftl_loss = self.compute_loss(batch_states.to(self.device), batch_expert_actions.to(self.device))
            # for averaging
            ftl_loss = ftl_loss / batch_states.size()[0]
            if call_backward:
                (ftl_loss).backward()
            cum_ftl_loss += (ftl_loss).detach()
            # compute linearized term
            if self.prev_grad_sum is not None:
                lin_loss = -1 * torch.dot(self.prev_grad_sum.detach(), parameters_to_vector(self.policy.parameters()))
            else:
                lin_loss =  0 * parameters_to_vector(self.policy.parameters()).mean()
            # divide this by length of all data
            lin_loss = lin_loss /  batch_states.size()[0]
            # compute trust region
            prev_parameter_vec = deepcopy(parameters_to_vector(self.prev_policy.parameters()).detach())
            parameter_vec = parameters_to_vector(self.policy.parameters())
            diff = (prev_parameter_vec-parameter_vec).pow(2)
            tr_loss = diff.sum() / batch_states.size()[0]
            # pick ftrl_variant for scaling
            scaling = self.outer_lr * np.sqrt(self.episode)
            # compute
            loss = (lin_loss + scaling * tr_loss)
            #
            if call_backward:
                loss.backward()
            # return it
            if not verbose:
                return loss + cum_ftl_loss
            else:
                return cum_ftl_loss, lin_loss, tr_loss

        # step optimizer (do we need a closure?)
        if self.args.inner_policy_optim == 'LSOpt':
            # direct step
            loss = self.optimizer.step(closure)
            ftl_loss, lin_loss, tr_loss = closure(verbose=True, call_backward=True)
            grad_norm = self.compute_grad().detach().pow(2).mean()
        else:
            # backprop through computation graph
            loss = closure(call_backward=True)
            grad_norm =  self.compute_grad().detach().pow(2).mean()
            self.optimizer.step()
            ftl_loss, lin_loss, tr_loss = closure(verbose=True)

        # info return
        return loss.detach(), grad_norm.detach(), (ftl_loss.detach(), lin_loss.detach(), tr_loss.detach())

    def update_parameters(self, new_examples):

        # get sampled trajectories and push to memory
        self.memory, (new_states, new_expert_actions, _, _), self.avg_return = new_examples
        self.replay_size = self.memory.__len__()
        self.interactions += self.args.samples_per_update
        self.prev_policy = deepcopy(self.policy)

        # gather examples from memory
        states, expert_actions, _, _ = map(np.stack, zip(*self.memory.buffer))
        tensor_states = torch.stack([torch.tensor(s) for s in states]).float().detach()
        tensor_expert_actions = torch.stack([torch.tensor(s) for s in expert_actions]).float().detach()

        # move to device
        tensor_states = tensor_states.reshape(-1, *self.obs_size[1:])
        tensor_expert_actions = tensor_expert_actions.reshape(-1, 1)

        # generate memory permutation + data generator
        dataset = torch.utils.data.TensorDataset(tensor_states, tensor_expert_actions)
        self.data_generator = torch.utils.data.DataLoader(dataset, batch_size=self.args.mini_batch_size, shuffle=True)

        # set optimization info
        self.args.lr = self.inner_lr
        self.optimizer = select_optimizer(self.policy, self.args, self.args.inner_policy_optim)

        # update gradsum for adaptive
        self.update_squared_gradsum(new_states, new_expert_actions)
        assert self.episode >= 1
        self.params_average += torch.tensor(np.sqrt(self.episode) - np.sqrt(self.episode-1)) * \
                parameters_to_vector(self.policy.parameters()).detach()

        # step through iterative updates
        for epoch in range(self.epochs_per_update):
            # take steps
            if not self.args.use_sgd:
                loss, grad_norm, (ftl_loss, lin_loss, tr_loss) = self.step()
            else:
                # iterate over the entire memory and grab ftl loss
                for batch_states, batch_expert_actions in self.data_generator:
                    loss, grad_norm, (ftl_loss, lin_loss, tr_loss) = self.sgd_step(batch_states, batch_expert_actions)
            self.updates += 1

            # early stopping crit
            if grad_norm.item() < self.early_stop_crit:
                break

        # update gradsum
        self.prev_grad_sum = self.update_gradsum(new_states, new_expert_actions)

        # store info
        self.info = {'ftl_loss': ftl_loss, 'lin_loss': lin_loss,
                     'tr_loss': tr_loss, 'eta': 1 / self.sigma * self.episode,
                     'grad_norm': grad_norm.detach(), 'internal_steps': epoch}

        #
        self.old_memory = deepcopy(self.memory)

        #
        return loss

class AFTRL(FTRL):

    def __init__(self, env, args):
        super(AFTRL,self).__init__(env, args)
        self.algo = 'AFTRL'

    def step(self):

        #
        def closure(verbose=False, call_backward=False):
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # either accumulate the gradient or use mini-batch
            cum_ftl_loss = 0
            # iterate over the entire memory and grab ftl loss
            for batch_states, batch_expert_actions in self.data_generator:
                # accumulate gradients
                ftl_loss = self.compute_loss(batch_states.to(self.device), batch_expert_actions.to(self.device))
                # for averaging
                ftl_loss = ftl_loss / self.replay_size
                if call_backward:
                    (ftl_loss).backward()
                cum_ftl_loss += (ftl_loss).detach()
            # compute linearized term
            if self.prev_grad_sum is not None:
                lin_loss = -1 * torch.dot(self.prev_grad_sum.detach(), parameters_to_vector(self.policy.parameters()))
            else:
                lin_loss =  0 * parameters_to_vector(self.policy.parameters()).mean()
            # divide this by length of all data
            lin_loss = lin_loss / self.replay_size
            # compute trust region
            prev_parameter_vec = deepcopy(parameters_to_vector(self.prev_policy.parameters()).detach())
            parameter_vec = parameters_to_vector(self.policy.parameters())
            diff = (prev_parameter_vec-parameter_vec).pow(2)
            tr_loss = diff.sum() / self.replay_size
            # pick ftrl_variant for scaling
            scaling = self.outer_lr *  self.squared_gradsum.pow(0.5)
            # compute
            loss = (lin_loss + scaling * tr_loss)
            #
            if call_backward:
                loss.backward()
            # return it
            if not verbose:
                return loss + cum_ftl_loss
            else:
                return cum_ftl_loss, lin_loss, tr_loss

        # step optimizer (do we need a closure?)
        if self.args.inner_policy_optim == 'LSOpt':
            # direct step
            loss = self.optimizer.step(closure)
            ftl_loss, lin_loss, tr_loss = closure(verbose=True, call_backward=True)
            grad_norm = self.compute_grad().detach().pow(2).mean()
        else:
            # backprop through computation graph
            loss = closure(call_backward=True)
            grad_norm =  self.compute_grad().detach().pow(2).mean()
            self.optimizer.step()
            ftl_loss, lin_loss, tr_loss = closure(verbose=True)

        # info return
        return loss.detach(), grad_norm.detach(), (ftl_loss.detach(), lin_loss.detach(), tr_loss.detach())

    def sgd_step(self, batch_states, batch_expert_actions):

        #
        def closure(verbose=False, call_backward=False):
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # either accumulate the gradient or use mini-batch
            cum_ftl_loss = 0
            # accumulate gradients
            ftl_loss = self.compute_loss(batch_states.to(self.device), batch_expert_actions.to(self.device))
            # for averaging
            ftl_loss = ftl_loss / batch_states.size()[0]
            if call_backward:
                (ftl_loss).backward()
            cum_ftl_loss += (ftl_loss).detach()
            # compute linearized term
            if self.prev_grad_sum is not None:
                lin_loss = -1 * torch.dot(self.prev_grad_sum.detach(), parameters_to_vector(self.policy.parameters()))
            else:
                lin_loss =  0 * parameters_to_vector(self.policy.parameters()).mean()
            # divide this by length of all data
            lin_loss = lin_loss / self.replay_size
            # compute trust region
            prev_parameter_vec = deepcopy(parameters_to_vector(self.prev_policy.parameters()).detach())
            parameter_vec = parameters_to_vector(self.policy.parameters())
            diff = (prev_parameter_vec-parameter_vec).pow(2)
            tr_loss = diff.sum() / self.replay_size
            # pick ftrl_variant for scaling
            scaling = self.outer_lr * self.squared_gradsum.pow(0.5)
            # compute
            loss = (lin_loss + scaling * tr_loss)
            #
            if call_backward:
                loss.backward()
            # return it
            if not verbose:
                return loss + cum_ftl_loss
            else:
                return cum_ftl_loss, lin_loss, tr_loss

        # step optimizer (do we need a closure?)
        if self.args.inner_policy_optim == 'LSOpt':
            # direct step
            loss = self.optimizer.step(closure)
            ftl_loss, lin_loss, tr_loss = closure(verbose=True, call_backward=True)
            grad_norm = self.compute_grad().detach().pow(2).mean()
        else:
            # backprop through computation graph
            loss = closure(call_backward=True)
            grad_norm =  self.compute_grad().detach().pow(2).mean()
            self.optimizer.step()
            ftl_loss, lin_loss, tr_loss = closure(verbose=True)

        # info return
        return loss.detach(), grad_norm.detach(), (ftl_loss.detach(), lin_loss.detach(), tr_loss.detach())

class SFTRL(FTRL):

    def __init__(self, env, args):
        super(SFTRL,self).__init__(env, args)
        self.algo = 'SFTRL'

    def step(self):

        #
        def closure(verbose=False, call_backward=False, zero_grad=True):
            # zero the parameter gradients
            if zero_grad:
                self.optimizer.zero_grad()
            # either accumulate the gradient or use mini-batch
            cum_ftl_loss = 0
            # iterate over the entire memory and grab ftl loss
            for batch_states, batch_expert_actions in self.data_generator:
                # accumulate gradients
                ftl_loss = self.compute_loss(batch_states.to(self.device), batch_expert_actions.to(self.device))
                # for averaging
                ftl_loss = ftl_loss / self.replay_size
                if call_backward:
                    (ftl_loss).backward()
                cum_ftl_loss += (ftl_loss).detach()
            # regularization
            current_params = parameters_to_vector(self.policy.parameters())
            lin_loss = torch.dot(current_params,current_params)
            # divide this by length of all data
            lin_loss = self.outer_lr * 0.5 * (np.sqrt(self.episode)) * lin_loss / self.replay_size
            # compute trust region
            tr_loss =  self.outer_lr * torch.dot(current_params, self.params_average.detach()) / self.replay_size
            # compute
            loss = (cum_ftl_loss + lin_loss + tr_loss).mean()
            #
            if call_backward:
                loss.backward()
            # return it
            if not verbose:
                return loss
            else:
                return cum_ftl_loss, lin_loss, tr_loss

        # step optimizer (do we need a closure?)
        if self.args.inner_policy_optim == 'LSOpt':
            # direct step
            loss = self.optimizer.step(closure)
            ftl_loss, lin_loss, tr_loss = closure(verbose=True, call_backward=True)
            grad_norm = self.compute_grad().detach().pow(2).mean()
        else:
            # backprop through computation graph
            loss = closure(call_backward=True)
            grad_norm =  self.compute_grad().detach().pow(2).mean()
            self.optimizer.step()
            ftl_loss, lin_loss, tr_loss = closure(verbose=True)

        # info return
        return loss.detach(), grad_norm.detach(), (ftl_loss.detach(), lin_loss.detach(), tr_loss.detach())

    def sgd_step(self, batch_states, batch_expert_actions):

        #
        def closure(verbose=False, call_backward=False):
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # accumulate gradients
            ftl_loss = self.compute_loss(batch_states.to(self.device), batch_expert_actions.to(self.device))
            # for averaging
            ftl_loss = ftl_loss / batch_states.size()[0]
            if call_backward:
                (ftl_loss).backward()
            # regularization
            current_params = parameters_to_vector(self.policy.parameters())
            lin_loss = torch.dot(current_params,current_params)
            # divide this by length of all data
            lin_loss = self.outer_lr * 0.5 * (np.sqrt(self.episode)) * lin_loss / batch_states.size()[0]
            # compute trust region
            tr_loss =  self.outer_lr * torch.dot(current_params, self.params_average) / batch_states.size()[0]
            # compute
            loss = (ftl_loss.detach() + lin_loss + tr_loss).mean()
            #
            if call_backward:
                loss.backward()
            # return it
            if not verbose:
                return loss + ftl_loss.detach()
            else:
                return ftl_loss.detach(), lin_loss, tr_loss

        # step optimizer (do we need a closure?)
        if self.args.inner_policy_optim == 'LSOpt':
            # direct step
            loss = self.optimizer.step(closure)
            ftl_loss, lin_loss, tr_loss = closure(verbose=True, call_backward=True)
            grad_norm = self.compute_grad().detach().pow(2).mean()
        else:
            # backprop through computation graph
            loss = closure(call_backward=True)
            grad_norm =  self.compute_grad().detach().pow(2).mean()
            self.optimizer.step()
            ftl_loss, lin_loss, tr_loss = closure(verbose=True)

        # info return
        return loss.detach(), grad_norm.detach(), (ftl_loss.detach(), lin_loss.detach(), tr_loss.detach())
