
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

#
from .utils import timer, load_expert, select_optimizer
from .utils import parameters_to_vector, avg_dicts
from .policies import select_policy
from .replay_memory import ReplayMemory
from .evaluation import AlgoEvaluator

#
AVG_EXPERT_RETURNS = {"Hopper-v2":3447.3, "Walker2d-v2":4367.9, "HalfCheetah-v2":16170.3}
class OnlineLearningAlgo():
    #
    def __init__(self, env, args):
        super(OnlineLearningAlgo, self).__init__()

        # hyper-parameters / policy info
        self.args = args
        self.device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

        # environment info
        self.num_inputs = env.observation_space.shape[0]
        self.action_space = env.action_space
        self.env = env
        self.avg_return, self.regret = None, 0.
        self.current_state = self.env.reset()
        self.memory = ReplayMemory(args.replay_size, args.seed, args)

        # set expert policy type and load expert parameters
        self.expert = load_expert(self.num_inputs, self.action_space, args, self.device)
        self.expert.to(self.device)
        self.evaluator = AlgoEvaluator(env, args)

        # set policy type
        self.policy = select_policy(self.num_inputs, self.action_space.shape[0], args.hidden_size, \
                action_space=self.action_space, model_type=args.model_type, bandwidth=args.bandwidth, \
                transform_rv=args.transform_dist, nonlin=args.nonlin, clamp=args.clamp, \
                static_cov=args.static_cov).to(self.device)

        # control mixing
        self.beta = self.args.beta
        self.beta_update = self.args.beta_update
        self.stochastic_interaction = self.args.stochastic_interaction
        self.episode = 1

        # logging info
        self.updates, self.interactions = 0, 0
        self.policy_return, self.policy_loss = None, None
        self.start = time.time()
        self.expert_return = AVG_EXPERT_RETURNS[args.env_name]
        self.expert_loss = self.evaluate_loss(env, use_expert=True, examples=10000)
        self.info = {}
        self.inclass_regret = 0
        self.optimal_regret = 0
        self.inclass_return = None
        self.optimal_return = None
        self.use_sgd = args.use_sgd
        self.early_stop_crit = args.early_stop_crit
        # quick checks
        assert args.samples_per_update <= args.replay_size
        assert not (args.transform_dist and args.use_exact)
        assert not ((not args.static_cov) and args.use_exact)

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

    # environment interactions
    def select_action(self, state, use_mean=False, use_expert=False):

        # use expert
        if use_expert:
            if use_mean:
                return self.expert.sample(state)[2]
            else:
                return self.expert.sample(state)[0]

        # sample expert
        if torch.rand(1)[0] <= self.beta:
            if use_mean:
                return self.expert.sample(state)[2]
            else:
                return self.expert.sample(state)[0]
        # sample policy data
        else:
            if use_mean is False:
                return self.policy.sample(state)[0]
            else:
                return self.policy.sample(state)[2]

        # return it all
        return action

    def gather_examples(self, env, memory, examples, use_expert=False, evaluate=False):
        # init
        states = torch.zeros( examples, self.num_inputs )
        expert_actions = torch.zeros( examples, self.action_space.shape[0] )
        rewards = torch.zeros( examples, 1 )
        dones = torch.zeros( examples, 1 )
        avg_reward, current_return, done_counter = 0, 0, 0
        e, done = 0, 0
        # do we want to interact unde the mode?
        det_policy = (not self.stochastic_interaction) or evaluate
        # state = self.current_state
        # set loop
        while e < examples:
            # grab whats needed from model
            tensor_state = torch.FloatTensor(self.current_state).to(self.device).unsqueeze(0)
            expert_action = self.select_action(tensor_state, use_mean=True, use_expert=True)
            tensor_action = self.select_action(tensor_state, use_mean=det_policy, use_expert=use_expert)
            action = tensor_action.detach().cpu().numpy()[0]
            # store what matters
            states[e,:] = tensor_state.squeeze(0)
            expert_actions[e,:] = expert_action
            # step the simulator
            next_state, reward, done, _ = env.step(action)
            # push to larger memory
            if memory is not None:
                memory.push(self.current_state, expert_action.detach().cpu().numpy()[0], reward, done)
            # grab flags and rewards
            dones[e,:] = done
            rewards[e,:] = reward
            current_return += reward
            # update the current state
            self.current_state = deepcopy(next_state)
            e += 1
            # check for reset
            if done:
                self.current_state = env.reset()
                avg_reward += current_return
                current_return = 0
                done_counter += 1
        avg_reward = avg_reward / max(1, done_counter)
        # return the tensors
        return memory, (states, expert_actions, rewards, dones), avg_reward

    # evaluation
    def evaluate_return(self, env, use_expert=False, duplicates=10):
        avg_reward = 0.
        for _  in range(duplicates):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                tensor_state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                tensor_action = self.select_action(tensor_state, use_mean=True, use_expert=use_expert)
                action = tensor_action.detach().cpu().numpy()[0]
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
            avg_reward += episode_reward
        avg_reward /= duplicates
        return avg_reward

    def evaluate_loss(self, env, use_expert=False, examples=1000):
        # grab examples and compute loss
        _, (states, expert_actions, _, _), avg_return = self.gather_examples(env, None, examples, use_expert)
        #
        states = states.to(self.device)
        expert_actions = expert_actions.to(self.device)
        #
        return self.compute_loss(states, expert_actions, use_expert).item()

    def evaluate(self, new_examples):
        #
        temp = deepcopy(self.beta)
        state = deepcopy(self.current_state)
        self.beta = 0.
        # evaluate performance vs expert
        with torch.no_grad():
            self.policy_return = self.evaluate_return(self.args.env_copy, duplicates=10)
            self.beta = temp
            self.current_state = self.args.env_copy.reset()
            self.policy_loss = self.evaluate_loss(self.args.env_copy, examples=1000)
            self.current_state = self.args.env_copy.reset()
            # self.regret += self.policy_loss - self.evaluate_loss(self.args.env_copy, use_expert=True, examples=1000)
        self.current_state = state
        # update evaluator module
        self.evaluator.evaluate(self.args.env_copy, self.memory, new_examples)
        # compute different types of regret
        self.inclass_regret += self.policy_loss - self.evaluator.inclass_loss
        self.optimal_regret += self.policy_loss - self.evaluator.optimal_loss
        self.inclass_return = self.evaluator.inclass_return
        self.optimal_return = self.evaluator.optimal_return

    # different losses for use in model update
    def compute_loss(self, state_batch, expert_action_batch, use_expert=False):

        if self.args.loss_type == 'l2':
            if use_expert:
                _, lp, policy_action = self.expert.sample(state_batch, reparam=False)
            else:
                _, lp, policy_action = self.policy.sample(state_batch, reparam=False)
            return (policy_action - expert_action_batch.detach()).pow(2).sum() + 0. * lp.sum()
        elif self.args.loss_type == 'l1':
            if use_expert:
                _, lp, policy_action = self.expert.sample(state_batch, reparam=False)
            else:
                _, lp, policy_action = self.policy.sample(state_batch, reparam=False)
            return (policy_action - expert_action_batch.detach()).abs().sum() + 0. * lp.sum()
        elif self.args.loss_type == 'bc':
            if use_expert:
                policy_action, _, _ = self.expert.sample(state_batch, reparam=True)
            else:
                policy_action, _, _ = self.policy.sample(state_batch, reparam=True)
            return (policy_action - expert_action_batch.detach()).pow(2).sum()
        elif self.args.loss_type == 'mle':
            if use_expert:
                log_liklihood = self.expert.log_prob(state_batch, expert_action_batch)
            else:
                log_liklihood = self.policy.log_prob(state_batch, expert_action_batch)
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
        new_examples = self.gather_examples(self.env, self.memory, self.samples, use_expert=True)
        self.evaluate(new_examples)
        self.log()
        self.display()

        # check that the model will not diverge
        with torch.autograd.set_detect_anomaly(True):
            self.update_parameters(new_examples)

        # with torch.autograd.detect_anomaly():
        for episode in range(self.args.episodes):

            # gather new loss and increment
            self.episode = episode + 1
            new_examples = self.gather_examples(self.env, self.memory, self.samples, use_expert=False)

            # evaluate + display
            if (self.episode % self.args.log_interval == 0):
                # evalaute current performance
                self.evaluate(new_examples)
                # log evaluation info to wandb
                self.log()
                # also display everything
                self.display()

            # update model parameters
            self.update_parameters(new_examples)

            # update beta
            self.beta *= self.beta_update

        # final log and display
        new_examples = self.gather_examples(self.env, self.memory, self.samples, use_expert=False)
        self.evaluate(new_examples)
        self.log()
        self.display()

    # multi-runner funcions
    def agent_update(self):

        # grab new losses
        new_examples = self.gather_examples(self.env, self.memory, self.samples, self.episode == 1)

        # evaluate + display
        if (self.episode % self.args.log_interval == 0):
            self.evaluate(new_examples)

        # algorithm update
        self.update_parameters(new_examples)

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
                   'optimal_return': self.optimal_return,
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
        self.samples = args.samples_per_update
        self.replay_size = 0

    def update_parameters(self, new_examples):
        # grab examples
        _, (states, expert_actions, _, _), self.avg_return = new_examples
        self.replay_size = self.args.samples_per_update
        self.interactions += self.args.samples_per_update
        #
        states = states.to(self.device)
        expert_actions = expert_actions.to(self.device)
        # compute loss
        loss = self.compute_loss(states, expert_actions) # / self.replay_size
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # backprop through computation graph
        loss.backward()
        grad_norm = torch.cat([param.grad.view(-1) for param in self.policy.parameters()]).data.detach().pow(2).mean()
        # step optimizer
        self.optimizer.step()
        self.updates += 1
        # store
        self.info = {'ogd_loss':  self.compute_loss(states, expert_actions),
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
        self.samples = args.samples_per_update
        self.replay_size = 0
        self.use_exact = args.use_exact
        self.lambda_reg = 1e-8

    def exact_step(self, tensor_states, tensor_expert_actions):
        # make sure we are in the correct regime
        assert self.policy.model_type != 'NNPolicy'
        assert self.policy.static_cov == 1
        assert self.loss_type in ['l2']
        # transform state
        state = self.policy.transform_state(tensor_states).to('cpu').double()
        state = torch.cat((state,torch.ones(state.size()[0], 1).double()), dim=1)
        # compute ls solution
        proj = torch.inverse(torch.mm(state.t(),state) + self.lambda_reg * torch.eye(state.size()[-1]).double())
        w = torch.mm(proj, torch.mm(state.t(),tensor_expert_actions.to('cpu').double()))
        weight, bias = w[:-1,:], w[-1,:]
        # set the weights
        with torch.no_grad():
            self.policy.mean_linear.weight.data = weight.t().float().to(tensor_states.device)
            self.policy.mean_linear.bias.data = bias.float().to(tensor_states.device)
        # compute grad
        grad = torch.mm(torch.mm(state.t(),state),w) + self.lambda_reg * w - torch.mm(state.t(), tensor_expert_actions.to('cpu').double())
        # compute loss check
        lse_loss = (torch.mm(state[:,:-1],weight).to(tensor_states.device) + bias.to(tensor_states.device) - tensor_expert_actions).pow(2).sum()
        network_loss = (self.policy(tensor_states)[0] - tensor_expert_actions).pow(2).sum()
        lib_loss = self.compute_loss(tensor_states, tensor_expert_actions)
        # store
        self.info = {'rlse_loss': network_loss,
                     'grad_norm': grad.pow(2).sum()}
        # return computed loss
        return lib_loss, grad.pow(2).mean()

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
        tensor_states = tensor_states.to(self.device)
        tensor_expert_actions = tensor_expert_actions.to(self.device)

        # generate memory permutation + data generator
        dataset = torch.utils.data.TensorDataset(tensor_states, tensor_expert_actions)
        self.data_generator = torch.utils.data.DataLoader(dataset, batch_size=self.args.mini_batch_size, shuffle=True)

        # iterative vs non iterative
        if self.use_exact:
            loss, grad_norm = self.exact_step(tensor_states, tensor_expert_actions)
            return loss

        # otherwise run iterative
        for epoch in range(self.epochs_per_update):
            # okey ...
            loss, grad_norm = self.iterative_step(tensor_states, tensor_expert_actions)
            self.updates += 1

            # early stopping crit
            if grad_norm.item() < self.early_stop_crit:
                break
        # print(loss, grad_norm, epoch)
        # store logging info
        self.info = {'ftl_loss': self.compute_loss(tensor_states, tensor_expert_actions) / self.replay_size,
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
        self.samples = args.samples_per_update
        self.replay_size = 0
        self.prev_grad_sum = None
        self.eta = torch.tensor([self.outer_lr]).to(self.device)

        self.epsilon = 1e-12

        # args to use classic version
        self.use_exact = args.use_exact
        self.sigma = torch.tensor([1/self.outer_lr]).to(self.device)

        # projected grad decent params
        self.ftrl_clip = args.ftrl_clip
        self.ftrl_variant = args.ftrl_variant
        self.squared_gradsum = 0.
        self.params_average = 0. * parameters_to_vector(self.policy.parameters()).detach()

    def reg_scaling(self):
        return self.outer_lr * np.sqrt(self.episode)

    def update_gradsum(self, tensor_states, tensor_expert_actions):
        # gather examples from memory
        states, expert_actions, _, _ = map(np.stack, zip(*self.memory.buffer))
        tensor_states = torch.stack([torch.tensor(s) for s in states]).float().detach().to(self.device)
        tensor_expert_actions = torch.stack([torch.tensor(s) for s in expert_actions]).float().detach().to(self.device)
        #
        dataset = torch.utils.data.TensorDataset(tensor_states, tensor_expert_actions)
        data_generator = torch.utils.data.DataLoader(dataset, batch_size=self.args.mini_batch_size, shuffle=True)
        #
        self.prev_grad_sum = 0. * parameters_to_vector(self.policy.parameters())
        # iterate over the entire memory
        for batch_states, batch_expert_actions in data_generator:
            # compute gradient for the sum of those function evals
            self.policy.zero_grad()
            loss = self.compute_loss(batch_states, batch_expert_actions)
            loss.backward()
            self.prev_grad_sum += self.compute_grad()
        # return it
        return self.prev_grad_sum

    def update_squared_gradsum(self, states, expert_actions):
        # compute MLE / loss explicitly
        loss =  self.compute_loss(states.to(self.device), expert_actions.to(self.device))
        # compute gradient for the sum of those function evals
        self.policy.zero_grad()
        loss.backward()
        # update to new sum of gradients
        self.squared_gradsum += self.compute_grad().detach().pow(2).sum()
        # return for use in eta update
        return self.squared_gradsum

    def step(self):

        #
        def closure(verbose=False, call_backward=False):

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # iterate over the entire memory and grab ftl loss
            cum_ftl_loss = torch.tensor(0.0).to(self.device)
            for batch_states, batch_expert_actions in self.data_generator:

                # accumulate gradients
                ftl_loss = self.compute_loss(batch_states.to(self.device), batch_expert_actions.to(self.device))
                ftl_loss = ftl_loss / self.replay_size
                if call_backward:
                    (ftl_loss).backward()
                cum_ftl_loss += ftl_loss.detach()

            # compute linearized term
            if self.prev_grad_sum is not None:
                lin_loss = -1 * torch.dot(self.prev_grad_sum.detach(), parameters_to_vector(self.policy.parameters())) / self.replay_size
            else:
                lin_loss =  0 * parameters_to_vector(self.policy.parameters()).mean()  / self.replay_size

            # compute trust region
            prev_parameter_vec = deepcopy(parameters_to_vector(self.prev_policy.parameters()).detach())
            parameter_vec = parameters_to_vector(self.policy.parameters())
            diff = (prev_parameter_vec-parameter_vec).pow(2)
            tr_loss = diff.sum()  / self.replay_size

            # compute
            loss = (lin_loss + self.reg_scaling() * tr_loss) + cum_ftl_loss

            #
            if call_backward:
                (loss).backward()

            if not verbose:
                return loss
            else:
                return cum_ftl_loss, \
                       lin_loss, \
                       tr_loss

        # step optimizer (do we need a closure?)
        if self.args.inner_policy_optim == 'LSOpt':
            # direct step
            loss = self.optimizer.step(closure)
            ftl_loss, lin_loss, tr_loss = closure(verbose=True, call_backward=True)
            grad_norm = self.compute_grad().detach().pow(2).mean()
        else:
            # backprop through computation graph
            loss = closure(call_backward=True)
            grad_norm = torch.cat([param.grad.view(-1) for param in self.policy.parameters()]).data.detach().pow(2).mean()
            self.optimizer.step()
            ftl_loss, lin_loss, tr_loss = closure(verbose=True)

        # return it
        return loss.detach(), grad_norm.detach(), (ftl_loss.detach(), lin_loss.detach(), tr_loss.detach())

    def sgd_step(self, batch_states, batch_expert_actions):

        #
        def closure(verbose=False, call_backward=False):

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # iterate over the entire memory and grab ftl loss
            cum_ftl_loss = torch.tensor(0.0).to(self.device)

            # accumulate gradients
            ftl_loss = self.compute_loss(batch_states.to(self.device), batch_expert_actions.to(self.device))
            ftl_loss = ftl_loss / batch_states.size()[0]
            if call_backward:
                (ftl_loss).backward()
            cum_ftl_loss += ftl_loss.detach()

            # compute linearized term
            if self.prev_grad_sum is not None:
                lin_loss = -1 * torch.dot(self.prev_grad_sum.detach(), parameters_to_vector(self.policy.parameters())) / batch_states.size()[0]
            else:
                lin_loss =  0 * parameters_to_vector(self.policy.parameters()).mean() / batch_states.size()[0]

            # compute trust region
            prev_parameter_vec = deepcopy(parameters_to_vector(self.prev_policy.parameters()).detach())
            parameter_vec = parameters_to_vector(self.policy.parameters())
            diff = (prev_parameter_vec-parameter_vec).pow(2)
            tr_loss = diff.sum() / batch_states.size()[0]

            # compute
            loss = (lin_loss + self.reg_scaling() * tr_loss) + cum_ftl_loss

            #
            if call_backward:
                (loss).backward()

            if not verbose:
                return loss
            else:
                return cum_ftl_loss, \
                       lin_loss, \
                       tr_loss

        # step optimizer (do we need a closure?)
        if self.args.inner_policy_optim == 'LSOpt':
            # direct step
            loss = self.optimizer.step(closure)
            ftl_loss, lin_loss, tr_loss = closure(verbose=True, call_backward=True)
            grad_norm = self.compute_grad().detach().pow(2).mean()
        else:
            # backprop through computation graph
            loss = closure(call_backward=True)
            grad_norm = torch.cat([param.grad.view(-1) for param in self.policy.parameters()]).data.detach().pow(2).mean()
            self.optimizer.step()
            ftl_loss, lin_loss, tr_loss = closure(verbose=True)

        # return it
        return loss.detach(), grad_norm.detach(), (ftl_loss.detach(), lin_loss.detach(), tr_loss.detach())

    def exact_update_helper(self, scaling, new_tensor_states, new_tensor_expert_actions):

        # compute the first term
        states, expert_actions, _, _ = map(np.stack, zip(*self.memory.buffer))
        X = torch.stack([torch.tensor(s) for s in states]).detach()
        b = torch.stack([torch.tensor(s) for s in expert_actions]).detach()
        X = self.policy.transform_state(X)
        X = torch.cat((X, torch.ones(X.size()[0], 1)), dim=1)
        precond_w = torch.inverse(torch.mm(X.t(), X) + self.sigma * scaling * torch.eye(X.size()[-1]))

        # compute the second term
        if self.episode > 1:
            w_k = torch.cat([self.policy.mean_linear.weight.data, self.policy.mean_linear.bias.data.unsqueeze(1)],dim=1).detach().t()
            states, expert_actions, _, _ = map(np.stack, zip(*self.old_memory.buffer))
            X = torch.stack([torch.tensor(s) for s in states]).detach()
            b = torch.stack([torch.tensor(s) for s in expert_actions]).detach()
            X = self.policy.transform_state(X)
            X = torch.cat((X, torch.ones(X.size()[0], 1)), dim=1)
            precond_w_k = torch.mm(X.t(), X) + self.sigma * scaling * torch.eye(X.size()[-1])
            residual_1 = torch.mm(precond_w_k.double(), w_k.double())
        else:
            w_k = torch.cat([self.policy.mean_linear.weight.data, self.policy.mean_linear.bias.data.unsqueeze(1)],dim=1).detach().t()
            residual_1 = 0 * torch.mm(precond_w.double(), w_k.double())

        # compute the third term
        X = self.policy.transform_state(new_tensor_states)
        X = torch.cat((X, torch.ones(X.size()[0], 1)), dim=1)
        b = new_tensor_expert_actions
        residual_2 = torch.mm(X.t(), b)

        # update w
        w = torch.mm(precond_w.double(), residual_1.double() + residual_2.double())

        # return it
        return w, residual_1, residual_2

    def exact_step(self, new_tensor_states, new_tensor_expert_actions):

        # make sure we are in the correct regime
        assert self.policy.model_type != 'NNPolicy'
        assert self.policy.static_cov == 1
        assert self.loss_type == 'l2'

        # pick ftrl_variant
        scaling = self.reg_scaling()

        # compute update with desired scaling
        w, residual_1, residual_2 = self.exact_update_helper(scaling, new_tensor_states, new_tensor_expert_actions)

        # split into weights and biases
        weight, bias = w[:-1,:].to(self.device).float(), w[-1,:].to(self.device).float()
        weight_size = self.policy.mean_linear.weight.data.size()
        bias_size = self.policy.mean_linear.bias.data.size()

        # set the weights
        with torch.no_grad():
            self.policy.mean_linear.weight.data = weight.t().float().reshape(weight_size).to(self.device)
            self.policy.mean_linear.bias.data = bias.float().reshape(bias_size).to(self.device)

        # re-gen info for final evaluation
        states, expert_actions, _, _ = map(np.stack, zip(*self.memory.buffer))
        X = torch.stack([torch.tensor(s) for s in states]).detach().float().to(self.device)
        b = torch.stack([torch.tensor(s) for s in expert_actions]).detach().float().to(self.device)

        # look at the different losses
        ftl_loss = self.compute_loss(X, b)
        linear_loss = torch.tensor(0.) if self.episode == 1 else torch.dot(self.prev_grad_sum.detach(), parameters_to_vector(self.policy.parameters()))
        trust_region_loss = self.sigma * scaling * (parameters_to_vector(self.prev_policy.parameters()) - \
                                parameters_to_vector(self.policy.parameters())).pow(2).sum()

        # compute gradient after prescision drop
        X = self.policy.transform_state(X)
        X = torch.cat((X, torch.ones(X.size()[0], 1)), dim=1)
        grad = torch.mm(torch.mm(X.t(), X) + self.sigma * scaling * torch.eye(X.size()[-1]), w.float()) - (residual_1+residual_2).float()
        grad_norm = grad.pow(2).sum()

        # store
        self.info = {'rlse_loss': (ftl_loss + linear_loss + trust_region_loss).item(),
                     'ftl_loss': ftl_loss.item(),
                     'linear_loss': linear_loss.item(),
                     'trust_region_loss': trust_region_loss.item(),
                     'grad_norm': grad_norm.item()}

        # return computed loss
        return ftl_loss + linear_loss + trust_region_loss, grad.pow(2).sum()

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
        tensor_states = tensor_states.to(self.device)
        tensor_expert_actions = tensor_expert_actions.to(self.device)

        # generate memory permutation + data generator
        dataset = torch.utils.data.TensorDataset(tensor_states, tensor_expert_actions)
        self.data_generator = torch.utils.data.DataLoader(dataset, batch_size=self.args.mini_batch_size, shuffle=True)

        # train model with exact
        if self.use_exact:
            loss, grad_norm = self.exact_step(new_states, new_expert_actions)

        # or use a decent method
        else:
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

                # check for full batch or sgd
                if not self.use_sgd:
                    loss, grad_norm, (ftl_loss, lin_loss, tr_loss) = self.step()
                else:
                    for batch_states, batch_expert_actions in self.data_generator:
                        loss, grad_norm, (ftl_loss, lin_loss, tr_loss) = self.sgd_step(batch_states, batch_expert_actions)
                # print(timer(start,time.time()))
                # take steps
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
    def reg_scaling(self):
        return self.outer_lr * self.squared_gradsum.pow(0.5).detach()

class DAFTRL(AFTRL):

        def __init__(self, env, args):
            super(DAFTRL,self).__init__(env, args)
            self.squared_gradsum = 0. * parameters_to_vector(self.policy.parameters())
            self.algo = 'DAFTRL'
        def update_squared_gradsum(self, states, expert_actions):
            # compute MLE / loss explicitly
            loss =  self.compute_loss(states.to(self.device), expert_actions.to(self.device))
            # compute gradient for the sum of those function evals
            self.policy.zero_grad()
            loss.backward()
            # update to new sum of gradients
            self.squared_gradsum += self.compute_grad().detach().pow(2)
            # return for use in eta update
            return self.squared_gradsum

        def step(self):

            #
            def closure(verbose=False, call_backward=False):

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # iterate over the entire memory and grab ftl loss
                cum_ftl_loss = torch.tensor(0.0)
                for batch_states, batch_expert_actions in self.data_generator:
                    # accumulate gradients
                    ftl_loss = self.compute_loss(batch_states.to(self.device), batch_expert_actions.to(self.device))
                    ftl_loss = ftl_loss / self.replay_size
                    if call_backward:
                        (ftl_loss).backward()
                    cum_ftl_loss += (ftl_loss).cpu().detach()

                # compute linearized term
                if self.prev_grad_sum is not None:
                    lin_loss = -1 * torch.dot(self.prev_grad_sum.detach(), parameters_to_vector(self.policy.parameters())) / self.replay_size
                else:
                    lin_loss =  0 * parameters_to_vector(self.policy.parameters()).mean()  / self.replay_size

                # compute trust region
                prev_parameter_vec = deepcopy(parameters_to_vector(self.prev_policy.parameters()).detach())
                parameter_vec = parameters_to_vector(self.policy.parameters())
                diff = (prev_parameter_vec-parameter_vec).pow(2)
                tr_loss = diff * self.squared_gradsum.pow(0.5).detach()
                tr_loss = self.outer_lr * torch.dot(tr_loss, diff)

                # compute
                loss = (lin_loss + tr_loss) + cum_ftl_loss

                #
                if call_backward:
                    (loss).backward()

                if not verbose:
                    return loss
                else:
                    return cum_ftl_loss, \
                           lin_loss, \
                           tr_loss

            # step optimizer (do we need a closure?)
            if self.args.inner_policy_optim == 'LSOpt':
                # direct step
                loss = self.optimizer.step(closure)
                ftl_loss, lin_loss, tr_loss = closure(verbose=True, call_backward=True)
                grad_norm = self.compute_grad().detach().pow(2).mean()
            else:
                # backprop through computation graph
                loss = closure(call_backward=True)
                grad_norm = torch.cat([param.grad.view(-1) for param in self.policy.parameters()]).data.detach().pow(2).mean()
                self.optimizer.step()
                ftl_loss, lin_loss, tr_loss = closure(verbose=True)

            # return it
            return loss.detach(), grad_norm.detach(), (ftl_loss.detach(), lin_loss.detach(), tr_loss.detach())

class SFTRL(FTRL):

    def __init__(self, env, args):
        super(SFTRL,self).__init__(env, args)
        self.algo = 'SFTRL'

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
            # regularization
            current_params = parameters_to_vector(self.policy.parameters())
            lin_loss = torch.dot(current_params,current_params)
            # divide this by length of all data
            lin_loss = self.outer_lr * (np.sqrt(self.episode) - 1) * lin_loss / self.replay_size
            # compute trust region
            tr_loss =  self.outer_lr * torch.dot(current_params, self.params_average) / self.replay_size
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
            grad_norm = torch.cat([param.grad.view(-1) for param in self.policy.parameters()]).data.detach().pow(2).mean()
            self.optimizer.step()
            ftl_loss, lin_loss, tr_loss = closure(verbose=True)

        # return it
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
            lin_loss = self.outer_lr * (np.sqrt(self.episode) - 1) * lin_loss /  batch_states.size()[0]
            # compute trust region
            tr_loss =  self.outer_lr * torch.dot(current_params, self.params_average) / batch_states.size()[0]
            # compute
            loss = (ftl_loss.detach() + lin_loss + tr_loss).mean()
            #
            if call_backward:
                loss.backward()
            # return it
            if not verbose:
                return loss
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
            grad_norm = torch.cat([param.grad.view(-1) for param in self.policy.parameters()]).data.detach().pow(2).mean()
            self.optimizer.step()
            ftl_loss, lin_loss, tr_loss = closure(verbose=True)

        # return it
        return loss.detach(), grad_norm.detach(), (ftl_loss.detach(), lin_loss.detach(), tr_loss.detach())
