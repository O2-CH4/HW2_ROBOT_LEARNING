from collections import OrderedDict
from gym import wrappers

# hw1 imports
from hw1.roble.infrastructure.rl_trainer import RL_Trainer
from hw1.roble.infrastructure import pytorch_util as ptu

# hw2 imports
from hw2.roble.agents.mb_agent import MBAgent
from hw2.roble.infrastructure import utils
# register all of our envs
from hw2.roble.envs import register_envs

import gym
import numpy as np
import pickle
import os
import sys
import time
import torch

register_envs()

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 1
MAX_VIDEO_LEN = 40 # we overwrite this in the code below

class RL_Trainer(RL_Trainer):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, params , agent_class):

        # Inherit from hw1 RL_Trainer
        super().__init__(params, agent_class)

        # Make the gym environment
        print (self._env) 
        self.add_wrappers()

        # Are the observations images?
        img = len(self._env.observation_space.shape) > 2

        # Observation and action sizes

        ob_dim = self._env.observation_space.shape if img else self._env.observation_space.shape[0]
        ac_dim = self._env.action_space.n if self._params['alg']['discrete'] else self._env.action_space.shape[0]
        self._params['alg']['ac_dim'] = ac_dim
        self._params['alg']['ob_dim'] = ob_dim

    def add_wrappers(self):
        if 'env_wrappers' in self._params:
            # These operations are currently only for Atari envs
            self._env = wrappers.Monitor(self._env, os.path.join(self._params['logging']['logdir'], "gym"), force=True)
            self._env = self._params['env_wrappers'](self._env)
            self._mean_episode_reward = -float('nan')
            self._best_mean_episode_reward = -float('inf')
        if 'non_atari_colab_env' in self._params and self._params['logging']['video_log_freq'] > 0:
            self._env = wrappers.Monitor(self._env, os.path.join(self._params['logging']['logdir'], "gym"), force=True)
            self._mean_episode_reward = -float('nan')
            self._best_mean_episode_reward = -float('inf')
        
        
    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        """

        # init vars at beginning of training
        self._total_envsteps = 0
        self._total_train_rewards = 0
        self._total_eval_rewards = 0
        self._start_time = time.time()

        print_period = 1

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self._params['logging']['video_log_freq'] == 0 and self._params['logging']['video_log_freq'] != -1:
                self._log_video = True
            else:
                self._log_video = False

            # decide if metrics should be logged
            if self._params['logging']['scalar_log_freq'] == -1:
                self._logmetrics = False
            elif itr % self._params['logging']['scalar_log_freq'] == 0:
                self._logmetrics = True
            else:
                self._logmetrics = False

            use_batchsize = self._params['alg']['batch_size']
            if itr == 0:
                use_batchsize = self._params['alg']['batch_size_initial']
            paths, envsteps_this_batch, train_video_paths = (
                self.collect_training_trajectories(
                    itr, initial_expertdata, collect_policy, use_batchsize)
            )

            self._total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            if isinstance(self._agent, MBAgent):
                self._agent.add_to_replay_buffer(paths, self._params['alg']['add_sl_noise'])
            else:
                self._agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            all_logs = self.train_agent()

            # if there is a model, log model predictions
            if isinstance(self._agent, MBAgent) and itr == 0:
                self.log_model_predictions(itr, all_logs)

            # log/save
            if self._log_video or self._logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

                if self._params['logging']['save_params']:
                    self._agent.save('{}/agent_itr_{}.pt'.format(self._params['logging']['logdir'], itr))
        
        return self._logger.get_table_dict()

    def collect_training_trajectories(
            self,
            itr,
            load_initial_expertdata,
            collect_policy,
            batch_size,
    ):
        from hw2.roble.infrastructure import utils
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        if itr == 0:
            if load_initial_expertdata:
                paths = pickle.load(open(self._params['env']['expert_data'], 'rb'))
                return paths, 0, None
            else:
                num_transitions_to_sample = self._params['alg']['batch_size_initial']
        else:
            num_transitions_to_sample = self._params['alg']['batch_size']
    
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(
            self._env, collect_policy, num_transitions_to_sample, self._params['env']['max_episode_length'])
        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN

        train_video_paths = None
        if self._log_video:
            print('\nCollecting train rollouts to be used for saving videos...')

            train_video_paths = utils.sample_n_trajectories(self._env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
        return paths, envsteps_this_batch, train_video_paths

    ####################################
    ####################################

    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self._params['alg']['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self._agent.sample(self._params['alg']['train_batch_size'])
            train_log = self._agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################
    

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self._env, eval_policy, 
                                                                         self._params['alg']['eval_batch_size'], 
                                                                         self._params['env']['max_episode_length'])

        # save eval rollouts as videos in the video folder (for grading)
        if self._log_video:
            if train_video_paths is not None:
                #save train/eval videos
                print('\nSaving train rollouts as videos...')
                self._logger.log_paths_as_videos(train_video_paths, itr, fps=self._fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self._env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
            print('\nSaving eval rollouts as videos...')
            self._logger.log_paths_as_videos(eval_video_paths, itr, fps=self._fps,max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')
        #######################

        # save eval metrics
        if self._logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self._total_envsteps
            logs["Train_AverageRewardSoFar"] = self._total_train_rewards / self._total_envsteps
            logs["Eval_AverageRewardSoFar"] = self._total_eval_rewards / self._total_envsteps
            logs["TimeSinceStart"] = time.time() - self._start_time
            logs.update(last_log)

            if itr == 0:
                self._initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self._initial_return

            # perform the logging
            # for key, value in logs.items():
            #     print('{} : {}'.format(key, value))
            #     self._logger.log_file(value, key,itr)
            #     self._logger.log_scalar(value, key, itr)
                
            self._logger.record_dict(logs, prefix='trainer/')
            self._logger.dump_tabular(with_prefix=False, with_timestamp=False)
            print('Done logging MBRL...\n\n')

    def log_model_predictions(self, itr, all_logs):
        # model predictions
        import matplotlib.pyplot as plt
        self._fig = plt.figure()

        # sample actions
        action_sequence = self._agent._actor.sample_action_sequences(num_sequences=1, horizon=10) #20 reacher
        action_sequence = action_sequence[0]

        # calculate and log model prediction error
        mpe, true_states, pred_states = utils.calculate_mean_prediction_error(self._env, action_sequence, 
                                                                              self._agent._dyn_models, self._agent._actor._data_statistics)
        print("assert:", self._params['alg']['ob_dim'], " == " , true_states.shape[1], " == ", pred_states.shape[1])
        assert self._params['alg']['ob_dim'] == true_states.shape[1] == pred_states.shape[1]
        ob_dim = self._params['alg']['ob_dim']
        ob_dim = 2*int(ob_dim/2.0) ## skip last state for plotting when state dim is odd

        # plot the predictions
        self._fig.clf()
        for i in range(ob_dim):
            plt.subplot(ob_dim/2, 2, i+1)
            plt.plot(true_states[:,i], 'g')
            plt.plot(pred_states[:,i], 'r')
        self._fig.suptitle('MPE: ' + str(mpe))
        self._fig.savefig(self._params['logging']['logdir']+'/itr_'+str(itr)+'_predictions.png', dpi=200, bbox_inches='tight')

        # plot all intermediate losses during this iteration
        all_losses = np.array([log['Training Loss'] for log in all_logs])
        np.save(self._params['logging']['logdir']+'/itr_'+str(itr)+'_losses.npy', all_losses)
        self._fig.clf()
        plt.plot(all_losses)
        self._fig.savefig(self._params['logging']['logdir']+'/itr_'+str(itr)+'_losses.png', dpi=200, bbox_inches='tight')

