# hw1 imports
from hw1.roble.policies.base_policy import BasePolicy

import numpy as np

class MPCPolicy(BasePolicy):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self,
                 env,
                 dyn_models,
                 mpc_horizon,
                 mpc_num_action_sequences,
                 mpc_action_sampling_strategy='random',
                 **kwargs
                 ):
        super().__init__()

        # init vars
        self._data_statistics = None  # NOTE must be updated from elsewhere

        # action space
        self._ac_space = self._env.action_space
        self._low = self._ac_space.low
        self._high = self._ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert self._mpc_action_sampling_strategy in allowed_sampling, f"self._mpc_action_sampling_strategy must be one of the following: {allowed_sampling}"

        print(f"Using action sampling strategy: {self._mpc_action_sampling_strategy}")
        if self._mpc_action_sampling_strategy == 'cem':
            print(f"CEM params: alpha={self._cem_alpha}, "
                + f"num_elites={self._cem_num_elites}, iterations={self._cem_iterations}")

    def get_random_actions(self, num_sequences, horizon):
       return np.random.uniform(low=self._low, high=self._high,
						size=(num_sequences, horizon, self._ac_dim))
						
    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self._mpc_action_sampling_strategy == 'random' \
            or (self._mpc_action_sampling_strategy == 'cem' and obs is None):
            # DONE (Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self._ac_dim) in the range [self._low, self._high]
            return self.get_random_actions(num_sequences, horizon)
        
        elif self._mpc_action_sampling_strategy == 'cem':
            # DONE(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf

            actions = self.get_random_actions(num_sequences, horizon)
            elite_mean, elite_std = np.zeros(actions.shape[1:]), np.zeros(actions.shape[1:])

            for i in range(self._cem_iterations):
                if i > 0:
                    actions = np.random.normal(elite_mean, elite_std, size=(num_sequences, *elite_mean.shape))
                rewards = self.evaluate_candidate_sequences(actions, obs)
                sorted_idxs = sorted(range(len(actions)), key=lambda i: rewards[i])
                elites = actions[sorted_idxs][-self._cem_num_elites:]
                if i == 0:
                    elite_mean, elite_std = np.mean(elites, axis=0), np.std(elites, axis=0)
                else:
                    elite_mean = self._cem_alpha * np.mean(elites, axis=0) + (1 - self._cem_alpha) * elite_mean
                    elite_std = self._cem_alpha * np.std(elites, axis=0) + (1 - self._cem_alpha) * elite_std
            return elite_mean[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self._mpc_action_sampling_strategy}")

                # - Sample candidate sequences from a Gaussian with the current
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self._cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance
                
                # DONE(Q5): Set `cem_action` to the appropriate action sequence chosen by CEM.
                # The shape should be (horizon, self._ac_dim)  
                #     cem_action = None
                #     return cem_action[None]
                # else:
                #     raise Exception(f"Invalid sample_strategy: {self._mpc_action_sampling_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # DONE (Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)
    
        # for each model in ensemble:
        predicted_sum_of_rewards_per_model = []
        for model in self._dyn_models:
            sum_of_rewards = self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            predicted_sum_of_rewards_per_model.append(sum_of_rewards)

        # calculate mean_across_ensembles (predicted rewards)
        return np.mean(predicted_sum_of_rewards_per_model, axis=0)  


    def get_action(self, obs):
        if self._data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self._mpc_num_action_sequences, horizon=self._mpc_horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = candidate_action_sequences[predicted_rewards.argmax()]  # DONE (Q2)
            action_to_take = best_action_sequence[0]  # DONE (Q2)
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        N, H, _ = candidate_action_sequences.shape
        pred_obs = np.zeros((N, H, self._ob_dim))
        pred_obs[:, 0] = np.tile(obs[None, :], (N, 1))
        rewards = np.zeros((N, H))
        for s in range(H):
            ob = pred_obs[:, s]
            ac = candidate_action_sequences[:, s]
            rewards[:, s], _  = self._env.get_reward(ob, ac)
            if s < H - 1:
                ob = pred_obs[:, s]
                ac = candidate_action_sequences[:, s]
                pred_obs[:, s + 1] = model.get_prediction(ob, ac, self._data_statistics)

        sum_of_rewards = np.sum(rewards,axis=1)
        return sum_of_rewards
    

        # # DONE (Q2)
        # # For each candidate action sequence, predict a sequence of
        # # states for each dynamics model in your ensemble.
        # # Once you have a sequence of predicted states from each model in
        # # your ensemble, calculate the sum of rewards for each sequence
        # # using `self._env.get_reward(predicted_obs, action)` at each step.
        # # You should sum across `self._mpc_horizon` time step.
        # # Hint: you should use model.get_prediction and you shouldn't need
        # #       to import pytorch in this file.
        # # Hint: Remember that the model can process observations and actions
        # #       in batch, which can be much faster than looping through each
        # #       action sequence.
        # return sum_of_rewards
