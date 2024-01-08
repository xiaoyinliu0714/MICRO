import numpy as np
import torch
from torch.distributions import kl_divergence


def get_policy_kl(policy, observation, noised_obs):
    action_dist = policy.policy_dist(observation)
    noised_action_dist = policy.policy_dist(noised_obs)
    kl_loss = kl_divergence(action_dist, noised_action_dist).sum(axis=-1) \
    + kl_divergence(noised_action_dist,action_dist).sum(axis=-1)

    return kl_loss

class Evaluation_Attacker:
    def __init__(self, policy, eps, obs_dim, action_dim, obs_std=None, attack_mode='random', num_samples=50,
                 device='cuda'):
        self.policy = policy
        self.eps = eps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.attack_mode = attack_mode
        self.num_samples = num_samples
        self.device = device
        self.obs_std = self.from_numpy(obs_std) if obs_std is not None else torch.ones(1, self.obs_dim,
                                                                                       device=self.device)

    def from_numpy(self, *args, **kwargs):
        return torch.from_numpy(*args, **kwargs).float().to(self.device)

    def get_numpy(self, tensor):
        return tensor.to('cpu').detach().numpy()

    def sample_random(self, size):
        return 2 * self.eps * self.obs_std * (torch.rand(size, self.obs_dim, device=self.device) - 0.5)

    def noise_action_diff(self, observation, M):
        observation = observation.reshape(M, self.obs_dim)
        size = self.num_samples  # for zero order

        def _loss_action(observation, para):
            noised_obs = observation + para
            return - get_policy_kl(self.policy, observation, noised_obs)

        if self.attack_mode == 'action_diff':
            delta_s = self.sample_random(size).reshape(1, size, self.obs_dim).repeat(M, 1, 1).reshape(-1, self.obs_dim)
            tmp_obs = observation.reshape(-1, 1, self.obs_dim).repeat(1, size, 1).reshape(-1, self.obs_dim)
            with torch.no_grad():
                kl_loss = _loss_action(tmp_obs, delta_s)
                max_id = torch.argmin(kl_loss.reshape(M, size), axis=1)
            noise_obs_final = self.get_numpy(delta_s.reshape(M, size, self.obs_dim)[np.arange(M), max_id])
        else:
            raise NotImplementedError

        return self.get_numpy(observation) + noise_obs_final

    def noise_min_Q(self, observation, M):
        observation = observation.reshape(M, self.obs_dim)
        size = self.num_samples
        weight_std = 10

        def _loss_Q(observation, para):
            noised_obs = observation + para
            pred_actions, _ = self.policy.actforward(noised_obs, deterministic=True)
            return self.policy.q_fun(observation, pred_actions)

        def _loss_Q_std(observation, para):
            Q_loss = _loss_Q(observation, para)
            Q_std = Q_loss.std(axis=0).reshape(1, -1, 1)

            return - weight_std * Q_std  

        loss_fun = _loss_Q_std if 'std' in self.attack_mode else _loss_Q

        if self.attack_mode == 'min_Q' or self.attack_mode == 'min_Q_std':
            delta_s = self.sample_random(size + 1)
            delta_s[-1, :] = torch.zeros((1, self.obs_dim), device=self.device)
            delta_s = delta_s.reshape(1, size + 1, self.obs_dim).repeat(M, 1, 1).reshape(-1, self.obs_dim)
            tmp_obs = observation.reshape(-1, 1, self.obs_dim).repeat(1, size + 1, 1).reshape(-1, self.obs_dim)
            noised_qs_pred = loss_fun(tmp_obs, delta_s).mean(axis=0).reshape(-1, size + 1)
            min_id = torch.argmin(noised_qs_pred, axis=1)
            noise_obs_final = self.get_numpy(delta_s).reshape(M, size + 1, self.obs_dim)[np.arange(M), min_id]
        else:
            raise NotImplementedError

        return self.get_numpy(observation) + noise_obs_final

    def attack_obs(self, observation):
        M = observation.shape[0] if len(observation.shape) == 2 else 1
        observation = self.from_numpy(observation)

        if self.attack_mode == 'random':
            delta_s = self.sample_random(M)
            noised_observation = observation.reshape(M, self.obs_dim) + delta_s
            noised_observation = self.get_numpy(noised_observation)

        elif 'action_diff' in self.attack_mode:
            noised_observation = self.noise_action_diff(observation, M)

        elif 'min_Q' in self.attack_mode:
            noised_observation = self.noise_min_Q(observation, M)

        else:
            raise NotImplementedError

        return noised_observation