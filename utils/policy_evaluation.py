import torch
import gym
import numpy as np
from typing import Optional, Dict, List
from policies import BasePolicy


# model-based policy trainer
class PolicyEvaluation:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        path: str,
        eval_episodes: int = 10,
        attack = None
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self._eval_episodes = eval_episodes
        self.path = path
        self.attack = attack

    def evaluate(self) -> Dict[str, List[float]]:
        self.policy.load_state_dict(torch.load(self.path))
        self.policy.eval()

        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            if self.attack is not None:
                obs = self.attack.attack_obs(obs)
            action = self.policy.select_action(obs, deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        eval_info = {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

        ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(
            eval_info["eval/episode_reward"])
        reward_mean, reward_std = self.eval_env.get_normalized_score(ep_reward_mean) * 100, self.eval_env.get_normalized_score(
            ep_reward_std) * 100

        return reward_mean, reward_std