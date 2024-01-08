import os
import argparse
import gym
import torch
import d4rl
import numpy as np
import pandas as pd

from policies import MICROPolicy
from configs import loaded_args
from models.nets import MLP
from models.actor_critic import ActorProb, Critic
from models.dist import TanhDiagGaussian
from models.dynamics_model import EnsembleDynamicsModel
from dynamics import EnsembleDynamics
from utils.scaler import StandardScaler
from utils.termination_fns import get_termination_fn
from utils.policy_evaluation import PolicyEvaluation
from robust.envs import get_new_friction_env, get_new_gravity_env, get_new_env
from robust.attacker import Evaluation_Attacker
from robust.data_mean_std import get_obs_mean_std


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mobile")
    parser.add_argument("--task", type=str, default="walker2d-medium-expert-v2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--choice", type=str, default="attack") # attack,sim2real
    parser.add_argument("--attack_mode", type=str, default="random") # random, action_diff, min_Q
    parser.add_argument("--policy_path", type=str, default=None) # load policy

    known_args, _ = parser.parse_known_args()
    default_args = loaded_args[known_args.task]
    for arg_key, default_value in default_args.items():
        parser.add_argument(f'--{arg_key}', default=default_value, type=type(default_value))

    return parser.parse_args()

def evaluate(args=get_args()):
    env = gym.make(args.task)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    critics = []
    for i in range(args.num_q_ensemble):
        critic_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
        critics.append(Critic(critic_backbone, args.device))
    critics = torch.nn.ModuleList(critics)
    critics_optim = torch.optim.Adam(critics.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )

    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn
    )

    # create policy
    policy = MICROPolicy(
        dynamics,
        actor,
        critics,
        actor_optim,
        critics_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        penalty_coef=args.penalty_coef,
        num_samples=args.num_samples,
        deterministic_backup=args.deterministic_backup,
        max_q_backup=args.max_q_backup
    )

    # evaluation
    _, obs_std = get_obs_mean_std(args.task)
    log_dirs = os.path.join("log", args.task)
    if  not os.path.exists(log_dirs):
        os.makedirs(log_dirs)

    norm_ep_rew_mean = []
    norm_ep_rew_std = []

    if args.choice =="attack":
        env = get_new_friction_env(1.0, args.task)
        scale = np.linspace(0, 0.2, 10)
        for index in range(len(scale)):
            attack = Evaluation_Attacker(
                policy,
                eps = scale[index] ,
                obs_dim = np.prod(args.obs_shape),
                action_dim=args.action_dim,
                obs_std = obs_std,
                attack_mode=args.attack_mode,
                device=args.device)

            Policy_Evaluate = PolicyEvaluation(
                policy=policy,
                eval_env=env,
                path = args.policy_path,
                eval_episodes=args.eval_episodes,
                attack = attack
            )
            ep_reward_mean, ep_reward_std = Policy_Evaluate.evaluate()
            norm_ep_rew_mean.append(ep_reward_mean)
            norm_ep_rew_std.append(ep_reward_std)
        dataframe = pd.DataFrame({'scale': scale, 'mean': norm_ep_rew_mean, 'std': norm_ep_rew_std})
        dataframe.to_csv(log_dirs+"/{}.csv".format(args.attack_mode),index=False, sep=',')

    elif args.choice =="sim2real":
        import csv
        degree_friction = np.linspace(0.5, 1.5, 10)
        degree_gravity = np.linspace(0.5, 5, 10)
        with open(log_dirs+"/result.csv", "w") as fw:
            writer = csv.writer(fw)
            for i in range(len(degree_friction)):
                variety_friction = degree_friction[i]
                norm_ep_rew_mean=[]
                for j in range(len(degree_gravity)):
                    variety_gravity = degree_gravity[j]
                    env = get_new_env(variety_friction, variety_gravity,args.task)
                    Policy_Evaluate = PolicyEvaluation(
                        policy=policy,
                        eval_env=env,
                        path=args.policy_path,
                        eval_episodes=args.eval_episodes
                    )
                    ep_reward_mean, ep_reward_std = Policy_Evaluate.evaluate()
                    norm_ep_rew_mean.append(ep_reward_mean)
                writer.writerow(norm_ep_rew_mean)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    evaluate()