import gym
from gym.spaces import Box, Discrete, Tuple
from robust.utils import update_target_env_gravity, update_target_env_friction, update_source_env,update_target_env

def get_source_env(env_name="walker2d-v2"):
    update_source_env(env_name)
    env = gym.make(env_name)

    return env

def get_new_gravity_env(variety, env_name):
    update_target_env_gravity(variety, env_name)
    env = gym.make(env_name)

    return env

def get_new_friction_env(variety, env_name):
    update_target_env_friction(variety, env_name)
    env = gym.make(env_name)

    return env

def get_new_env(variety_friction, variety_gravity, env_name):
    update_target_env(variety_friction, variety_gravity, env_name)
    env = gym.make(env_name)

    return env

def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))