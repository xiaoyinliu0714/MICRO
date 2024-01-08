from .halfcheetah_medium import halfcheetah_medium_args
from .hopper_medium import hopper_medium_args
from .walker2d_medium import walker2d_medium_args
from .halfcheetah_medium_replay import halfcheetah_medium_replay_args
from .hopper_medium_replay import hopper_medium_replay_args
from .walker2d_medium_replay import walker2d_medium_replay_args
from .halfcheetah_medium_expert import halfcheetah_medium_expert_args
from .hopper_medium_expert import hopper_medium_expert_args
from .walker2d_medium_expert import walker2d_medium_expert_args

loaded_args = {
    "halfcheetah-medium-v2": halfcheetah_medium_args,
    "hopper-medium-v2": hopper_medium_args,
    "walker2d-medium-v2": walker2d_medium_args,
    "halfcheetah-medium-replay-v2": halfcheetah_medium_replay_args,
    "hopper-medium-replay-v2": hopper_medium_replay_args,
    "walker2d-medium-replay-v2": walker2d_medium_replay_args,
    "halfcheetah-medium-expert-v2": halfcheetah_medium_expert_args,
    "hopper-medium-expert-v2": hopper_medium_expert_args,
    "walker2d-medium-expert-v2": walker2d_medium_expert_args,

}