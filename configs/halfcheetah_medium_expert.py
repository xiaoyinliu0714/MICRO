from copy import deepcopy
from configs.default import default_args

halfcheetah_medium_expert_args = deepcopy(default_args)
halfcheetah_medium_expert_args["rollout_length"] = 5
halfcheetah_medium_expert_args["penalty_coef"] = 0.5
halfcheetah_medium_expert_args["real_ratio"] = 0.5
halfcheetah_medium_expert_args["deterministic_backup"] = True
halfcheetah_medium_expert_args["max_epochs_since_update"] = 30