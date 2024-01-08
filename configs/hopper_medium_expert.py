from copy import deepcopy
from configs.default import default_args

hopper_medium_expert_args = deepcopy(default_args)
hopper_medium_expert_args["rollout_length"] = 10
hopper_medium_expert_args["penalty_coef"] = 1
hopper_medium_expert_args["deterministic_backup"] = True
hopper_medium_expert_args["max_epochs_since_update"] = 30