"""Keep SAPG evaluation on the same policy branch for every environment."""


def use_evaluation_policy(player, policy_id=50.0):
    # BasePlayer otherwise distributes linspace(50, 0, num_envs). Those IDs
    # usually do not match the six learned rows; unknown IDs fall back to row 0.
    # Match the single-env viewer and deployment wrapper explicitly.
    if 'learn_param' in getattr(player, 'expl_type', ''):
        ids = player.model.a2c_network.param_ids
        if not bool((ids == policy_id).any()):
            raise ValueError(f'Policy coefficient {policy_id} absent from checkpoint model')
        player.intr_reward_coef_embd.fill_(policy_id)
