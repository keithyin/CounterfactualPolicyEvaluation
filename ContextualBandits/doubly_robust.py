
def doubly_robust(dataset):
    assert isinstance(dataset, list)
    assert len(dataset) > 0, ""
    tot_scores = 0
    tot_ins = 0
    for (true_action_reward, dm_action_reward, new_policy_choose_action_prob,
         old_policy_choose_action_prob, dm_new_action_reward) in dataset:
        tot_ins += 1
        tot_scores += doubly_robust_core(true_action_reward, dm_action_reward, new_policy_choose_action_prob,
                                         old_policy_choose_action_prob,
                                         dm_new_action_reward)

    return tot_scores / tot_ins


def doubly_robust_core(true_action_reward,
                       dm_action_reward,
                       new_policy_choose_action_prob,
                       old_policy_choose_action_prob,
                       dm_new_action_reward):
    return (true_action_reward - dm_action_reward) * new_policy_choose_action_prob / old_policy_choose_action_prob \
           + dm_new_action_reward
