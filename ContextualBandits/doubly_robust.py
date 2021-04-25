from collections import namedtuple
import numpy as np


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
    old_policy_choose_action_prob = 1e-5 if old_policy_choose_action_prob < 1e-5 else old_policy_choose_action_prob
    return (true_action_reward - dm_action_reward) * new_policy_choose_action_prob / old_policy_choose_action_prob \
        + dm_new_action_reward


Row = namedtuple("Row", ['info'])
DrTuple = namedtuple("DrTuple", ['fake_view_amt', 'fake_cvr', 'fake_cac', 'true_cvt', 'true_cost', 'true_view_amt'])


def dr_process(row, lamb=0.1, rho=1):
    results = []
    infos = set(row.info)
    val_dict = {}
    for item in infos:
        fake_view_amt, cvr, cac, true_cvt, true_cost, true_view_amt = item.split('#')
        if true_view_amt not in val_dict:
            val_dict[true_view_amt] = []
        val_dict[true_view_amt].append(DrTuple(fake_view_amt, cvr, cac, true_cvt, true_cost, true_view_amt))

    for key, vals in val_dict.items():
        arr_vals = np.array([list(map(float, val)) for val in vals])
        rc = arr_vals[:, 1] - lamb * arr_vals[:, 2]
        policy_state_value = rho * np.log(np.sum(np.exp(rc / rho)))
        pi = np.exp((rc - policy_state_value) / rho)
        print(pi)
        cvr_state_value = np.sum(pi * arr_vals[:, 1])
        cost_state_value = np.sum(pi * arr_vals[:, 2])
        print(cvr_state_value, cost_state_value)

        for i, dr_tuple in enumerate(vals):
            assert isinstance(dr_tuple, DrTuple)
            if key == dr_tuple.fake_view_amt:
                cvr_dr = cvr_state_value + pi[i] * (float(dr_tuple.true_cvt) - float(dr_tuple.fake_cvr))
                cost_dr = cost_state_value + pi[i] * (float(dr_tuple.true_cost) - float(dr_tuple.fake_cac))
                results.append([cvr_dr, cost_dr])
    assert len(results) > 0
    return results


if __name__ == '__main__':
    print(dr_process(Row(info=["100#0.3#5#0#0#100", "200#0.4#6#0#0#100"])))
