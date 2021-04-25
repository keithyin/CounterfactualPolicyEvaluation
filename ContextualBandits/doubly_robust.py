from collections import namedtuple
import numpy as np
from collections import Counter

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


def dr_process(row, lamb=0.1, rho=1, pick_stra='argmax'):
    results = []
    infos = sorted(list(set(row.info)))
    val_dict = {}
    picks = []
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
        cvr_state_value = np.sum(pi * arr_vals[:, 1])
        cost_state_value = np.sum(pi * arr_vals[:, 2])

        if pick_stra == 'argmax':
            picks.append(arr_vals[np.argmax(pi)][0])
        elif pick_stra == 'cate_sampling':
            picks.append(arr_vals[np.random.choice(len(pi), p=pi)][0])
        else:
            raise ValueError("invalid pick_stra: {}, expected argmax|cate_sampling".format(pick_stra))

        for i, dr_tuple in enumerate(vals):
            assert isinstance(dr_tuple, DrTuple)
            if key == dr_tuple.fake_view_amt:
                cvr_dr = cvr_state_value + pi[i] * (float(dr_tuple.true_cvt) - float(dr_tuple.fake_cvr))
                cost_dr = cost_state_value + pi[i] * (float(dr_tuple.true_cost) - float(dr_tuple.fake_cac))
                results.append([cvr_dr, cost_dr])
    assert len(results) > 0
    return results, picks


def print_counter(counter):
    tot_record = 0
    for _, val in counter.items():
        tot_record += val
    print("Counter: --------->")
    print("tot_record: {}".format(tot_record))
    logs = []
    for k, val in counter.items():
        logs.append([k, val / float(tot_record)])
    logs = sorted(logs, key=lambda x: x[0])
    logs = ["amt: {}, prop: {}".format(k, prob) for k, prob in logs]
    logs = "\n".join(logs)
    print(logs)
    print("-----------------------")


if __name__ == '__main__':
    counter = Counter()
    for i in range(10000):
        _, picked = dr_process(Row(info=["100#0.3#5#0#0#100", "200#0.4#5.5#0#0#100"]), lamb=2,
                               pick_stra="cate_sampling")
        counter.update(picked)
    print_counter(counter)

