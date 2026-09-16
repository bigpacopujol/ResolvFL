import torch
from fedgrain.aggregation import fedgrain_endpoint_average, active_cohort_endpoint_average

def test_inactive_clients_are_excluded():
    vals = {0: torch.tensor([1.0]), 1: torch.tensor([3.0]), 2: torch.tensor([100.0])}
    w = {0: 1/3, 1: 1/3, 2: 1/3}
    m = {0: 0.25, 1: 0.75, 2: 0.0}
    got = fedgrain_endpoint_average(vals, w, m)
    assert torch.allclose(got, torch.tensor([2.5]))

def test_no_extra_mass_ablation():
    vals = {0: torch.tensor([1.0]), 1: torch.tensor([3.0])}
    w = {0: 1.0, 1: 1.0}
    m = {0: 0.25, 1: 0.75}
    got = active_cohort_endpoint_average(vals, w, m)
    assert torch.allclose(got, torch.tensor([2.0]))
