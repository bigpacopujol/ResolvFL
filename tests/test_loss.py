import torch
from resolvfl.taxonomy import Taxonomy
from resolvfl.loss import exact_path_nll

def test_internal_label_stops_path():
    t = Taxonomy("r", {"r": ["A", "B"], "A": ["a1", "a2"], "B": ["b1", "b2"]})
    logits = {"r": torch.tensor([2.0, 0.0]), "A": torch.tensor([-10.0, 10.0]), "B": torch.tensor([0.0, 0.0])}
    coarse = exact_path_nll(logits, "A", t)
    expected = torch.nn.functional.cross_entropy(logits["r"].unsqueeze(0), torch.tensor([0]))
    assert torch.allclose(coarse, expected)

def test_leaf_label_uses_both_heads():
    t = Taxonomy("r", {"r": ["A", "B"], "A": ["a1", "a2"], "B": ["b1", "b2"]})
    logits = {"r": torch.tensor([2.0, 0.0]), "A": torch.tensor([0.0, 2.0]), "B": torch.tensor([0.0, 0.0])}
    got = exact_path_nll(logits, "a2", t)
    e1 = torch.nn.functional.cross_entropy(logits["r"].unsqueeze(0), torch.tensor([0]))
    e2 = torch.nn.functional.cross_entropy(logits["A"].unsqueeze(0), torch.tensor([1]))
    assert torch.allclose(got, e1 + e2)


