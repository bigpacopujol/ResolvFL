from fedgrain.taxonomy import Taxonomy
from fedgrain.coverage import empirical_supervision_counts, empirical_coverage_flags, maximal_uncovered_roots

def test_coverage_counts_and_maximal_uncovered():
    t = Taxonomy("r", {"r": ["A", "B"], "A": ["a1", "a2"], "B": ["b1", "b2"]})
    hist = {"A": 20, "a1": 3, "a2": 2, "B": 15}
    c = empirical_supervision_counts(hist, t)
    assert c["r"] == 40
    assert c["A"] == 5
    assert c["B"] == 0
    flags = empirical_coverage_flags({0: hist}, t)
    assert flags == {"r": True, "A": True, "B": False}
    assert maximal_uncovered_roots(flags, t) == ["B"]
