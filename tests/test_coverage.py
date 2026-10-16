from resolvfl.taxonomy import Taxonomy
from resolvfl.coverage import empirical_supervision_counts, empirical_coverage_flags, maximal_uncovered_roots

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



def test_bottom_up_counts_match_ancestor_definition():
    # Mixed depths, a root annotation and multiple clients exercise the actual contract.
    t = Taxonomy('r', {'r': ['a', 'b'], 'b': ['c', 'd'], 'd': ['e', 'f']})
    hist = {'r': 7, 'a': 2, 'b': 3, 'c': 5, 'd': 11, 'e': 13, 'f': 17}
    expected = {u: sum(n for label, n in hist.items() if u in t.ancestors(label))
                for u in t.internal_nodes}
    assert empirical_supervision_counts(hist, t) == expected
    assert empirical_supervision_counts({'r': 20}, t) == {'r': 0, 'b': 0, 'd': 0}
    assert empirical_coverage_flags({0: {'b': 2}, 1: {'e': 1}}, t) == {'r': True, 'b': True, 'd': True}


def test_uncovered_subtree_is_reported_once():
    t = Taxonomy('r', {'r': ['a', 'b'], 'b': ['c', 'd'], 'd': ['e', 'f']})
    assert maximal_uncovered_roots({'r': True, 'b': False, 'd': False}, t) == ['b']
    assert maximal_uncovered_roots({'r': False, 'b': False, 'd': False}, t) == ['r']

