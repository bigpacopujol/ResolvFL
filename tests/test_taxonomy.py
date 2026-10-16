from resolvfl.taxonomy import Taxonomy

def toy():
    return Taxonomy("r", {"r": ["A", "B"], "A": ["a1", "a2"], "B": ["b1", "b2"]})

def test_paths_and_leaves():
    t = toy()
    assert t.path("a2") == ["r", "A", "a2"]
    assert set(t.leaves) == {"a1", "a2", "b1", "b2"}
    assert set(t.descendant_leaves("A")) == {"a1", "a2"}


