from fedgrain.partitions import controlled_deep_cohort_size, controlled_realized_overlap

def test_controlled_overlap_rounding_matches_experiment():
    nominal = [1.0, 0.75, 0.5, 0.25, 0.1]
    sizes = [controlled_deep_cohort_size(x, 10) for x in nominal]
    realized = [controlled_realized_overlap(x, 10) for x in nominal]
    assert sizes == [10, 8, 5, 2, 1]
    assert realized == [1.0, 0.8, 0.5, 0.2, 0.1]
