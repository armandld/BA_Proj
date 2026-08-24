"""Master-table output paths cannot collide across configurations."""

from pathlib import Path

from aggregate_master_table import DEFAULT_FOLDS, output_paths


def test_canonical_configuration_keeps_the_published_paths(tmp_path):
    paths = output_paths(tmp_path, 256, 2, DEFAULT_FOLDS)
    assert Path(paths["markdown"]).name == "v4_master_table.md"
    assert Path(paths["csv"]).name == "v4_master_table.csv"
    assert Path(paths["npz"]).name == "v4_master.npz"


def test_noncanonical_dimensions_cannot_overwrite_canonical_outputs(tmp_path):
    canonical = set(output_paths(tmp_path, 256, 2, DEFAULT_FOLDS).values())
    unusual = output_paths(tmp_path, 7, 99, DEFAULT_FOLDS)
    assert canonical.isdisjoint(unusual.values())
    assert all("_N7_dim99_" in path for path in unusual.values())


def test_partial_fold_sets_get_distinct_paths(tmp_path):
    full = set(output_paths(tmp_path, 256, 2, DEFAULT_FOLDS).values())
    partial = output_paths(tmp_path, 256, 2, ("ot", "kh"))
    assert full.isdisjoint(partial.values())
    assert all("folds-ot-kh" in path for path in partial.values())
