from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .adapters import RawSample


def resolve_group_key(sample: RawSample, group_priority: Sequence[str]) -> str:
    for spec in group_priority:
        field_names = [field.strip() for field in str(spec).split("+") if field.strip()]
        if not field_names:
            continue

        values: list[str] = []
        has_any = False
        for field_name in field_names:
            value = str(getattr(sample, field_name, "") or "")
            if value:
                has_any = True
            values.append(f"{field_name}={value}")
        if has_any:
            joined = "|".join(values)
            return f"{sample.dataset}|group|{joined}"
    return f"{sample.dataset}|video_id|{sample.video_id}"


def create_group_splits(
    samples: list[RawSample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_priority: Sequence[str],
) -> dict[str, list[int]]:
    if len(samples) == 0:
        return {"train": [], "val": [], "test": []}

    if val_ratio < 0.0 or test_ratio < 0.0:
        raise ValueError("val_ratio and test_ratio must be >= 0")

    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    all_groups = np.array([resolve_group_key(s, group_priority) for s in samples])
    class_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        class_to_indices[sample.class_name].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for class_offset, class_name in enumerate(sorted(class_to_indices.keys())):
        class_indices = np.array(class_to_indices[class_name], dtype=np.int64)
        class_groups = all_groups[class_indices]
        class_split = _split_group_indices(
            indices=class_indices,
            groups=class_groups,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed + (class_offset * 17),
            class_name=class_name,
        )
        train_idx.extend(class_split["train"])
        val_idx.extend(class_split["val"])
        test_idx.extend(class_split["test"])

    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    test_idx = sorted(test_idx)

    _validate_no_group_overlap(all_groups, train_idx, val_idx, test_idx)

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _split_group_indices(
    indices: np.ndarray,
    groups: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    class_name: str,
) -> dict[str, list[int]]:
    unique_groups = np.unique(groups)
    required_splits = 1 + int(val_ratio > 0.0) + int(test_ratio > 0.0)
    if len(unique_groups) < required_splits:
        raise ValueError(
            f"Class `{class_name}` has only {len(unique_groups)} unique groups, "
            f"but the current split requires {required_splits} subsets. "
            "Collect more distinct sources or reduce val/test splits."
        )

    if test_ratio > 0.0:
        split_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_local, test_local = next(split_test.split(indices, groups=groups))
    else:
        train_val_local = np.arange(len(indices))
        test_local = np.array([], dtype=np.int64)

    if val_ratio > 0.0:
        val_ratio_in_train_val = val_ratio / (1.0 - test_ratio)
        groups_train_val = groups[train_val_local]
        split_val = GroupShuffleSplit(
            n_splits=1,
            test_size=val_ratio_in_train_val,
            random_state=seed + 1,
        )
        train_local, val_local = next(
            split_val.split(train_val_local, groups=groups_train_val)
        )
    else:
        train_local = np.arange(len(train_val_local))
        val_local = np.array([], dtype=np.int64)

    train_idx = indices[train_val_local[train_local]].tolist()
    val_idx = indices[train_val_local[val_local]].tolist()
    test_idx = indices[test_local].tolist()
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _validate_no_group_overlap(
    groups: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
) -> None:
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])

    if train_groups & val_groups:
        raise RuntimeError("Leakage detected between train and val splits")
    if train_groups & test_groups:
        raise RuntimeError("Leakage detected between train and test splits")
    if val_groups & test_groups:
        raise RuntimeError("Leakage detected between val and test splits")
