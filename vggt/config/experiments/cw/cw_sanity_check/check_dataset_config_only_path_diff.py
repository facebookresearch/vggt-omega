import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


def format_path(path_parts: tuple[Any, ...]) -> str:
    if not path_parts:
        return "<root>"
    out = []
    for part in path_parts:
        if isinstance(part, int):
            out.append(f"[{part}]")
        else:
            if not out:
                out.append(str(part))
            else:
                out.append(f".{part}")
    return "".join(out)


def is_allowed_diff_path(path_parts: tuple[Any, ...], allowed_leaf_keys: set[str]) -> bool:
    if not path_parts:
        return False
    leaf = path_parts[-1]
    if not isinstance(leaf, str):
        return False
    return leaf.lower() in allowed_leaf_keys


def diff_yaml(
    cw_obj: Any,
    mast_obj: Any,
    path_parts: tuple[Any, ...],
    allowed_leaf_keys: set[str],
    diffs: list[str],
) -> None:
    if type(cw_obj) is not type(mast_obj):
        if not is_allowed_diff_path(path_parts, allowed_leaf_keys):
            diffs.append(
                f"type mismatch at `{format_path(path_parts)}`: "
                f"cw={type(cw_obj).__name__}, mast={type(mast_obj).__name__}"
            )
        return

    if isinstance(cw_obj, dict):
        cw_keys = set(cw_obj.keys())
        mast_keys = set(mast_obj.keys())
        for key in sorted(cw_keys | mast_keys):
            child_path = (*path_parts, key)
            if key not in cw_obj:
                diffs.append(f"key missing in cw at `{format_path(child_path)}`")
                continue
            if key not in mast_obj:
                diffs.append(f"key missing in mast at `{format_path(child_path)}`")
                continue
            diff_yaml(cw_obj[key], mast_obj[key], child_path, allowed_leaf_keys, diffs)
        return

    if isinstance(cw_obj, list):
        if len(cw_obj) != len(mast_obj):
            if not is_allowed_diff_path(path_parts, allowed_leaf_keys):
                diffs.append(
                    f"list length mismatch at `{format_path(path_parts)}`: "
                    f"cw={len(cw_obj)}, mast={len(mast_obj)}"
                )
            return
        for idx, (cw_item, mast_item) in enumerate(zip(cw_obj, mast_obj)):
            diff_yaml(cw_item, mast_item, (*path_parts, idx), allowed_leaf_keys, diffs)
        return

    if cw_obj != mast_obj and not is_allowed_diff_path(path_parts, allowed_leaf_keys):
        diffs.append(
            f"value mismatch at `{format_path(path_parts)}`: "
            f"cw={cw_obj!r}, mast={mast_obj!r}"
        )


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def collect_yaml_files(directory: Path) -> set[str]:
    return {p.name for p in directory.glob("*.yaml")}


def main() -> int:
    script_path = Path(__file__).resolve()
    default_cw_dir = script_path.parents[1] / "datasets"
    default_mast_dir = script_path.parents[2] / "mast" / "datasets"

    parser = argparse.ArgumentParser(
        description=(
            "Compare dataset YAMLs in cw/datasets and mast/datasets, "
            "ignoring only data-root and seq-list-file path keys."
        )
    )
    parser.add_argument(
        "--cw-datasets-dir",
        type=Path,
        default=default_cw_dir,
        help="Path to cw datasets directory.",
    )
    parser.add_argument(
        "--mast-datasets-dir",
        type=Path,
        default=default_mast_dir,
        help="Path to mast datasets directory.",
    )
    parser.add_argument(
        "--allowed-diff-keys",
        nargs="*",
        default=["UNIFIED_DIR", "sequence_list_file", "scene_list_file"],
        help=(
            "Leaf key names allowed to differ across files "
            "(case-insensitive)."
        ),
    )
    args = parser.parse_args()

    cw_dir = args.cw_datasets_dir.expanduser().resolve()
    mast_dir = args.mast_datasets_dir.expanduser().resolve()

    if not cw_dir.exists():
        print(f"[ERROR] cw datasets directory not found: {cw_dir}")
        return 2
    if not mast_dir.exists():
        print(f"[ERROR] mast datasets directory not found: {mast_dir}")
        return 2

    allowed_leaf_keys = {k.lower() for k in args.allowed_diff_keys}

    cw_files = collect_yaml_files(cw_dir)
    mast_files = collect_yaml_files(mast_dir)

    only_in_cw = sorted(cw_files - mast_files)
    only_in_mast = sorted(mast_files - cw_files)
    common = sorted(cw_files & mast_files)

    has_non_allowed_diff = False

    print(f"[INFO] cw yaml files: {len(cw_files)}")
    print(f"[INFO] mast yaml files: {len(mast_files)}")
    print(f"[INFO] common yaml files: {len(common)}")
    print(f"[INFO] allowed diff keys: {sorted(allowed_leaf_keys)}")

    if only_in_cw:
        has_non_allowed_diff = True
        print("\n[DIFF] Files only in cw:")
        for name in only_in_cw:
            print(f"  - {name}")

    if only_in_mast:
        has_non_allowed_diff = True
        print("\n[DIFF] Files only in mast:")
        for name in only_in_mast:
            print(f"  - {name}")

    for filename in common:
        cw_file = cw_dir / filename
        mast_file = mast_dir / filename
        try:
            cw_data = load_yaml(cw_file)
            mast_data = load_yaml(mast_file)
        except Exception as exc:
            has_non_allowed_diff = True
            print(f"\n[DIFF] Failed to parse {filename}: {exc}")
            continue

        diffs: list[str] = []
        diff_yaml(cw_data, mast_data, tuple(), allowed_leaf_keys, diffs)
        if diffs:
            has_non_allowed_diff = True
            print(f"\n[DIFF] {filename}")
            for msg in diffs:
                print(f"  - {msg}")

    if has_non_allowed_diff:
        print("\n[RESULT] Found differences beyond allowed path keys.")
        return 1

    print("\n[RESULT] All common dataset YAMLs differ only in allowed path keys.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

