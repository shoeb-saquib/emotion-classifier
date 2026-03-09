"""
Delete reports for the selected emotion representations, context methods, and context windows
from configuration.py. Removes files and directories that become empty.
Uses the same reports layout as generate_reports: reports/<cluster_subdir>/er0/, er1/, er2/
"""

import re
from pathlib import Path

from src.configuration import (
    CONTEXT_WINDOWS,
    KNN_NEIGHBORS,
    SELECTED_CLUSTER_VARIATIONS,
    SELECTED_CONTEXT_METHODS,
    SELECTED_EMOTION_REPRESENTATIONS,
)

_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir
while _project_root != _project_root.parent and (_project_root / "src").is_dir() is False:
    _project_root = _project_root.parent
REPORTS_ROOT = _project_root / "reports"
_cluster_subdir = (
    f"cv_{'_'.join(SELECTED_CLUSTER_VARIATIONS)}_knn{KNN_NEIGHBORS}"
    if SELECTED_CLUSTER_VARIATIONS
    else "no_cluster"
)
REPORTS_DIR = REPORTS_ROOT / _cluster_subdir
SEP = "=" * 80


def _parse_context_window_from_report(block: str) -> int | None:
    """Extract Context Window value from a report block; return None if not found."""
    m = re.search(r"Context Window\s+:\s*(\d+)", block)
    return int(m.group(1)) if m else None


def _parse_reports_file(path: Path) -> dict[int, str]:
    """Read a combined reports file and return {context_window: report_content}."""
    out = {}
    if not path.exists():
        return out
    text = path.read_text()
    blocks = text.split("\n\n" + SEP + "\n")
    for block in blocks:
        block = block.strip()
        if not block or "EMOTION CLASSIFICATION" not in block:
            continue
        cw = _parse_context_window_from_report(block)
        if cw is not None:
            out[cw] = block if block.startswith(SEP) else (SEP + "\n" + block)
    return out


def _write_combined_reports(path: Path, reports_by_cw: dict[int, str]) -> bool:
    """
    Write reports to path in ascending order of context window.
    If reports_by_cw is empty, delete the file. Return True if file was deleted.
    """
    if not reports_by_cw:
        path.unlink(missing_ok=True)
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [reports_by_cw[cw] for cw in sorted(reports_by_cw)]
    path.write_text("\n\n".join(s.rstrip() for s in ordered))
    return False


def _remove_empty_dirs(dir_path: Path, up_to: Path) -> None:
    """Remove dir_path and its parents if they become empty, up to (but not including) up_to."""
    while dir_path != up_to and dir_path.exists():
        try:
            if next(dir_path.iterdir(), None) is None:
                dir_path.rmdir()
                dir_path = dir_path.parent
            else:
                break
        except OSError:
            break


def delete_reports(
    er_ids: list[int],
    context_method_ids: list[int],
    context_windows: list[int],
    dry_run: bool = False,
) -> None:
    """
    Delete reports for the given (er_id, context_method, context_window) combinations.
    Removes files and directories that become empty.
    """
    context_windows_list = list(context_windows)
    dirs_touched: set[Path] = set()

    for er_id in er_ids:
        er_dir = REPORTS_DIR / f"er{er_id}"
        if not er_dir.is_dir():
            continue

        if 0 in context_windows_list:
            cw0_path = er_dir / f"er{er_id}_cw0_reports.txt"
            if cw0_path.exists():
                if dry_run:
                    print(f"[dry run] Would delete {cw0_path}")
                else:
                    cw0_path.unlink()
                    print(f"Deleted {cw0_path}")
                dirs_touched.add(er_dir)

        cws_to_remove = [cw for cw in context_windows_list if cw > 0]
        if not cws_to_remove:
            continue

        for cm_id in context_method_ids:
            path = er_dir / f"er{er_id}_cm{cm_id}_reports.txt"
            if not path.exists():
                continue
            reports_by_cw = _parse_reports_file(path)
            removed = [cw for cw in cws_to_remove if cw in reports_by_cw]
            for cw in removed:
                del reports_by_cw[cw]
            if not removed:
                continue
            dirs_touched.add(er_dir)
            if dry_run:
                print(f"[dry run] Would remove cw {removed} from {path}")
            else:
                deleted = _write_combined_reports(path, reports_by_cw)
                if deleted:
                    print(f"Deleted {path}")
                else:
                    print(f"Removed context window(s) {removed} from {path}")

    if not dry_run and dirs_touched:
        for er_dir in sorted(dirs_touched, key=lambda p: -len(p.parts)):
            _remove_empty_dirs(er_dir, REPORTS_DIR)
        _remove_empty_dirs(REPORTS_DIR, REPORTS_ROOT)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Delete reports for selected emotion representations, context methods, and context windows from configuration.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without changing files.",
    )
    args = parser.parse_args()

    delete_reports(
        er_ids=list(SELECTED_EMOTION_REPRESENTATIONS),
        context_method_ids=list(SELECTED_CONTEXT_METHODS),
        context_windows=list(CONTEXT_WINDOWS),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
