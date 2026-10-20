from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import review_catalog, review_dataset, save_report
def _print_report(report) -> None:
    print(f"dataset: {report.dataset_path}")
    print(f"format:  {report.format}")
    print(f"scored:  {report.n_episodes_scored}/{report.n_episodes_total} episodes")
    print(f"roles:   {report.role_map}")
    overall = "N/A" if report.overall_score is None else f"{report.overall_score:.3f}"
    print(f"overall: {overall}  status={report.status}")
    if report.gates_failed:
        print(f"gates failed: {report.gates_failed}")
    prof = (report.meta or {}).get("profile") or {}
    if prof:
        print(
            f"profile: family={prof.get('family')} "
            f"tactile={prof.get('expects_tactile')} action={prof.get('expects_action')} "
            f"sequence={prof.get('expects_sequence')}"
        )
    print("-" * 60)
    for d in report.dimensions:
        sc = "N/A" if d.score is None else f"{d.score:.3f}"
        print(f"  [{d.status:4}] {d.id:14} {sc:>6}  {d.name}")
        for n in d.notes[:2]:
            print(f"           · {n}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="DataReviewer — schema-agnostic robotics/tactile dataset quality scoring."
    )
    sub = p.add_subparsers(dest="cmd")

    # single dataset
    s = sub.add_parser("review", help="Review one local dataset root")
    s.add_argument("--dataset", type=Path, required=True)
    s.add_argument("--max-episodes", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--roles", type=Path, default=None)
    s.add_argument("--out", type=Path, default=None)
    s.add_argument("--video-probe-per-ep", type=int, default=2)

    # catalog
    c = sub.add_parser(
        "catalog",
        help="Parse a living catalog CSV (Notion export or compatible) and score local hits",
    )
    c.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Catalog CSV (reference list; append rows as the inventory grows)",
    )
    c.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Local root(s) to search for matching dataset folders (repeatable)",
    )
    c.add_argument("--max-episodes", type=int, default=10)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--roles", type=Path, default=None)
    c.add_argument("--out", type=Path, default=None)
    c.add_argument(
        "--print-profiles",
        action="store_true",
        help="Print inferred family/modality profile for each CSV row",
    )

    # backwards-compatible flat args (no subcommand)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--max-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--roles", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--video-probe-per-ep", type=int, default=2)

    args = p.parse_args()

    if args.cmd == "catalog":
        summary = review_catalog(
            args.csv,
            search_roots=args.search_root or None,
            max_episodes=args.max_episodes,
            seed=args.seed,
            roles_config=args.roles,
        )
        if args.print_profiles:
            for r in summary["results"]:
                pr = r["profile"]
                print(
                    f"- {pr['name'][:40]:40} family={pr['family']:12} "
                    f"act={int(pr['expects_action'])} tac={int(pr['expects_tactile'])} "
                    f"seq={int(pr['expects_sequence'])} dims={r['applicable_dimensions']} "
                    f"local={len(r['local_hits'])} status={r['status']}"
                )
        out = args.out or Path("outputs") / "catalog_review.json"
        save_report(summary, out)
        print(
            f"catalog datasets={summary['n_datasets']} "
            f"local_reviewed={summary['n_with_local_review']} wrote {out}"
        )
        return

    # default / review subcommand
    dataset = args.dataset
    if args.cmd == "review":
        dataset = args.dataset
    if dataset is None:
        p.print_help()
        print("\nExamples:")
        print("  python -m data_reviewer review --dataset /path/to/lerobot_v2.1")
        print(
            "  python -m data_reviewer catalog --csv Notion_Deduplicated_Datasets.csv "
            "--search-root ./SharpaDex-v1.0_sample10 --print-profiles"
        )
        raise SystemExit(2)

    report = review_dataset(
        dataset,
        max_episodes=args.max_episodes,
        roles_config=args.roles,
        seed=args.seed,
        video_probe_per_ep=getattr(args, "video_probe_per_ep", 2),
    )
    _print_report(report)
    out = args.out or Path("outputs") / f"review_{Path(report.dataset_path).name}.json"
    save_report(report, out)
    print("-" * 60)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
