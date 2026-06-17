#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split teacher voucher trajectories into trajectory-aligned SFT train/eval files."
    )
    parser.add_argument(
        "--input-rollout",
        default="data/teacher_voucher_train_clean691_state_folded.jsonl",
        help="Source trajectory JSONL. One line is one trajectory.",
    )
    parser.add_argument(
        "--input-synthesize",
        default="data/teacher_voucher_train_clean691_synthesize.jsonl",
        help="Source synthesize JSONL aligned with the rollout file.",
    )
    parser.add_argument(
        "--input-sft",
        default="data/teacher_voucher_train_clean691_sft.json",
        help="Source Alpaca-style SFT JSON built from the rollout file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sft_splits/teacher_voucher_clean691_train_all",
        help="Directory for split rollout/synthesize/SFT/report outputs.",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="teacher_voucher_clean691",
        help="Prefix for output file names and dataset_info entries.",
    )
    parser.add_argument(
        "--train-trajectories",
        type=int,
        default=0,
        help="Number of trajectories to put in train. Use 0 for all trajectories not reserved for eval.",
    )
    parser.add_argument("--eval-trajectories", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument(
        "--selection",
        choices=["first", "random"],
        default="first",
        help="Use first N trajectories for smoke checks, or seeded random selection.",
    )
    parser.add_argument(
        "--sft-data-dir",
        default="src/sft/data",
        help="Optional LLaMA-Factory data directory to receive SFT files and dataset_info.json.",
    )
    parser.add_argument(
        "--skip-sft-data-dir",
        action="store_true",
        help="Only write output-dir files; do not update src/sft/data.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(obj, fout, ensure_ascii=False, indent=2)
        fout.write("\n")


def trajectory_query(row: list[dict]) -> str:
    if not row:
        return ""
    return row[0].get("extra_info", {}).get("query", "")


def group_sft_by_query(sft_samples: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for sample in sft_samples:
        query = sample.get("extra_info", {}).get("query")
        if not query:
            raise ValueError("SFT sample is missing extra_info.query")
        grouped[query].append(sample)
    return dict(grouped)


def select_indices(total: int, train_n: int, eval_n: int, selection: str, seed: int) -> tuple[list[int], list[int]]:
    if train_n < 0 or eval_n < 0:
        raise ValueError("--train-trajectories and --eval-trajectories must be non-negative.")
    if train_n == 0:
        train_n = total - eval_n
    needed = train_n + eval_n
    if needed > total:
        raise ValueError(f"Requested {needed} trajectories, but only {total} are available.")

    indices = list(range(total))
    if selection == "random":
        rng = random.Random(seed)
        rng.shuffle(indices)

    picked = indices[:needed]
    return picked[:train_n], picked[train_n:]


def validate_alignment(rollout_rows: list, synthesize_rows: list, sft_by_query: dict[str, list[dict]]) -> None:
    if len(rollout_rows) != len(synthesize_rows):
        raise ValueError(
            f"Rollout/synthesize row count mismatch: {len(rollout_rows)} vs {len(synthesize_rows)}"
        )

    missing = []
    step_mismatches = []
    synth_mismatches = []
    for idx, (rollout, synthesize) in enumerate(zip(rollout_rows, synthesize_rows), 1):
        query = trajectory_query(rollout)
        if not query:
            raise ValueError(f"Rollout row {idx} is missing extra_info.query")
        if synthesize.get("query") != query:
            synth_mismatches.append(idx)
        sft_steps = sft_by_query.get(query)
        if not sft_steps:
            missing.append(idx)
        elif len(sft_steps) != len(rollout):
            step_mismatches.append((idx, len(rollout), len(sft_steps)))

    if synth_mismatches:
        raise ValueError(f"Synthesize query mismatch at rollout rows: {synth_mismatches[:10]}")
    if missing:
        raise ValueError(f"SFT samples missing for rollout rows: {missing[:10]}")
    if step_mismatches:
        raise ValueError(f"SFT step count mismatches: {step_mismatches[:10]}")


def split_payload(
    indices: list[int],
    rollout_rows: list,
    synthesize_rows: list,
    sft_by_query: dict[str, list[dict]],
) -> tuple[list, list, list[dict]]:
    rollout_out = [rollout_rows[i] for i in indices]
    synthesize_out = [synthesize_rows[i] for i in indices]
    sft_out = []
    for row in rollout_out:
        sft_out.extend(sft_by_query[trajectory_query(row)])
    return rollout_out, synthesize_out, sft_out


def update_dataset_info(
    data_dir: Path,
    train_name: str,
    train_file: str,
    eval_name: str | None = None,
    eval_file: str | None = None,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    info_path = data_dir / "dataset_info.json"
    if info_path.exists():
        with info_path.open(encoding="utf-8") as fin:
            info = json.load(fin)
    else:
        info = {}

    columns = {"prompt": "instruction", "query": "input", "response": "output"}
    info[train_name] = {"file_name": train_file, "columns": columns}
    if eval_name and eval_file:
        info[eval_name] = {"file_name": eval_file, "columns": columns}
    write_json(info_path, info)


def main() -> None:
    args = parse_args()
    input_rollout = ROOT / args.input_rollout
    input_synthesize = ROOT / args.input_synthesize
    input_sft = ROOT / args.input_sft
    output_dir = ROOT / args.output_dir

    rollout_rows = read_jsonl(input_rollout)
    synthesize_rows = read_jsonl(input_synthesize)
    with input_sft.open(encoding="utf-8") as fin:
        sft_samples = json.load(fin)

    sft_by_query = group_sft_by_query(sft_samples)
    validate_alignment(rollout_rows, synthesize_rows, sft_by_query)

    train_indices, eval_indices = select_indices(
        len(rollout_rows),
        args.train_trajectories,
        args.eval_trajectories,
        args.selection,
        args.seed,
    )

    train_rollout, train_synthesize, train_sft = split_payload(
        train_indices, rollout_rows, synthesize_rows, sft_by_query
    )
    eval_rollout, eval_synthesize, eval_sft = split_payload(
        eval_indices, rollout_rows, synthesize_rows, sft_by_query
    )

    prefix = args.dataset_prefix
    paths = {
        "train_rollout": output_dir / f"{prefix}_train_rollout.jsonl",
        "train_synthesize": output_dir / f"{prefix}_train_synthesize.jsonl",
        "train_sft": output_dir / f"{prefix}_train_sft.json",
        "report": output_dir / f"{prefix}_split_report.json",
    }
    if eval_indices:
        paths.update(
            {
                "eval_rollout": output_dir / f"{prefix}_eval_rollout.jsonl",
                "eval_synthesize": output_dir / f"{prefix}_eval_synthesize.jsonl",
                "eval_sft": output_dir / f"{prefix}_eval_sft.json",
            }
        )

    write_jsonl(paths["train_rollout"], train_rollout)
    write_jsonl(paths["train_synthesize"], train_synthesize)
    write_json(paths["train_sft"], train_sft)
    if eval_indices:
        write_jsonl(paths["eval_rollout"], eval_rollout)
        write_jsonl(paths["eval_synthesize"], eval_synthesize)
        write_json(paths["eval_sft"], eval_sft)

    train_name = f"{prefix}_train"
    eval_name = f"{prefix}_eval" if eval_indices else None
    report = {
        "source": {
            "rollout": args.input_rollout,
            "synthesize": args.input_synthesize,
            "sft": args.input_sft,
        },
        "selection": args.selection,
        "seed": args.seed,
        "train": {
            "dataset_name": train_name,
            "trajectory_count": len(train_rollout),
            "sft_count": len(train_sft),
            "indices_0_based": train_indices,
            "line_no": [row[0].get("extra_info", {}).get("line_no") for row in train_rollout],
            "sample_id": [row[0].get("extra_info", {}).get("sample_id") for row in train_rollout],
        },
        "eval": None,
        "outputs": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
    }
    if eval_indices:
        report["eval"] = {
            "dataset_name": eval_name,
            "trajectory_count": len(eval_rollout),
            "sft_count": len(eval_sft),
            "indices_0_based": eval_indices,
            "line_no": [row[0].get("extra_info", {}).get("line_no") for row in eval_rollout],
            "sample_id": [row[0].get("extra_info", {}).get("sample_id") for row in eval_rollout],
        }

    if not args.skip_sft_data_dir:
        data_dir = ROOT / args.sft_data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        train_file = f"{prefix}_train_sft.json"
        shutil.copyfile(paths["train_sft"], data_dir / train_file)
        eval_file = None
        if eval_indices:
            eval_file = f"{prefix}_eval_sft.json"
            shutil.copyfile(paths["eval_sft"], data_dir / eval_file)
        update_dataset_info(data_dir, train_name, train_file, eval_name, eval_file)
        report["sft_data_dir"] = {
            "dir": args.sft_data_dir,
            "train_file": train_file,
            "dataset_info": str((data_dir / "dataset_info.json").relative_to(ROOT)),
        }
        if eval_file:
            report["sft_data_dir"]["eval_file"] = eval_file

    write_json(paths["report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
