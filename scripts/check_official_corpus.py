#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GZIP_SHA256 = "a0ba9a8618ce72a7b383a6de31acf6e5bf138873da0d13ad8a1a6b0271e8e9d1"
EXPECTED_GZIP_SIZE = 1_471_795_629
OFFICIAL_REPO = "https://github.com/yjwjy/ShoppingBench.git"
KNOWN_FORKS = [
    "https://github.com/yzd2002/ShoppingBench.git",
    "https://github.com/kato114/ShoppingBench.git",
    "https://github.com/orionfei/ShoppingBench.git",
    "https://github.com/bittensorrider/ShoppingBench.git",
    "https://github.com/Ariya12138/ShoppingBench.git",
    "https://github.com/AntiQuality/ShoppingBench.git",
    "https://github.com/kejunxiao/ShoppingBench.git",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lfs_pointer(path: Path) -> dict | None:
    try:
        raw = path.read_bytes()[:512]
    except OSError:
        return None
    if not raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return None
    text = raw.decode("utf-8", errors="replace")
    pointer = {"is_lfs_pointer": True}
    for line in text.splitlines():
        if line.startswith("oid sha256:"):
            pointer["oid_sha256"] = line.split("oid sha256:", 1)[1].strip()
        elif line.startswith("size "):
            try:
                pointer["lfs_declared_size"] = int(line.split("size ", 1)[1].strip())
            except ValueError:
                pointer["lfs_declared_size"] = line.split("size ", 1)[1].strip()
    return pointer


def gzip_smoke(path: Path) -> dict:
    try:
        with gzip.open(path, "rb") as fin:
            sample = fin.read(1024)
        return {"gzip_smoke_ok": True, "sample_bytes": len(sample)}
    except Exception as exc:
        return {"gzip_smoke_ok": False, "error": str(exc)}


def inspect_file(path: Path, expected_sha: str, expected_size: int, gzip_check: bool = False) -> dict:
    info = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    info["file_size"] = path.stat().st_size
    pointer = lfs_pointer(path)
    if pointer:
        info.update(pointer)
        info["valid_official_gzip"] = False
        return info
    info["size_matches_expected"] = info["file_size"] == expected_size
    if info["size_matches_expected"]:
        info["sha256"] = sha256_file(path)
        info["sha256_matches_expected"] = info["sha256"] == expected_sha
    else:
        info["sha256"] = None
        info["sha256_matches_expected"] = False
    info["valid_official_gzip"] = bool(info["size_matches_expected"] and info["sha256_matches_expected"])
    if gzip_check and info["file_size"] > 0:
        info.update(gzip_smoke(path))
    return info


def candidate_files(raw_candidates: list[str]) -> list[Path]:
    out = []
    for raw in raw_candidates:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            out.append(path)
        elif path.is_dir():
            out.extend(path.rglob("documents.jsonl.gz"))
    seen = set()
    unique = []
    for path in out:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def copy_candidate(candidate: Path, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(candidate, tmp)
    tmp.replace(target)
    return {"copied_from": str(candidate), "copied_to": str(target)}


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 180, env: dict | None = None) -> dict:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {"cmd": cmd, "returncode": proc.returncode, "output_tail": proc.stdout[-4000:]}


def attempt_lfs_download(repo_url: str, workdir: Path) -> dict:
    workdir.parent.mkdir(parents=True, exist_ok=True)
    result = {"repo": repo_url, "workdir": str(workdir), "steps": []}
    if not workdir.exists():
        result["steps"].append(
            run(
                [
                    "git",
                    "-c",
                    "filter.lfs.smudge=",
                    "-c",
                    "filter.lfs.required=false",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    repo_url,
                    str(workdir),
                ],
                timeout=300,
                env={"GIT_LFS_SKIP_SMUDGE": "1"},
            )
        )
        if result["steps"][-1]["returncode"] != 0:
            return result
    result["steps"].append(run(["git", "lfs", "install", "--local"], cwd=workdir))
    result["steps"].append(
        run(
            ["git", "lfs", "pull", "--include=resources/documents.jsonl.gz"],
            cwd=workdir,
            timeout=600,
        )
    )
    candidate = workdir / "resources" / "documents.jsonl.gz"
    result["candidate"] = inspect_file(candidate, EXPECTED_GZIP_SHA256, EXPECTED_GZIP_SIZE)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Check or recover the official ShoppingBench product corpus.")
    parser.add_argument("--target", default="resources/documents.jsonl.gz")
    parser.add_argument("--jsonl-target", default="resources/documents.jsonl")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--copy-valid-candidate", action="store_true")
    parser.add_argument("--attempt-lfs", action="store_true")
    parser.add_argument("--try-forks", action="store_true")
    parser.add_argument("--gzip-smoke", action="store_true")
    parser.add_argument("--output-report", default="reports/official_corpus_status.json")
    return parser.parse_args()


def main():
    args = parse_args()
    target = Path(args.target)
    if not target.is_absolute():
        target = ROOT / target
    jsonl_target = Path(args.jsonl_target)
    if not jsonl_target.is_absolute():
        jsonl_target = ROOT / jsonl_target

    report = {
        "expected_gzip": {
            "sha256": EXPECTED_GZIP_SHA256,
            "size": EXPECTED_GZIP_SIZE,
            "official_repo": OFFICIAL_REPO,
        },
        "target": inspect_file(target, EXPECTED_GZIP_SHA256, EXPECTED_GZIP_SIZE, args.gzip_smoke),
        "jsonl_target": {"path": str(jsonl_target), "exists": jsonl_target.exists()},
        "candidates": [],
        "lfs_attempts": [],
        "ready_for_formal_eval": False,
        "recommended_next_action": "",
    }

    if report["target"].get("valid_official_gzip"):
        report["ready_for_formal_eval"] = True
        report["recommended_next_action"] = "Run scripts/run_sft_qwen3_4b_formal_eval.sh."
    else:
        for candidate in candidate_files(args.candidate):
            info = inspect_file(candidate, EXPECTED_GZIP_SHA256, EXPECTED_GZIP_SIZE, args.gzip_smoke)
            report["candidates"].append(info)
            if info.get("valid_official_gzip"):
                if args.copy_valid_candidate:
                    report["copy"] = copy_candidate(candidate, target)
                    report["target"] = inspect_file(target, EXPECTED_GZIP_SHA256, EXPECTED_GZIP_SIZE, args.gzip_smoke)
                report["ready_for_formal_eval"] = args.copy_valid_candidate
                break

    if not report["ready_for_formal_eval"] and args.attempt_lfs:
        repos = [OFFICIAL_REPO]
        if args.try_forks:
            repos.extend(KNOWN_FORKS)
        for idx, repo in enumerate(repos):
            attempt = attempt_lfs_download(repo, Path("/tmp") / f"ShoppingBench_corpus_lfs_{idx}")
            report["lfs_attempts"].append(attempt)
            candidate_info = attempt.get("candidate", {})
            if candidate_info.get("valid_official_gzip"):
                report["copy"] = copy_candidate(Path(candidate_info["path"]), target)
                report["target"] = inspect_file(target, EXPECTED_GZIP_SHA256, EXPECTED_GZIP_SIZE, args.gzip_smoke)
                report["ready_for_formal_eval"] = True
                break

    if not report["ready_for_formal_eval"]:
        if report["target"].get("is_lfs_pointer"):
            report["recommended_next_action"] = "Target is only a Git LFS pointer; copy the real 1.47GB documents.jsonl.gz from a machine with LFS access."
        elif jsonl_target.exists():
            report["recommended_next_action"] = "Uncompressed documents.jsonl exists, but compressed official SHA was not verified; build indexes only if this file is known official."
        else:
            report["recommended_next_action"] = "Obtain resources/documents.jsonl.gz with the expected SHA256, then rerun this script."

    output = Path(args.output_report)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_formal_eval"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
