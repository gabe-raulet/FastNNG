#!/usr/bin/env python3
"""
Parse results.m1.txt and plot phase times vs centers.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PHASE_ORDER = [
    "read input",
    "partition",
    "coalesce",
    "ghost points",
    "neighbors",
    "redistribute",
    "complete",
]


def phase_key(msg: str) -> str:
    if msg.startswith("read input file"):
        return "read input"
    if msg.startswith("computed point partitioning"):
        return "partition"
    if msg.startswith("coalesced cells"):
        return "coalesce"
    if msg.startswith("added ghost points"):
        return "ghost points"
    if msg.startswith("found neighbors"):
        return "neighbors"
    if msg.startswith("redistributed edges"):
        return "redistribute"
    if msg.startswith("complete"):
        return "complete"
    return msg


def parse_runs(path: Path):
    time_re = re.compile(r"^\[time=([0-9.]+)\]\s+(.+?)(?:\s+\[.*\])?$")
    centers_re = re.compile(r"centers=(\d+)")

    runs = []
    current = None

    for raw in path.read_text().splitlines():
        if not raw.startswith("[time="):
            continue
        m = time_re.match(raw)
        if not m:
            continue
        t = float(m.group(1))
        msg = m.group(2)

        if msg.startswith("read input file"):
            if current:
                runs.append(current)
            current = {"centers": None, "times": {}}

        if current is None:
            continue

        if "computed point partitioning" in msg:
            cm = centers_re.search(raw)
            if cm:
                current["centers"] = int(cm.group(1))

        current["times"][phase_key(msg)] = t

    if current:
        runs.append(current)

    return runs


def summarize(runs):
    data = defaultdict(lambda: defaultdict(list))
    for r in runs:
        c = r.get("centers")
        if not c:
            continue
        for phase, t in r["times"].items():
            data[c][phase].append(t)

    centers = sorted(data.keys())
    summary = {phase: [] for phase in PHASE_ORDER}
    for c in centers:
        for phase in PHASE_ORDER:
            values = data[c].get(phase, [])
            if values:
                summary[phase].append(sum(values) / len(values))
            else:
                summary[phase].append(None)

    return centers, summary


def plot(centers, summary, output: Path, title: str | None):
    plt.figure(figsize=(9, 5.5))
    for phase in PHASE_ORDER:
        y = summary[phase]
        if all(v is None for v in y):
            continue
        plt.plot(centers, y, marker="o", label=phase)

    plt.xscale("log", base=2)
    plt.xlabel("Centers")
    plt.ylabel("Time (s)")
    if title:
        plt.title(title)
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(output, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results.m1.txt", help="input results file")
    ap.add_argument("-o", "--output", default="results.m1.png", help="output image file")
    ap.add_argument("--title", default="M1 results: phase times vs centers", help="plot title")
    args = ap.parse_args()

    runs = parse_runs(Path(args.path))
    if not runs:
        raise SystemExit("No runs parsed. Check input format.")

    centers, summary = summarize(runs)
    plot(centers, summary, Path(args.output), args.title)


if __name__ == "__main__":
    main()
