#!/usr/bin/env python3
"""
07_parse_sniper_results.py — Cache Hierarchy & Energy Telemetry Parser

Parses output directories under sniper_out/<workload>_<policy>/ for
workload in {spmv, bfs, join, pagerank} and policy in {base, ots} from
Sniper (sim.out / McPAT) or Valgrind Cachegrind (cachegrind.out).
Reports both:
  - l3_local_miss_pct  : LLC misses / LLC accesses (D1 misses)
  - l3_global_miss_pct : LLC misses / total data references (Dr + Dw)
And applies a unified energy model across both tools when McPAT powerstack.txt
is not present: E = (Ir * 0.42 nJ + D1_misses * 1.8 nJ + LLC_misses * 35.0 nJ).
"""
import os
import re
import sys
import pandas as pd


def compute_unified_energy_mj(ir, d1_misses, llc_misses):
    return (ir * 0.42e-9 + d1_misses * 1.8e-9 + llc_misses * 35.0e-9) * 1e3


def parse_cachegrind_out(path):
    """Parse Valgrind cachegrind.out summary totals line."""
    if not os.path.isfile(path):
        return None
    events = []
    summary = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("events:"):
                events = line.split(":", 1)[1].strip().split()
            elif line.startswith("summary:"):
                summary = [int(x) for x in line.split(":", 1)[1].strip().split()]
    if not events or len(events) != len(summary):
        return None
    ev = dict(zip(events, summary))
    ir = ev.get("Ir", 0)
    dr = ev.get("Dr", 0)
    dw = ev.get("Dw", 0)
    d1m = ev.get("D1mr", 0) + ev.get("D1mw", 0)
    dlm = ev.get("DLmr", 0) + ev.get("DLmw", 0)
    d_refs = max(1, dr + dw)

    l1d_miss_pct = 100.0 * d1m / d_refs
    l3_local_miss_pct  = 100.0 * dlm / max(1, d1m)
    l3_global_miss_pct = 100.0 * dlm / d_refs
    energy_mj = compute_unified_energy_mj(ir, d1m, dlm)
    return {
        "instructions": ir,
        "data_refs": d_refs,
        "l1d_miss_pct": round(l1d_miss_pct, 2),
        "l3_local_miss_pct": round(l3_local_miss_pct, 2),
        "l3_global_miss_pct": round(l3_global_miss_pct, 3),
        "llc_misses": dlm,
        "energy_mj": round(energy_mj, 3),
        "simulator_source": "valgrind_cachegrind",
    }


def parse_sim_out(path):
    """Parse Sniper's formatted ASCII table in sim.out."""
    if not os.path.isfile(path):
        return None
    metrics = {}
    section = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("-"):
                continue
            if "|" not in line:
                section = line.lower()
                continue
            parts = [p.strip() for p in line.split("|")]
            key = parts[0].lower()
            vals = []
            for tok in parts[1:]:
                clean = tok.replace("%", "").replace(",", "").strip()
                m = re.match(r"^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", clean)
                if m:
                    vals.append(float(m.group(0)))
            if not vals:
                continue
            full_key = f"{section}::{key}" if section else key
            metrics[full_key] = vals
            metrics[key] = vals
    if not metrics:
        return None

    def avg_or_sum(vals, mode="avg"):
        if not vals:
            return 0.0
        return sum(vals) / len(vals) if mode == "avg" else sum(vals)

    instr = int(avg_or_sum(metrics.get("instructions", []), "sum"))
    d_refs = max(1, instr // 3)
    l1d_miss_pct = avg_or_sum(metrics.get("cache l1-d::miss rate", metrics.get("l1-d::miss rate", [0.0])), "avg")
    l3_local_miss_pct = avg_or_sum(metrics.get("cache l3::miss rate", metrics.get("l3::miss rate", [0.0])), "avg")
    dram_acc = int(avg_or_sum(metrics.get("dram::num accesses", metrics.get("num dram accesses", [0.0])), "sum"))
    d1_misses = int(d_refs * (l1d_miss_pct / 100.0))
    l3_global_miss_pct = 100.0 * dram_acc / d_refs
    energy_mj = compute_unified_energy_mj(instr, d1_misses, dram_acc)
    return {
        "instructions": instr,
        "data_refs": d_refs,
        "l1d_miss_pct": round(l1d_miss_pct, 2),
        "l3_local_miss_pct": round(l3_local_miss_pct, 2),
        "l3_global_miss_pct": round(l3_global_miss_pct, 3),
        "llc_misses": dram_acc,
        "energy_mj": round(energy_mj, 3),
        "simulator_source": "sniper_energy_model",
    }


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "sniper_out"
    out_csv  = sys.argv[2] if len(sys.argv) > 2 else "sniper_cache_energy.csv"

    rows = []
    for wl, label in [("spmv", "SpMV"), ("bfs", "BFS"), ("join", "HashJoin"), ("pagerank", "PageRank")]:
        for pol in ["base", "ots"]:
            rdir = os.path.join(base_dir, f"{wl}_{pol}")
            res = parse_sim_out(os.path.join(rdir, "sim.out"))
            if res is None:
                res = parse_cachegrind_out(os.path.join(rdir, "cachegrind.out"))
            if res is None:
                continue
            row = {"workload": label, "policy": pol}
            row.update(res)
            rows.append(row)

    if not rows:
        print(f"No Sniper or Cachegrind runs found in {base_dir}/. Run ./run_stage3_sniper.sh first.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("=== Stage 3: Cache Hierarchy Miss Rate & Energy Summary ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
