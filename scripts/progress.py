"""Show progress of a running run_forecasting.py experiment from its log.

Usage:
    python scripts/progress.py                      # results_v2/full_star.log
    python scripts/progress.py results_v2/other.log
"""

import re
import sys
from collections import defaultdict
from datetime import datetime

LINE = re.compile(r"^(\S+ \S+),\d+ INFO run_forecasting: H=(\d+) seed=(\d+) fold=(\d+)  Oracle")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path =sys.argv[1] if len(sys.argv) > 1 else "results_v2/full_star.log"
    text = open(path, encoding="utf-8", errors="replace").read().splitlines()

    cfg = {}
    try:
        import json, os
        cfg = json.load(open(os.path.join(os.path.dirname(path), os.path.basename(path)[:-4], "config.json")))
    except Exception:
        pass
    horizons = cfg.get("horizons", [5, 15, 30, 60])
    seeds = cfg.get("seeds", [0, 1, 2, 3, 4])
    folds = cfg.get("folds", 4) if cfg.get("split", "star") == "star" else 1
    total = len(horizons) * len(seeds) * folds

    done = [(datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S"), int(m[2])) for m in map(LINE.match, text) if m]
    per_h = defaultdict(int)
    for _, h in done:
        per_h[h] += 1

    print(f"Log: {path}")
    print(f"Hotovo: {len(done)} / {total} běhů ({100 * len(done) / total:.0f} %)")
    for h in horizons:
        n = per_h[h]
        bar = "#" * int(20 * n / (len(seeds) * folds))
        print(f"  horizont {h:>3} kadencí: {n:>2}/{len(seeds) * folds}  [{bar:<20}]")
    if text:
        print(f"Poslední záznam v logu: {text[-1][:19]}")
    if len(done) >= 2 and len(done) < total:
        # recent pace (last 4 runs) is the best guide; longer horizons are slower
        recent = done[-5:]
        pace = (recent[-1][0] - recent[0][0]) / (len(recent) - 1)
        eta = done[-1][0] + pace * (total - len(done))
        print(f"Tempo posledních běhů: {pace.total_seconds() / 60:.0f} min/běh")
        print(f"Odhad dokončení: {eta:%d.%m. %H:%M} (platí jen při připojené nabíječce)")
    elif len(done) == total:
        print("Experiment je dokončený.")


if __name__ == "__main__":
    main()
