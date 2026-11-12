"""Write random scene jobs from a generation config. Runs in sip."""

import argparse
import json
from collections import Counter
from pathlib import Path

from synperc.generation.sampling import load_config, sample_scene


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--out-root", type=Path, default=Path("data/synthetic/gen_v1"))
    a = p.parse_args()

    cfg = load_config(a.config)
    counts = Counter()
    for idx in range(a.start, a.start + a.count):
        job = sample_scene(cfg, idx, cfg["seed"])
        for o in job["objects"]:
            if not Path(o["mesh"]).exists():
                raise FileNotFoundError(o["mesh"])
            counts[o["obj_id"]] += 1
        out = a.out_root / f"scene_{idx:06d}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    print(f"wrote {a.count} jobs to {a.out_root} (indices {a.start}..{a.start + a.count - 1})")
    print(f"object instances per id: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()