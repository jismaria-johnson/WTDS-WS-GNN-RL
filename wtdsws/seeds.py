"""Disjoint seed ranges: every graph seed and weight seed of a split is drawn from its range."""
SEED_RANGES = {
    "train": (0, 1_000_000_000),
    "val": (1_000_000_000, 1_100_000_000),
    "test": (1_200_000_000, 1_300_000_000),
    "pilot": (1_400_000_000, 1_500_000_000),          # pilot sweeps that choose plot settings; never test data
}
MASTER_SEEDS = {"val": 777_001, "test": 777_002}      # RNG seeds of the val/test samplers
TRAIN_MASTER = 1_000_003                               # training sampler: TRAIN_MASTER * (seed + 1)


def split_of(seed: int) -> str:
    for k, (lo, hi) in SEED_RANGES.items():
        if lo <= seed < hi:
            return k
    return "other"


def check_disjoint(verbose: bool = False) -> bool:
    items = sorted(SEED_RANGES.items(), key=lambda kv: kv[1])
    for (a, (_, a1)), (b, (b0, _)) in zip(items, items[1:]):
        if a1 > b0:
            raise AssertionError(f"seed ranges {a} and {b} overlap")
    if verbose:
        for k, (lo, hi) in items:
            print(f"  {k:<6} graph/weight seeds in [{lo:,}, {hi:,})")
        print("  seed ranges are pairwise disjoint: OK")
    return True
