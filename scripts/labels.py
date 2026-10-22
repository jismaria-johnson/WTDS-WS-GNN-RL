"""Display labels: the method is called "Proposed" in every figure and table (decisions R40).

REPL is the ordered list of display-string substitutions applied to the report scripts (and, for the number check,
to the archived outputs: scripts/relabel_check.py). Data keys such as "ours_rr" in CSVs are never renamed.
"""
REPL = [
    ("Ours & Greedy+RR & Ours & Greedy+RR & Ours beats", "Proposed + RR & Greedy + RR & Proposed + RR & Greedy + RR & Proposed beats"),
    ("Ours (GNN+RL + RR, 2 seeds)", "Proposed + RR (2 seeds)"),
    ("Ours beats Greedy", "Proposed beats Greedy"),
    ("ours: GNN-guided restarts", "Proposed (guided restarts)"),
    ("ours: sampled rollouts+RR+LS", "Proposed (best-of-k) + LS"),
    ("ours: greedy rollout+RR+LS", "Proposed + RR + LS"),
    ("ours: rollouts+RR+LS", "Proposed (best-of-k) + LS"),
    ("ours: rollouts+RR", "Proposed (best-of-k)"),
    ("Ours (rollouts + RR + LS)", "Proposed (best-of-k) + LS"),
    ("CP-SAT (8 workers)", "CP-SAT"),
    ("ours+RR+LS", "Proposed + RR + LS"),
    ("Ours+RR", "Proposed + RR"),
    ("ours+RR", "Proposed + RR"),
    ("greedy+RR+LS", "Greedy + RR + LS"),
    ("greedy+RR", "Greedy + RR"),
    ('"Ours + RR"', '"Proposed + RR"'),
    ("Ours = ", "Proposed = "),
    ("ours = ", "Proposed = "),
    ("Ours (mean", "Proposed (mean"),
    ("ours: mean", "Proposed: mean"),
    ("ours better", "Proposed better"),
    ("ours lower", "Proposed lower"),
    ("for ours", "for Proposed"),
    ("of ours", "of Proposed"),
    ("where ours is", "where Proposed is"),
    ("ours gap", "Proposed gap"),
    ("ours W/T/L", "Proposed W/T/L"),
    ("Ours optimal", "Proposed optimal"),
    ("D ours", "D Proposed"),
    ("Delta ours", "Delta Proposed"),
    ("Ours vs greedy", "Proposed vs Greedy"),
    ("ours vs greedy", "Proposed vs Greedy"),
    ("gap ours", "gap Proposed"),
    ("W ours", "W Proposed"),
    ("time ours", "time Proposed"),
]
# method keys -> display names (for tables that print a method key)
DISP = {"ours_rr": "Proposed + RR", "ours_rr_ls": "Proposed + RR + LS", "ours_raw": "Proposed",
        "greedy": "Greedy", "greedy_rr": "Greedy + RR", "greedy_rr_ls": "Greedy + RR + LS", "zero_rr": "zero-shot + RR"}


def relabel(text: str) -> str:
    for a, b in REPL:
        text = text.replace(a, b)
    return text
