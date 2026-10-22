"""Minimum vertex-Weighted Total Dominating Set with GNN + Double DQN,
under four weight-assignment schemes and four graph classes (wtds-ws)."""
from .checker import assert_tds, is_total_dominating_set, verify
from .graph import Graph, InvalidWeightError, NoTotalDominatingSetError
from .graph_classes import CLASSES, generate_graph
from .result import SolveResult
from .weights import SCHEMES, assign_weights
# `wtdsws.solve` is a callable module (see wtdsws/solve.py), so `from wtdsws import solve; solve(G, w)`
# works and `python -m wtdsws.solve` runs the CLI without importing it twice.

__version__ = "0.1.0"


__all__ = ["Graph", "NoTotalDominatingSetError", "InvalidWeightError", "SolveResult", "solve", "verify",
           "assert_tds", "is_total_dominating_set", "assign_weights", "generate_graph", "SCHEMES", "CLASSES"]
