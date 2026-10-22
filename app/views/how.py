import streamlit as st

from components.theme import table

from wtdsws.weights import FORMULA, MEANING

PIPE = [
    ("01", "Graph", "edges; isolated vertices rejected (no TDS exists)"),
    ("02", "Weight scheme", "degree / inverse degree / random / mixed (integers 1–100), or custom weights > 0"),
    ("03", "Reduction", "the neighbour of every leaf is forced into S (degree-1 rule)"),
    ("04", "GNN encoder", "3-layer MPNN, hidden 64; 10 features, index 9 = w(v) / max w"),
    ("05", "Q-values", "graph summary (mean ‖ max) joined with each vertex embedding"),
    ("06", "Pick a vertex", "highest Q among legal actions: not in S and gain(v) > 0"),
    ("07", "RR + LS", "redundancy removal (most expensive first); local search with strictly improving swaps"),
    ("08", "Verified TDS", "is_total_dominating_set; W(S) recomputed from S and the weights"),
]

SCHEMES = [
    {"Scheme": "Degree", "Formula": FORMULA["degree"], "Meaning": MEANING["degree"]},
    {"Scheme": "Inverse degree", "Formula": FORMULA["inverse"], "Meaning": MEANING["inverse"]},
    {"Scheme": "Random", "Formula": FORMULA["random"], "Meaning": MEANING["random"]},
    {"Scheme": "Mixed", "Formula": FORMULA["mixed"], "Meaning": MEANING["mixed"]},
]

CLASSES = [
    {"Class": "ER", "Generator": "Erdős–Rényi G(n, p), average degree U[3, 15]",
     "Isolated-vertex repair": "attach as a leaf to a random vertex"},
    {"Class": "BA", "Generator": "Barabási–Albert, m ∈ {1..8} (m = 1: trees with many leaves)",
     "Isolated-vertex repair": "none needed (BA graphs are connected)"},
    {"Class": "Chordal", "Generator": "G(n, p) + chordal completion; random k-trees (k ≤ 4); unit / proper interval graphs",
     "Isolated-vertex repair": "leaf (G(n,p), k-tree); re-sample with a larger r (interval)"},
    {"Class": "Bipartite", "Generator": "random bipartite (n1 + n2 = n, edge prob. p); convex bipartite",
     "Isolated-vertex repair": "edge to the other side (random); re-sample longer intervals (convex)"},
]

CHANGES = [
    {"Aspect": "Algorithm", "wtds-gnn-rl": "3-layer MPNN + Double DQN, mask gain > 0, degree-1 rule, RR, LS",
     "This system": "unchanged"},
    {"Aspect": "Reward", "wtds-gnn-rl": "r = −w(v) / w̄ (γ = 1)", "This system": "unchanged"},
    {"Aspect": "Features", "wtds-gnn-rl": "10 node features, index 9 = w(v) / max w", "This system": "unchanged"},
    {"Aspect": "Weights", "wtds-gnn-rl": "uniform 1–100, uniform 1–10, degree + noise",
     "This system": "4 weight-assignment schemes: degree, inverse degree, random, mixed"},
    {"Aspect": "Graphs", "wtds-gnn-rl": "ER, BA, Watts–Strogatz, geometric, lattices",
     "This system": "4 classes: ER, BA, chordal (+ k-trees, interval), bipartite (+ convex)"},
    {"Aspect": "Training", "wtds-gnn-rl": "fine-tuned from the TDS model",
     "This system": "short fine-tuning from the wtds checkpoints (lr 5e-5, ε 0.2 → 0.05)"},
]


def render():
    st.markdown('<div class="section-title" style="font-size:1.6rem;">How it works</div>', unsafe_allow_html=True)
    st.markdown("""<p>Every vertex v has a weight w(v) &gt; 0. A set S is a <b>total dominating set</b> when every vertex,
including the vertices of S, has a neighbour in S. We look for the one with the smallest total weight
W(S) = Σ<sub>v∈S</sub> w(v). Such a set exists exactly when the graph has no isolated vertex.</p>
<p>Real costs are rarely uniform. Monitoring a busy router costs more than a quiet one; equipping a remote site costs
more than a hub with existing infrastructure. This system takes our weighted solver (wtds-gnn-rl) and shows it works
under four realistic ways of assigning weights and on four graph classes, <b>without changing the algorithm</b>.</p>""",
                unsafe_allow_html=True)
    cells = []
    for i, (n, t, s) in enumerate(PIPE):
        cls = "step loop" if n == "06" else "step"
        cells.append(f'<div class="{cls}"><div class="n">{n}</div><div class="t">{t}</div><div class="s">{s}</div></div>')
        if i < len(PIPE) - 1:
            cells.append('<div class="arrow">→</div>')
    st.markdown(f'<div class="pipe">{"".join(cells)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="caption">Steps 05–06 repeat (dashed) until every vertex has a chosen neighbour. The agent '
                'receives r = −w(v) / w̄ per vertex (w̄ = mean weight of the graph, γ = 1), so maximising the return '
                'minimises W(S).</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">The four weight schemes</div>', unsafe_allow_html=True)
    table(SCHEMES, ["Scheme", "Formula", "Meaning"])
    st.markdown('<div class="caption">d(v) = degree, Δ = max degree, δ = min degree (≥ 1). All weights are integers '
                'in [1, 100]; "mixed" reuses the same random draw as "random" for the same seed.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">The four graph classes</div>', unsafe_allow_html=True)
    table(CLASSES, ["Class", "Generator", "Isolated-vertex repair"])
    st.markdown('<div class="caption">Every generated graph is checked: chordal graphs with networkx.is_chordal, '
                'bipartite graphs with networkx.is_bipartite, convex bipartite graphs with a convexity checker, and no '
                'graph has an isolated vertex. Total domination is NP-hard on general bipartite and chordal graphs '
                '(Laskar et al., 1984), so a learned heuristic is meaningful there. Proper interval and convex '
                'bipartite graphs have exact polynomial algorithms for weighted domination variants (Lee, 2025); they '
                'are reported only as a sanity check.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">What changed from wtds-gnn-rl</div>', unsafe_allow_html=True)
    table(CHANGES, ["Aspect", "wtds-gnn-rl", "This system"])

    st.markdown('<div class="section-title">References</div>', unsafe_allow_html=True)
    st.markdown("""<p class="caption">M. Chen, S. Liu, W. He (2024). <i>Learn to solve dominating set problem with GNN and
reinforcement learning.</i> Applied Mathematics and Computation 474:128717 (base paper).<br>
C.-M. Lee (2025). <i>Exploring Dominating Functions and Their Complexity in Subclasses of Weighted Chordal Graphs and
Bipartite Graphs.</i> Mathematics 13(3):403, doi:10.3390/math13030403.<br>
R. Laskar, J. Pfaff, S. M. Hedetniemi, S. T. Hedetniemi (1984). <i>On the algorithmic complexity of total
domination.</i> SIAM J. Algebraic Discrete Methods 5:420–425.</p>""", unsafe_allow_html=True)
