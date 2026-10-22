# Our exact DP for unit-interval graphs

`wtdsws/baselines/exact_special.py::unit_interval_tds`, labelled **"our exact DP for unit-interval graphs"**.

## Why our own DP

The literature gives exact polynomial algorithms for minimum(-weight) total domination on interval graphs:
Bertossi & Gori (1988), *SIAM J. Discrete Math.* 1(3), O(n log n), and M.-S. Chang (1998), *SIAM J. Computing*
27(6). We did not reproduce either one exactly. The panel uses **unit-interval graphs** (the project's
`unit_interval` generator: points x_v in [0, 1), edge iff |x_u − x_v| < r), which are the proper interval graphs.
On this class there is a short DP over the sorted interval model, and we prove it correct below. It does **not**
handle general interval graphs: it rejects any input whose sorted order is not a proper interval ordering.

## Setting

Sort the vertices by their point, giving 0, 1, …, n−1. In a unit-interval graph every closed neighbourhood is a
contiguous block, N[i] = [L(i), R(i)]. The code checks this, and also checks that L and R are non-decreasing.
Contiguity gives the **umbrella property**: if u < w < v and uv ∈ E, then uw ∈ E and wv ∈ E.

## Characterisation

Let S = {s_1 < s_2 < … < s_k}. S is a total dominating set if and only if all four of these hold:

1. L(s_1) = 0 (every vertex before s_1 is a neighbour of s_1);
2. R(s_k) = n − 1 (every vertex after s_k is a neighbour of s_k);
3. L(s_{j+1}) ≤ R(s_j) + 1 for every j (no undominated vertex strictly between consecutive chosen vertices);
4. every s_j is adjacent to s_{j−1} or to s_{j+1} (the chosen vertices are dominated too; the neighbourhood is
   open, so s_j does not dominate itself).

**Proof.** By the umbrella property, the chosen vertex nearest to a vertex v on each side is the only one that
needs checking:
- Suppose v ∉ S and s_j < v < s_{j+1}. If some s_i with i < j is adjacent to v, then s_i < s_j < v gives
  s_j ~ v. The same holds on the right side. So v is dominated iff v ≤ R(s_j) or v ≥ L(s_{j+1}). Every v in the
  open range (s_j, s_{j+1}) is dominated iff L(s_{j+1}) ≤ R(s_j) + 1: if this fails, v = R(s_j) + 1 lies in the
  range and is undominated.
- Vertices before s_1 and after s_k give conditions 1 and 2 in the same way.
- A chosen vertex s_j is dominated iff some other s_i is adjacent to it. By the umbrella property this holds iff
  s_{j−1} or s_{j+1} is adjacent to it, which is condition 4. ∎

## DP

f[t][h] = the minimum cost of a chosen prefix whose last chosen vertex is t, such that every vertex < t is
dominated. The flag h = 1 means t is already dominated by its predecessor. Costs are pairs (W, |S|) compared
lexicographically, so the DP returns a minimum-weight set and, among those, a smallest one.

- **Start:** f[t][0] = (w(t), 1) if L(t) = 0.
- **Transition** s → t (s < t, L(t) ≤ R(s) + 1): let a = [t ≤ R(s)] (s and t are adjacent). If h = 0, then t
  must dominate s, so a = 1 is required. Then f[t][a] ← min(f[t][a], f[s][h] + (w(t), 1)).
- **End:** the answer is min over t with R(t) = n − 1 of f[t][1]. The last vertex must be dominated by its
  predecessor.

Because R is non-decreasing, once R(s) + 1 < L(t) holds for some s, it holds for every smaller s too, so the inner
loop stops there. The running time is O(n + |E|) transitions; it is O(n²) only in the worst case. At n = 1000 it
takes about 0.01 s. Reconstruction follows the stored predecessors. The returned set goes through
`is_total_dominating_set`, and its weight is recomputed.

## Validation (tests/test_exact_special.py, results/exact_validation.json)

- 500 instances (100 graphs × 5 weightings: unit, degree, inverse, random, mixed), n = 3–12, against brute force.
  Both W and the minimum |S| among the minimum-weight sets are compared: 0 mismatches.
- 300 instances (60 graphs × 5 weightings), n = 50–200, against proven-optimal CP-SAT, using the plain
  min-weight ILP and the lexicographic ILP: 0 mismatches, 0 unproven.
- In the main experiment, the lexicographic CP-SAT ILP also runs on every unit-interval graph (`ilp_check` rows
  of results/exact_vs_ours.csv). The run aborts if a proven ILP ever disagrees with the DP.
