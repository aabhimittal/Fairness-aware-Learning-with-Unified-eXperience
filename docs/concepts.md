# Concepts

## Hyperbolic interest space

User interests are *hierarchical*: loving "dogs" is broader than loving
"golden retrievers". Flat Euclidean space needs many dimensions to keep a
deep hierarchy's branches apart, but the Poincaré ball grows exponentially
toward its boundary — matching the branching factor of interest taxonomies.
Broad interests embed near the origin; niche interests near the edge.
`PoincareBall.hierarchy_level()` reads this radius directly.

The core (`flux.hyperbolic`) implements Möbius addition, geodesic distance,
exp/log maps, and Riemannian SGD in pure NumPy. The torch backend
(`flux.torch_backend.manifold`) provides the same operations with autograd.

## Quantum-inspired attention

A user's interests are a **superposition**: unit-norm complex amplitudes
over interest dimensions. Context (time of day, session, social signals)
acts as a measurement operator that *collapses* the state (Born rule:
probabilities are squared amplitudes) into the interest profile used for
scoring. Because amplitudes are complex, interests **interfere** — related
interests reinforce (phases align), conflicting interests suppress each
other — expressiveness a plain softmax over real weights lacks.

## Causal debiasing

Observed clicks confound preference with exposure: rank-1 content gets
clicked regardless of quality. `IPSDebiaser` reweights each interaction by
the inverse examination propensity (`decay^rank`), recovering what users
*would* click if everything were examined equally. Weights are clipped to
bound variance on sparse logs.

## Fairness-aware re-ranking

Pure relevance ranking concentrates exposure on already-popular creators.
`FairReranker` greedily rebuilds each slate maximizing
`relevance + weight × group_deficit`, where deficit is how far a creator
group's *cumulative, position-discounted* exposure lags its target share.
Because exposure accumulates across requests, parity is **amortized over
time** rather than forced per-slate — individual slates stay coherent while
the system as a whole hits its targets.
