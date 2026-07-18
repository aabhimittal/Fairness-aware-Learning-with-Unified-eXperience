"""End-to-end FLUX demo: rank a small content pool for one user across
two contexts (morning vs. evening) and show fairness of exposure."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from flux import Content, FluxEngine

INTERESTS = ["tech", "fitness", "cooking"]

def make_pool(rng):
    pool = []
    specs = [
        ("established", 0.05, 8.0, 0),   # broad, popular, shown at top
        ("established", 0.10, 6.0, 1),
        ("established", 0.15, 5.0, 2),
        ("emerging",    0.60, 1.0, 7),   # niche, barely ever shown
        ("emerging",    0.65, 0.0, 8),
        ("emerging",    0.70, 2.0, 9),
    ]
    for i, (group, radius, clicks, rank) in enumerate(specs):
        direction = rng.normal(size=4)
        direction /= np.linalg.norm(direction)
        pool.append(Content(
            id=f"post-{i}",
            creator_group=group,
            embedding=radius * direction,
            interest_affinity=rng.dirichlet(np.ones(len(INTERESTS))),
            logged_clicks=clicks,
            logged_rank=rank,
        ))
    return pool


def show(title, slate):
    print(f"\n{title}")
    for pos, c in enumerate(slate):
        print(f"  {pos + 1}. {c.id:8s} [{c.creator_group}]")


def main():
    rng = np.random.default_rng(7)
    engine = FluxEngine(
        n_interests=len(INTERESTS),
        target_shares={"established": 0.55, "emerging": 0.45},
        fairness_weight=2.0,
        seed=7,
    )
    pool = make_pool(rng)
    user = np.array([0.1, 0.05, -0.08, 0.02])

    morning = np.array([1.0, 2.0, 0.3])   # fitness-heavy context
    evening = np.array([0.5, 0.2, 2.5])   # cooking-heavy context

    show("Morning slate (fitness context):", engine.rank(user, pool, morning))
    engine.feedback(np.array([0.2, 1.0, -0.3]))  # user engaged with fitness
    show("Evening slate (cooking context, after feedback):",
         engine.rank(user, pool, evening))

    print("\nCumulative exposure shares vs targets:")
    for group, share in engine.reranker.exposure_shares().items():
        target = engine.reranker.target_shares[group]
        print(f"  {group:12s} {share:.2f} (target {target:.2f})")


if __name__ == "__main__":
    main()
