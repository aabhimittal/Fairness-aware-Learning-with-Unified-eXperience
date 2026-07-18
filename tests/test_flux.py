import unittest

import numpy as np

from flux import (
    Content,
    FairReranker,
    FluxEngine,
    HyperbolicEmbedding,
    IPSDebiaser,
    PoincareBall,
    QuantumAttention,
)


class TestPoincareBall(unittest.TestCase):
    def setUp(self):
        self.ball = PoincareBall(c=1.0)

    def test_distance_symmetric_and_zero_on_diagonal(self):
        x = np.array([0.1, 0.2])
        y = np.array([-0.3, 0.05])
        self.assertAlmostEqual(self.ball.dist(x, y), self.ball.dist(y, x), places=6)
        self.assertAlmostEqual(self.ball.dist(x, x), 0.0, places=5)

    def test_boundary_points_are_far(self):
        near_origin = np.array([0.01, 0.0])
        near_boundary = np.array([0.999, 0.0])
        self.assertGreater(self.ball.dist(near_origin, near_boundary), 3.0)

    def test_exp_log_inverse(self):
        v = np.array([0.3, -0.1])
        back = self.ball.logmap0(self.ball.expmap0(v))
        np.testing.assert_allclose(back, v, atol=1e-5)

    def test_project_keeps_points_inside(self):
        x = self.ball.project(np.array([5.0, 5.0]))
        self.assertLess(np.linalg.norm(x), 1.0)


class TestHyperbolicEmbedding(unittest.TestCase):
    def test_training_pulls_positive_closer(self):
        emb = HyperbolicEmbedding(n_items=3, dim=8, seed=42)
        before_pos = emb.dist(0, 1)
        before_neg = emb.dist(0, 2)
        for _ in range(50):
            emb.train_step(0, positive=1, negative=2, lr=0.05)
        margin_before = before_neg - before_pos
        margin_after = emb.dist(0, 2) - emb.dist(0, 1)
        self.assertGreater(margin_after, margin_before)


class TestQuantumAttention(unittest.TestCase):
    def test_probabilities_normalized(self):
        qa = QuantumAttention(5, seed=1)
        self.assertAlmostEqual(qa.probabilities.sum(), 1.0, places=6)

    def test_collapse_respects_context(self):
        qa = QuantumAttention(4, seed=1)
        context = np.array([10.0, 0.0, 0.0, 0.0])
        w = qa.collapse(context)
        self.assertAlmostEqual(w.sum(), 1.0, places=6)
        self.assertEqual(int(np.argmax(w)), 0)

    def test_evolve_keeps_unit_norm(self):
        qa = QuantumAttention(3, seed=2)
        qa.evolve(np.array([1.0, -0.5, 0.2]))
        self.assertAlmostEqual(np.linalg.norm(qa.state), 1.0, places=6)

    def test_interference_bounded(self):
        qa = QuantumAttention(3, seed=3)
        val = qa.interference(0, 1)
        self.assertTrue(-1.0 <= val <= 1.0)


class TestIPSDebiaser(unittest.TestCase):
    def test_low_rank_click_upweighted(self):
        deb = IPSDebiaser(position_decay=0.5)
        clicks = np.array([1.0, 1.0])
        ranks = np.array([0, 4])  # rank 0 vs rank 4
        est = deb.debias(clicks, ranks)
        self.assertGreater(est[1], est[0])

    def test_clipping_bounds_weights(self):
        deb = IPSDebiaser(position_decay=0.5, clip=3.0)
        est = deb.debias(np.array([1.0]), np.array([10]))
        self.assertLessEqual(est[0], 3.0)

    def test_ctr_estimate_in_unit_interval(self):
        deb = IPSDebiaser()
        ctr = deb.estimate_ctr(np.array([1, 0, 1, 0]), np.array([0, 1, 2, 3]))
        self.assertTrue(0.0 <= ctr <= 1.0)


class TestFairReranker(unittest.TestCase):
    def test_rejects_bad_shares(self):
        with self.assertRaises(ValueError):
            FairReranker({"a": 0.9})

    def test_underexposed_group_gets_boosted(self):
        rr = FairReranker({"big": 0.5, "small": 0.5}, fairness_weight=5.0)
        # "big" items slightly outscore "small" ones everywhere
        scores = np.array([1.0, 0.99, 0.98, 0.8, 0.79, 0.78])
        groups = ["big", "big", "big", "small", "small", "small"]
        slate = rr.rerank(scores, groups, k=4)
        top_groups = [groups[i] for i in slate]
        self.assertIn("small", top_groups[:2])
        shares = rr.exposure_shares()
        self.assertGreater(shares["small"], 0.3)

    def test_zero_weight_is_pure_relevance(self):
        rr = FairReranker({"a": 0.5, "b": 0.5}, fairness_weight=0.0)
        scores = np.array([0.1, 0.9, 0.5])
        slate = rr.rerank(scores, ["a", "b", "a"])
        self.assertEqual(slate, [1, 2, 0])


class TestFluxEngine(unittest.TestCase):
    def _items(self):
        rng = np.random.default_rng(0)
        items = []
        for i in range(6):
            group = "emerging" if i % 2 else "established"
            items.append(Content(
                id=f"c{i}",
                creator_group=group,
                embedding=rng.normal(0, 0.1, size=4),
                interest_affinity=np.abs(rng.normal(size=3)),
                logged_clicks=float(i % 3),
                logged_rank=i,
            ))
        return items

    def test_rank_returns_full_permutation(self):
        eng = FluxEngine(3, {"emerging": 0.4, "established": 0.6}, seed=0)
        items = self._items()
        slate = eng.rank(np.zeros(4), items, context=np.ones(3))
        self.assertEqual(sorted(c.id for c in slate), sorted(c.id for c in items))

    def test_rank_top_k(self):
        eng = FluxEngine(3, {"emerging": 0.4, "established": 0.6}, seed=0)
        slate = eng.rank(np.zeros(4), self._items(), context=np.ones(3), k=3)
        self.assertEqual(len(slate), 3)

    def test_feedback_updates_state(self):
        eng = FluxEngine(3, {"emerging": 0.5, "established": 0.5}, seed=0)
        before = eng.attention.state.copy()
        eng.feedback(np.array([1.0, 0.0, -1.0]))
        self.assertFalse(np.allclose(before, eng.attention.state))


if __name__ == "__main__":
    unittest.main()
