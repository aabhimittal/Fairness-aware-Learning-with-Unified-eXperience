import unittest

import numpy as np

from flux.datasets import GENRES, parse_ml1m

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# sparse movie ids (10, 20, 30) exercise the dense re-indexing
MOVIES_1M = """10::Movie A (1995)::Action|Comedy
20::Movie B (1996)::Drama
30::Movie C (1997)::Sci-Fi|Thriller"""

RATINGS_1M = """1::10::5::100
1::20::4::200
2::10::4::150
2::30::2::160
3::20::5::300"""


class TestParseML1M(unittest.TestCase):
    def setUp(self):
        self.ml = parse_ml1m(RATINGS_1M, MOVIES_1M)

    def test_dense_reindexing(self):
        self.assertEqual(self.ml.n_items, 3)
        self.assertEqual(self.ml.item_genres.shape, (3, len(GENRES)))
        # movie 10 -> index 0 with Action + Comedy set
        self.assertEqual(self.ml.item_genres[0, GENRES.index("Action")], 1)
        self.assertEqual(self.ml.item_genres[0, GENRES.index("Comedy")], 1)

    def test_split_matches_ml100k_protocol(self):
        # user 1's latest positive (movie 20 -> idx 1) in test
        self.assertIn([0, 1], self.ml.test.tolist())
        self.assertIn([0, 0], self.ml.train.tolist())
        # user 2's rating-2 interaction is excluded from positives
        all_pairs = np.vstack([self.ml.train, self.ml.test]).tolist()
        self.assertNotIn([1, 2], all_pairs)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestEvaluationModule(unittest.TestCase):
    def test_ranking_metrics_on_perfect_model(self):
        from flux.datasets import MovieLens
        from flux.evaluation import ranking_metrics

        class Oracle:
            """Ranks the test target first for every user."""
            def rank_items(self, u):
                order = [u] + [i for i in range(4) if i != u]
                return torch.tensor(order)

        data = MovieLens(
            n_users=2, n_items=4,
            train=np.array([[0, 2], [1, 3]]),
            test=np.array([[0, 0], [1, 1]]),
            item_genres=np.zeros((4, len(GENRES)), dtype=np.int64),
            item_popularity=np.array([2, 2, 1, 1]),
        )
        m = ranking_metrics(Oracle(), data, k=10)
        self.assertEqual(m["hr@10"], 1.0)
        self.assertEqual(m["ndcg@10"], 1.0)

    def test_cli_parses_args(self):
        from flux.cli import main
        with self.assertRaises(SystemExit):  # bad dataset rejected by argparse
            main(["--dataset", "bogus"])


@unittest.skipUnless(HAS_FASTAPI, "fastapi/httpx not installed")
class TestServing(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from flux import Content, FluxEngine
        from flux.serving import create_app

        rng = np.random.default_rng(0)
        engine = FluxEngine(3, {"a": 0.5, "b": 0.5}, seed=0)
        catalog = [Content(id=f"c{i}", creator_group="a" if i % 2 else "b",
                           embedding=rng.normal(0, 0.1, size=4),
                           interest_affinity=np.abs(rng.normal(size=3)))
                   for i in range(6)]
        self.client = TestClient(create_app(engine, catalog))

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["catalog_size"], 6)

    def test_rank_and_fairness(self):
        r = self.client.post("/rank", json={
            "user_point": [0.0, 0.0, 0.0, 0.0],
            "context": [1.0, 1.0, 1.0], "k": 4})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["slate"]), 4)
        shares = self.client.get("/fairness").json()["exposure_shares"]
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=5)

    def test_bad_context_rejected(self):
        r = self.client.post("/rank", json={
            "user_point": [0.0] * 4, "context": [1.0]})
        self.assertEqual(r.status_code, 422)

    def test_feedback_returns_probabilities(self):
        r = self.client.post("/feedback", json={"engagement": [1.0, 0.0, -1.0]})
        self.assertEqual(r.status_code, 200)
        self.assertAlmostEqual(sum(r.json()["interest_probabilities"]), 1.0,
                               places=3)


if __name__ == "__main__":
    unittest.main()
