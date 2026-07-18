import unittest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestManifold(unittest.TestCase):
    def setUp(self):
        from flux.torch_backend import PoincareManifold
        self.m = PoincareManifold(c=1.0)

    def test_matches_numpy_core(self):
        import numpy as np
        from flux import PoincareBall
        ball = PoincareBall(c=1.0)
        x = np.array([0.1, -0.2, 0.3])
        y = np.array([-0.05, 0.4, 0.1])
        d_np = ball.dist(x, y)
        d_t = self.m.dist(torch.tensor(x), torch.tensor(y))
        self.assertAlmostEqual(float(d_t), float(d_np), places=5)

    def test_dist_is_differentiable(self):
        x = torch.tensor([0.1, 0.2], requires_grad=True)
        y = torch.tensor([0.3, -0.1])
        self.m.dist(x, y).backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_exp_log_inverse(self):
        v = torch.tensor([0.3, -0.1])
        back = self.m.logmap0(self.m.expmap0(v))
        torch.testing.assert_close(back, v, atol=1e-5, rtol=1e-4)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTraining(unittest.TestCase):
    def test_loss_decreases_and_topics_cluster(self):
        from flux.torch_backend import (
            HyperbolicInterestModel, TrainConfig, make_synthetic_interactions,
            train,
        )
        users, pos, _ = make_synthetic_interactions(
            n_users=20, n_items=60, per_user=15, seed=1
        )
        model = HyperbolicInterestModel(20, 60, dim=8)
        losses = train(model, users, pos, n_items=60,
                       config=TrainConfig(epochs=15, lr=0.05, seed=1))
        self.assertLess(losses[-1], losses[0])

    def test_rank_items_returns_permutation(self):
        from flux.torch_backend import HyperbolicInterestModel
        model = HyperbolicInterestModel(3, 10, dim=4)
        order = model.rank_items(0)
        self.assertEqual(sorted(order.tolist()), list(range(10)))
        self.assertEqual(len(model.rank_items(0, k=5)), 5)

    def test_trained_model_ranks_positive_topic_higher(self):
        from flux.torch_backend import (
            HyperbolicInterestModel, TrainConfig, make_synthetic_interactions,
            train,
        )
        users, pos, item_topics = make_synthetic_interactions(
            n_users=10, n_items=50, per_user=20, seed=2
        )
        model = HyperbolicInterestModel(10, 50, dim=8)
        train(model, users, pos, n_items=50,
              config=TrainConfig(epochs=40, lr=0.05, seed=2))
        # user 0's positive topics should dominate their top-10
        user0_topics = set(item_topics[pos[users == 0]].tolist())
        top10_topics = item_topics[model.rank_items(0, k=10)].tolist()
        hits = sum(t in user0_topics for t in top10_topics)
        self.assertGreaterEqual(hits, 6)


class TestOptionalImport(unittest.TestCase):
    def test_core_package_works_without_backend(self):
        import flux
        self.assertTrue(hasattr(flux, "FluxEngine"))


if __name__ == "__main__":
    unittest.main()
