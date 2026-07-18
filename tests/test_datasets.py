import unittest

import numpy as np

from flux.datasets import GENRES, parse_ml100k

# 3 users, 3 items; user 1 has two positives (item 1 then item 2, later ts)
U_DATA = """1\t1\t5\t100
1\t2\t4\t200
2\t1\t4\t150
2\t3\t2\t160
3\t2\t5\t300
"""

def _item_line(item_id, genre_idx):
    flags = ["0"] * len(GENRES)
    flags[genre_idx] = "1"
    return "|".join([str(item_id), f"Movie {item_id}", "01-Jan-1995", "", ""]
                    + flags)

U_ITEM = "\n".join([_item_line(1, 1), _item_line(2, 5), _item_line(3, 8)])


class TestParseML100K(unittest.TestCase):
    def setUp(self):
        self.ml = parse_ml100k(U_DATA, U_ITEM)

    def test_shapes_and_indexing(self):
        self.assertEqual(self.ml.n_users, 3)
        self.assertEqual(self.ml.n_items, 3)
        self.assertEqual(self.ml.item_genres.shape, (3, len(GENRES)))
        self.assertEqual(self.ml.item_genres[0, 1], 1)  # 0-indexed items

    def test_low_ratings_excluded_from_positives(self):
        # user 2's rating-2 interaction (item 3) is not a positive
        all_pairs = np.vstack([self.ml.train, self.ml.test])
        self.assertNotIn([1, 2], all_pairs.tolist())

    def test_leave_latest_out_split(self):
        # user 1's latest positive (item 2, ts=200) must be in test,
        # the earlier one (item 1) in train
        self.assertIn([0, 1], self.ml.test.tolist())
        self.assertIn([0, 0], self.ml.train.tolist())
        # exactly one test row per user with positives
        self.assertEqual(len(self.ml.test), 3)

    def test_popularity_counts_all_ratings(self):
        # item 1 rated twice, item 2 twice, item 3 once
        self.assertEqual(self.ml.item_popularity.tolist(), [2, 2, 1])

    def test_popularity_group_labels(self):
        groups = self.ml.popularity_group(head_fraction=0.4)
        self.assertEqual(set(groups), {"head", "tail"})
        self.assertEqual(groups[2], "tail")  # least-rated item is tail


if __name__ == "__main__":
    unittest.main()
