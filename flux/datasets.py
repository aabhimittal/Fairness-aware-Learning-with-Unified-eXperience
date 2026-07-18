"""Real-dataset loaders for FLUX evaluations.

Currently ships MovieLens 100K (Herlocker et al., GroupLens): 100k ratings,
943 users, 1682 movies with genre labels — a natural fit for FLUX because
genres give a shallow interest hierarchy and rating counts give a
popularity split for fairness experiments.

Dataset terms of use: https://grouplens.org/datasets/movielens/
"""

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
_CACHE = Path.home() / ".cache" / "flux"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


@dataclass
class MovieLens:
    """MovieLens 100K, 0-indexed and split for ranking evaluation.

    train / test: arrays of (user, item) positive pairs (rating >= 4).
    The test set holds each user's most recent positive interaction
    (leave-latest-out), the standard implicit-feedback protocol.
    """

    n_users: int
    n_items: int
    train: np.ndarray  # (n, 2) user, item
    test: np.ndarray   # (m, 2) user, item
    item_genres: np.ndarray  # (n_items, 19) multi-hot
    item_popularity: np.ndarray  # (n_items,) rating counts over all data

    def popularity_group(self, head_fraction: float = 0.2) -> np.ndarray:
        """Label each item 'head' (top fraction by rating count) or 'tail' —
        the real-data analogue of established vs. emerging creators."""
        cutoff = np.quantile(self.item_popularity, 1.0 - head_fraction)
        return np.where(self.item_popularity >= cutoff, "head", "tail")


def _download(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "ml-100k.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(ML100K_URL, zip_path)
    return zip_path


def parse_ml100k(u_data: str, u_item: str) -> MovieLens:
    """Build a MovieLens split from the raw file contents (testable offline)."""
    rows = np.array([line.split("\t") for line in u_data.strip().splitlines()],
                    dtype=np.int64)  # user, item, rating, timestamp
    users, items, ratings, ts = rows.T
    users, items = users - 1, items - 1
    n_users, n_items = int(users.max()) + 1, int(items.max()) + 1

    genres = np.zeros((n_items, len(GENRES)), dtype=np.int64)
    for line in u_item.strip().splitlines():
        parts = line.split("|")
        genres[int(parts[0]) - 1] = [int(v) for v in parts[-len(GENRES):]]

    popularity = np.bincount(items, minlength=n_items)

    pos = ratings >= 4
    pu, pi, pt = users[pos], items[pos], ts[pos]
    # leave-latest-out: newest positive per user goes to test
    test_mask = np.zeros(len(pu), dtype=bool)
    for u in np.unique(pu):
        idx = np.nonzero(pu == u)[0]
        test_mask[idx[np.argmax(pt[idx])]] = True

    return MovieLens(
        n_users=n_users,
        n_items=n_items,
        train=np.stack([pu[~test_mask], pi[~test_mask]], axis=1),
        test=np.stack([pu[test_mask], pi[test_mask]], axis=1),
        item_genres=genres,
        item_popularity=popularity,
    )


def load_ml100k(cache_dir: Path | str | None = None) -> MovieLens:
    """Download (once, ~5MB) and parse MovieLens 100K."""
    cache = Path(cache_dir) if cache_dir else _CACHE
    with zipfile.ZipFile(_download(cache)) as zf:
        u_data = io.TextIOWrapper(zf.open("ml-100k/u.data"),
                                  encoding="latin-1").read()
        u_item = io.TextIOWrapper(zf.open("ml-100k/u.item"),
                                  encoding="latin-1").read()
    return parse_ml100k(u_data, u_item)
