"""Real-dataset loaders for FLUX evaluations.

Ships MovieLens 100K and 1M (GroupLens). Both give a shallow interest
hierarchy via genres and a popularity split for fairness experiments.

Dataset terms of use: https://grouplens.org/datasets/movielens/
"""

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
_CACHE = Path.home() / ".cache" / "flux"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


@dataclass
class MovieLens:
    """A MovieLens variant, 0-indexed and split for ranking evaluation.

    train / test: arrays of (user, item) positive pairs (rating >= 4).
    The test set holds each user's most recent positive interaction
    (leave-latest-out), the standard implicit-feedback protocol.
    """

    n_users: int
    n_items: int
    train: np.ndarray  # (n, 2) user, item
    test: np.ndarray   # (m, 2) user, item
    item_genres: np.ndarray  # (n_items, len(GENRES)) multi-hot
    item_popularity: np.ndarray  # (n_items,) rating counts over all data

    def popularity_group(self, head_fraction: float = 0.2) -> np.ndarray:
        """Label each item 'head' (top fraction by rating count) or 'tail' —
        the real-data analogue of established vs. emerging creators."""
        cutoff = np.quantile(self.item_popularity, 1.0 - head_fraction)
        return np.where(self.item_popularity >= cutoff, "head", "tail")


def _download(url: str, filename: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / filename
    if not zip_path.exists():
        urllib.request.urlretrieve(url, zip_path)
    return zip_path


def _build_split(users: np.ndarray, items: np.ndarray, ratings: np.ndarray,
                 ts: np.ndarray, item_genres: np.ndarray) -> MovieLens:
    """Shared split logic: positives = rating >= 4, leave-latest-out per user."""
    n_users = int(users.max()) + 1
    n_items = item_genres.shape[0]
    popularity = np.bincount(items, minlength=n_items)

    pos = ratings >= 4
    pu, pi, pt = users[pos], items[pos], ts[pos]
    test_mask = np.zeros(len(pu), dtype=bool)
    for u in np.unique(pu):
        idx = np.nonzero(pu == u)[0]
        test_mask[idx[np.argmax(pt[idx])]] = True

    return MovieLens(
        n_users=n_users,
        n_items=n_items,
        train=np.stack([pu[~test_mask], pi[~test_mask]], axis=1),
        test=np.stack([pu[test_mask], pi[test_mask]], axis=1),
        item_genres=item_genres,
        item_popularity=popularity,
    )


def parse_ml100k(u_data: str, u_item: str) -> MovieLens:
    """Build a split from raw ML-100K file contents (testable offline)."""
    rows = np.array([line.split("\t") for line in u_data.strip().splitlines()],
                    dtype=np.int64)  # user, item, rating, timestamp
    users, items, ratings, ts = rows.T
    users, items = users - 1, items - 1

    n_items = int(items.max()) + 1
    genres = np.zeros((n_items, len(GENRES)), dtype=np.int64)
    for line in u_item.strip().splitlines():
        parts = line.split("|")
        genres[int(parts[0]) - 1] = [int(v) for v in parts[-len(GENRES):]]

    return _build_split(users, items, ratings, ts, genres)


def parse_ml1m(ratings_dat: str, movies_dat: str) -> MovieLens:
    """Build a split from raw ML-1M file contents (testable offline).

    ML-1M movie ids are sparse (max 3952 over ~3883 movies), so items are
    re-indexed densely; genres are named rather than multi-hot columns.
    """
    genre_idx = {g: i for i, g in enumerate(GENRES)}
    raw_genres = {}
    for line in movies_dat.strip().splitlines():
        movie_id, _title, names = line.split("::")
        flags = np.zeros(len(GENRES), dtype=np.int64)
        for name in names.split("|"):
            flags[genre_idx.get(name, 0)] = 1
        raw_genres[int(movie_id)] = flags

    item_map = {mid: i for i, mid in enumerate(sorted(raw_genres))}
    genres = np.stack([raw_genres[mid] for mid in sorted(raw_genres)])

    rows = np.array([line.split("::") for line in
                     ratings_dat.strip().splitlines()], dtype=np.int64)
    users, raw_items, ratings, ts = rows.T
    users = users - 1
    items = np.array([item_map[i] for i in raw_items])

    return _build_split(users, items, ratings, ts, genres)


def load_ml100k(cache_dir: Path | str | None = None) -> MovieLens:
    """Download (once, ~5MB) and parse MovieLens 100K."""
    cache = Path(cache_dir) if cache_dir else _CACHE
    with zipfile.ZipFile(_download(ML100K_URL, "ml-100k.zip", cache)) as zf:
        u_data = io.TextIOWrapper(zf.open("ml-100k/u.data"),
                                  encoding="latin-1").read()
        u_item = io.TextIOWrapper(zf.open("ml-100k/u.item"),
                                  encoding="latin-1").read()
    return parse_ml100k(u_data, u_item)


def load_ml1m(cache_dir: Path | str | None = None) -> MovieLens:
    """Download (once, ~6MB) and parse MovieLens 1M."""
    cache = Path(cache_dir) if cache_dir else _CACHE
    with zipfile.ZipFile(_download(ML1M_URL, "ml-1m.zip", cache)) as zf:
        ratings = io.TextIOWrapper(zf.open("ml-1m/ratings.dat"),
                                   encoding="latin-1").read()
        movies = io.TextIOWrapper(zf.open("ml-1m/movies.dat"),
                                  encoding="latin-1").read()
    return parse_ml1m(ratings, movies)


LOADERS = {"ml-100k": load_ml100k, "ml-1m": load_ml1m}
