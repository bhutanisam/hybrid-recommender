"""
Hybrid Recommender (Collaborative Filtering + Content-Based) with cold-start handling.
Dataset: MovieLens 1M.   Run:  python recommender.py data/ml-1m
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

# ---------------- Settings (easy to change) ----------------
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/ml-1m"
K = 10                 # recommend top-10 movies
N_FACTORS = 30         # SVD latent factors
CONF_K = 10            # blending speed: weight = n / (n + CONF_K)
COLD_USER_FRAC = 0.2   # 20% of users are turned into "cold" users
COLD_N = 3             # cold users keep only 3 training ratings (<5)
COLD_ITEM_MAX = 5      # item with <5 training ratings = cold item
LIKE = 4               # rating >= 4 means "user liked it"
SEED = 42

os.makedirs("results", exist_ok=True)

# ---------------- 1. Load data ----------------
ratings = pd.read_csv(f"{DATA_DIR}/ratings.dat", sep="::", engine="python",
                      names=["user", "item", "rating", "ts"])
movies = pd.read_csv(f"{DATA_DIR}/movies.dat", sep="::", engine="python",
                     names=["item", "title", "genres"], encoding="latin-1")

# convert ids to 0..n-1 positions
movies = movies.reset_index(drop=True)
item_pos = {m: i for i, m in enumerate(movies["item"])}
user_ids = ratings["user"].unique()
user_pos = {u: i for i, u in enumerate(user_ids)}
ratings["u"] = ratings["user"].map(user_pos)
ratings["i"] = ratings["item"].map(item_pos)
n_users, n_items = len(user_ids), len(movies)
print(f"Users: {n_users}, Movies: {n_items}, Ratings: {len(ratings)}")

# ---------------- 2. Train/test split (+ cold-start users) ----------------
# Each user's ratings are sorted by time. Warm users: first 80% = train, last 20% = test.
# Cold users (random 20%): only first 3 ratings = train, everything else = test.
rng = np.random.default_rng(SEED)
cold_users = set(rng.choice(np.arange(n_users), int(n_users * COLD_USER_FRAC), replace=False))
ratings = ratings.sort_values(["u", "ts"], kind="stable").reset_index(drop=True)
ratings["rank"] = ratings.groupby("u").cumcount()
ratings["n"] = ratings.groupby("u")["u"].transform("size")
is_cold = ratings["u"].isin(cold_users).values
in_train = np.where(is_cold, ratings["rank"].values < COLD_N,
                    ratings["rank"].values < 0.8 * ratings["n"].values)
train, test = ratings[in_train], ratings[~in_train]
print(f"Train: {len(train)}, Test: {len(test)}, Cold users: {len(cold_users)}")

rows = train["u"].values
cols = train["i"].values
vals = train["rating"].values.astype(float)
user_cnt = np.bincount(rows, minlength=n_users)    # ratings per user (train)
item_cnt = np.bincount(cols, minlength=n_items)    # ratings per movie (train)
user_mean = np.bincount(rows, weights=vals, minlength=n_users) / np.maximum(user_cnt, 1)

# centered rating matrix (rating minus the user's average)
C = csr_matrix((vals - user_mean[rows], (rows, cols)), shape=(n_users, n_items))
seen = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n_users, n_items)).toarray()

# what each user actually liked in the test set
test_liked = test[test["rating"] >= LIKE].groupby("u")["i"].apply(set).to_dict()
cold_items = set(np.where(item_cnt < COLD_ITEM_MAX)[0])

# ---------------- 3. Collaborative filtering (SVD) ----------------
U, s, Vt = svds(C, k=N_FACTORS)
cf_scores = ((U * s) @ Vt).astype(np.float32)

# ---------------- 4. Content-based (genre + decade) ----------------
genre = movies["genres"].str.get_dummies(sep="|")
year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
decade = (year // 10 * 10).fillna(0).astype(int).astype(str).str.get_dummies()
X = normalize(pd.concat([genre, decade], axis=1).values.astype(float))   # movie features
profile = normalize(C @ X)                    # user taste = average of features of movies they liked/disliked
content_scores = (profile @ X.T).astype(np.float32)

# ---------------- 5. Popularity baseline ----------------
pop_scores = np.tile(np.log1p(item_cnt), (n_users, 1)).astype(np.float32)

# ---------------- 6. Blending ----------------
def z(a):  # put every model on the same scale (per user)
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)

zcf, zct = z(cf_scores), z(content_scores)
# Trust in CF grows with data: n/(n+10). Few ratings -> weight near 0 -> content takes over.
user_conf = user_cnt / (user_cnt + CONF_K)
item_conf = item_cnt / (item_cnt + CONF_K)
w = np.outer(user_conf, item_conf).astype(np.float32)   # weight for CF, per (user, movie)
hybrid_scores = w * zcf + (1 - w) * zct
fixed_scores = 0.5 * zcf + 0.5 * zct

models = {"Popularity": pop_scores, "CF only (SVD)": cf_scores, "Content only": content_scores,
          "Hybrid fixed 50/50": fixed_scores, "Hybrid adaptive": hybrid_scores}

# ---------------- 7. Evaluation ----------------
def top_k(scores):
    sc = np.where(seen, -np.inf, scores)       # never recommend already-seen movies
    idx = np.argpartition(-sc, K, axis=1)[:, :K]
    order = np.argsort(-np.take_along_axis(sc, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)

def evaluate(recs, users, only_items=None):
    P, R, N, used = [], [], [], 0
    for u in users:
        rel = test_liked.get(u, set())
        if only_items is not None:
            rel = rel & only_items
        if not rel:
            continue
        hits = [1 if i in rel else 0 for i in recs[u]]
        dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
        idcg = sum(1 / np.log2(r + 2) for r in range(min(len(rel), K)))
        P.append(sum(hits) / K); R.append(sum(hits) / len(rel)); N.append(dcg / idcg); used += 1
    return [np.mean(P), np.mean(R), np.mean(N), used]

warm_list = [u for u in range(n_users) if u not in cold_users]
cold_list = sorted(cold_users)
all_users = list(range(n_users))
slices = {"Warm users": (warm_list, None), "Cold users": (cold_list, None),
          "Cold items (all users)": (all_users, cold_items)}

table, all_recs = [], {}
for name, sc in models.items():
    recs = top_k(sc)
    all_recs[name] = recs
    for sl, (us, items) in slices.items():
        p, r, n, c = evaluate(recs, us, items)
        table.append({"Model": name, "Slice": sl, f"P@{K}": p, f"R@{K}": r, f"NDCG@{K}": n, "Users": c})
res = pd.DataFrame(table)
res.to_csv("results/results.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))

# ---------------- 8. Failure analysis ----------------
recs = all_recs["Hybrid adaptive"]
title = movies["title"].values
gen = movies["genres"].values
lines = []
share = np.mean([i in cold_items for u in cold_users for i in recs[u]])
lines.append(f"Share of recommendations that are cold items (cold users): {share:.2%}")
share_w = np.mean([i in cold_items for u in warm_list for i in recs[u]])
lines.append(f"Share of recommendations that are cold items (warm users): {share_w:.2%}\n")
bad = [u for u in cold_list if test_liked.get(u) and not (set(recs[u]) & test_liked[u])]
lines.append(f"Cold users with zero hits: {len(bad)} of {len(cold_list)}\n")
for u in bad[:5]:
    seen_titles = [f"{title[i]} [{gen[i]}]" for i in np.where(seen[u])[0]]
    lines.append(f"--- Cold user #{u} ---")
    lines.append("Known history : " + "; ".join(seen_titles))
    lines.append("We recommended: " + "; ".join(f"{title[i]} [{gen[i]}]" for i in recs[u][:5]))
    lines.append("User later liked: " + "; ".join(title[i] for i in list(test_liked[u])[:5]) + "\n")
open("results/failure_cases.txt", "w").write("\n".join(lines))
print("\n".join(lines))
