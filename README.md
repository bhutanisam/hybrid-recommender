# Hybrid Recommendation Engine with Cold-Start Handling

A beginner-level movie recommender that blends **collaborative filtering (SVD)** with a
**content-based model (genre + decade)**. The blend automatically relies more on content
when a user or movie has few ratings (cold start).

> Note: This is my first AI/ML project. I built it with AI assistance and learned the
> concepts while doing it. See `WRITEUP.md` for a plain-language explanation.

## Dataset
[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) (6,040 users, 3,883 movies, ~1M ratings).
Not committed to the repo (see `.gitignore`); download it as below.

## Setup
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data
1. Download `ml-1m.zip` from https://grouplens.org/datasets/movielens/1m/
2. Unzip it so the files are at `data/ml-1m/ratings.dat` and `data/ml-1m/movies.dat`

## Reproduce results / re-run evaluation
```bash
python recommender.py data/ml-1m
```
This trains all models, evaluates them, and writes:
- `results/results.csv` - Precision@10, Recall@10, NDCG@10 for every model and slice
- `results/failure_cases.txt` - failure analysis examples

Settings (top-K, cold-user fraction, blending speed, seed) are constants at the top of
`recommender.py`. The random seed is fixed, so results are reproducible.

## How it works (short)
1. **Split**: each user's ratings sorted by time; first 80% train, last 20% test.
2. **Cold-start slice**: a random 20% of users are made "cold" by keeping only 3 training
   ratings (<5); the rest of their ratings go to test. Cold items = movies with <5 training ratings.
3. **CF**: SVD (30 factors) on the mean-centered rating matrix.
4. **Content**: movie features = genres + release decade; user profile = weighted average of features of rated movies.
5. **Blend**: `weight_for_CF = (n_user/(n_user+10)) * (n_movie/(n_movie+10))`,
   `score = w * CF + (1-w) * content`. Fewer ratings -> more content.
6. **Metrics**: Precision@10, Recall@10, NDCG@10 (a movie is "relevant" if the user rated it >= 4).
   Reported separately for warm users, cold users, and cold items.

## Files
- `recommender.py` - full pipeline
- `WRITEUP.md` - approach, decisions, results, failure analysis
- `results/` - generated outputs
