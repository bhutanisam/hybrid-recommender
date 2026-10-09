# Write-up: Hybrid Recommender with Cold-Start Handling

*(Fill the [BRACKETS] after running the code. Copy numbers from `results/results.csv`.)*

## 1. Problem
New users and new movies have no rating history, so collaborative filtering (CF) has
nothing to learn from. I built a hybrid that uses content (genres, decade) when history is thin.

## 2. Approach
- **CF (SVD):** compresses the user x movie rating table into 30 hidden "taste" factors
  and predicts how much a user would like unseen movies.
- **Content-based:** describes each movie by genres + decade. A user's profile is built from
  the movies they rated (liked movies pull the profile toward their genres). Movies similar
  to the profile score high. Works even for a movie with zero ratings.
- **Blending:** weight on CF = (user confidence) x (movie confidence), where confidence = n/(n+10).
  1 rating -> ~9% trust in CF; 100 ratings -> ~91%. This is not a fixed 50/50 split.
- **Cold-start test:** 20% of users were limited to 3 training ratings; movies with <5
  training ratings are cold items. Metrics are reported separately.

## 3. Key decisions
- Used MovieLens 1M for speed.
- Ranking metrics (Precision/Recall/NDCG @10) instead of RMSE, because recommending is a ranking task.
- Time-based split (train on past, test on future) to avoid "peeking".
- Simple SVD from scipy instead of heavy libraries, to keep it understandable.
- Genre + decade only (no text embeddings) to keep the project basic; see future work.

## 4. Results (K = 10)
Model	Warm P@10	Warm R@10	Warm NDCG	Cold P@10	Cold R@10	Cold NDCG
Popularity	0.0792	0.0471	0.0897	0.3581	0.0566	0.3773
CF only	0.0959	0.0660	0.1140	0.2579	0.0365	0.2756
Content only	0.0148	0.0124	0.0175	0.0507	0.0053	0.0555
Hybrid 50/50	0.0931	0.0656	0.1109	0.2513	0.0347	0.2696
Hybrid adaptive	0.0961	0.0661	0.1139	0.1792	0.0242	0.2055

**Observations (edit to match YOUR numbers):**
- [Did CF drop sharply for cold users? Did hybrid/content hold up better?]
- [Did adaptive hybrid beat fixed 50/50 on cold users while staying close to CF on warm users?]
- [Cold items: all models struggle? Say what the numbers show.]

## 5. Failure analysis
See `results/failure_cases.txt`. Typical reasons the hybrid still fails:
1. **Genre is too coarse.** Two "Drama" movies can be totally different, so content can't tell them apart.
2. **Only 3 ratings = weak taste signal.** Content may lock onto one genre and miss the user's wider taste.
3. **Popularity matters.** Popular movies are liked by many; content-only ignores that, so can recommend obscure movies.
4. **Cold items have no CF signal at all**, so ranking relies only on metadata.
5. [Add what you see in your own failure_cases.txt]

## 6. Limitations & future work
- Add tags / plot-description embeddings (e.g. sentence-transformers) to the content model.
- Tune the blending constant (10) and SVD factors with validation data.
- Try ALS or a neural model; use MovieLens 25M.

## 7. Note on AI assistance
I built this with AI assistance as my first ML project. I reviewed each part and can explain it.


