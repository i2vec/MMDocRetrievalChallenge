# MMDocRetrievalChallenge

## Best Solution
Solve the two tasks with serveral models.
We use GME for the m2kr dataset and use colqwen for the mmdocir datset. In the end, we use qwen2.5-vl-72B-AWQ for the rerank.
we got 65.5 on the leaderboard(With Rerank).
we got 64.0 on the leaderboard(Without Rerank)


### M2KR
1. GME image-to-text retrieval
2. GME Layout-level image-to-image retrieval
3. Multi-route recall merging
4. QwenVL as result reranker

### MMDocIR
1. Colqwen2-7B text-to-image retrieval
2. GME layout-level text-to-image retrieval
3. GME text-to-text retrieval
4. Rerank the colqwen results
5. Multi-route recall merging 

## Unified Model Solution
Solve the two tasks with a unified model.
We use GME as the unified model.
we get 61.7 on the leaderboard.

### M2KR
1. GME image-to-text retrieval
2. GME Layout-level image-to-image retrieval
3. Multi-route recall merging

### MMDocIR
1. GME layout-level text-to-image retrieval
2. GME text-to-text retrieval
3. Multi-route recall merging
