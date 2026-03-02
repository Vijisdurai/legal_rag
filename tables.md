### 95% Bootstrap Confidence Intervals (Recall@5, MRR, nDCG@5)

| System | Recall@5 (95% CI) | MRR (95% CI) | nDCG@5 (95% CI) |
|--------|-------------------|--------------|-----------------|
| BM25-Only | 0.6591 [0.5756-0.7424] | 0.4708 [0.3989-0.5447] | 0.5180 [0.4462-0.5936] |
| TF-IDF | 0.6288 [0.5455-0.7121] | 0.4414 [0.3707-0.5179] | 0.4881 [0.4194-0.5632] |
| Vector-Only | 0.7727 [0.7045-0.8409] | 0.5453 [0.4783-0.6167] | 0.6025 [0.5392-0.6708] |
| Hybrid (no MMR) | 0.5985 [0.5152-0.6818] | 0.4282 [0.3573-0.5052] | 0.4707 [0.3965-0.5453] |
| Hybrid+MMR | 0.5530 [0.4621-0.6439] | 0.4049 [0.3323-0.4835] | 0.4414 [0.3665-0.5198] |

### Wilcoxon Signed-Rank Tests + Cliff's Delta

| System A | System B | Metric | W | p-value | Cliff's delta | Magnitude | Sig? |
|----------|----------|--------|---|---------|---------------|-----------|------|
| Vector-Only | BM25-Only | Recall@5 | 33.0 | 0.0011 | +0.114 | negligible | Y p<0.05 |
| Vector-Only | Hybrid+MMR | Recall@5 | 34.0 | 4.46e-07 | +0.220 | small | Y p<0.05 |
| Vector-Only | Hybrid (no MMR) | Recall@5 | 64.0 | 3.61e-05 | +0.174 | small | Y p<0.05 |
| BM25-Only | Hybrid+MMR | Recall@5 | 19.0 | 9.67e-04 | +0.106 | negligible | Y p<0.05 |
| Hybrid (no MMR) | Hybrid+MMR | Recall@5 | 0.0 | 0.0143 | +0.045 | negligible | Y p<0.05 |
| Vector-Only | BM25-Only | MRR | 571.0 | 0.0175 | +0.108 | negligible | Y p<0.05 |
| Vector-Only | Hybrid+MMR | MRR | 538.5 | 1.67e-04 | +0.208 | small | Y p<0.05 |
