# Phase 2 — Product Grouping

## Assignment
- Input: `items_phase_2.csv` (~200k items, no labels)
- Output: CSV — one line per group, comma-separated `item_id`s, max 100/line, singletons on own line
- Metric: **Pairwise F1** (TP/FP/FN on item-pair edges)
- No confidence score needed

Phase 2 – Product Grouping

Task Description

Using all items from items_phase_2.csv, your task is to solve an end-to-end entity matching problem. You must construct disjoint product groups, where each group represents a single, distinct underlying product.

Submission Format

Your solution must be handed over as a simple text/CSV file where each line represents one product group.

The line must contain comma-separated `item_id`s.
An item can only appear in one group (hard clustering).
Constraint: There cannot be more than 100 items in a single group (i.e., on one line). If a line exceeds 100 items, the submission will be rejected.
Example Submission File:

```
item_1,item_2,item_5
item_3
item_4,item_7
item_6,item_8,item_9,item_10
```
(Note: Singletons—items that do not match with anything else—should be placed on their own line).

Evaluation Metric: Pairwise F1 Score

To evaluate the quality of your product groups, we will use the F1 score of pairwise comparisons between your groups and the ground truth.

To solve this, we evaluate your clusters by treating them as a full adjacency matrix of edges between items, and we compute the F1 score of these pairwise links.

How it works:

Imagine a matrix where every item is compared to every other item. An edge (link) exists between two items if they are placed in the same group.

We define the components as follows: * True Positives (TP): A pair of items are in the same group in the Ground Truth AND in your Prediction. * False Positives (FP): A pair of items are in different groups in the Ground Truth, BUT you put them in the same group (Over-merging). * False Negatives (FN): A pair of items are in the same group in the Ground Truth, BUT you put them in different groups (Split clusters).

Precision = TP / (TP + FP) Recall = TP / (TP + FN) F1 = 2 * Precision * Recall / (Precision + Recall)

Visualization & Toy Example

Let’s say we have 5 items in the dataset: A, B, C, D, E

1. Ground Truth (Real Products) The true identical products are: * Group 1: A, B, C * Group 2: D, E

This creates 4 True Edges in our adjacency matrix: (A-B), (A-C), (B-C), (D-E)

2. Your Prediction Your model outputs the following groups: * Predicted Group 1: A, B * Predicted Group 2: C, D, E

This creates 4 Predicted Edges: (A-B), (C-D), (C-E), (D-E)

3. Metric Calculation

Let’s cross-reference your predicted edges against the true edges:

Edge Pair	Status	Reason
(A-B)	TP	Correctly grouped together
(D-E)	TP	Correctly grouped together
(C-D)	FP	You linked them, but they are different products
(C-E)	FP	You linked them, but they are different products
(A-C)	FN	You missed this true link (split group)
(B-C)	FN	You missed this true link (split group)
Final Score: * TP: 2 * FP: 2 * FN: 2

Precision: 2 / (2 + 2) = 0.50
Recall: 2 / (2 + 2) = 0.50
Pairwise F1: 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.50
Submission

The submission process will be split into two parts. In the first part, you will submit your predictions from the items_phase_1.csv and you will get a full score on this part. In the second part, you will submit your predictions from the items_phase_2.csv where you will not be able to see the score that you achieved with your submission. This part will be used to sort the final leaderboard.

## Assets
- `clip_embeddings.pt` (768d), `text_embeddings.pt` (512d)
- `siamese_best.pth` — trained duplicate classifier
- ~900k train items with labels for validation

Implement a clustering model in new notebook which will use my siamese model. This model will find neareset neighbors for each point and then use the siamese model to find duplicates inside those neighbors.