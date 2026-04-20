Competition 2026

Cross-Geo Product Matching and Grouping Challenge

This year’s competition focuses on identifying identical products across different listings and geographical markets.

Participants must develop models capable of recognizing when multiple items correspond to the same underlying product, even if their:

titles differ,
descriptions vary,
prices differ,
images differ slightly,
metadata differs across geographies.
The dataset consists of real-world product data provided by Glami, a leading fashion search engine operating across multiple European markets.

The challenge consists of two phases:

Phase 1 – Duplicate detection within predefined groups
Phase 2 – Product grouping with confidence estimation
The goal is to combine multimodal signals such as text, image, price, and categorical metadata to perform robust duplicate detection under real-world noise.

Dataset

The competition uses real-world product listing data provided by Glami.

Files Provided

Participants will receive:

items_train.csv (~900k items with labels)
items_phase_1.csv (~200k items without labels)
items_phase_2.csv (~200k items without labels, released in Phase 2)
task_1.csv (15,000 predefined groups of 5 items)
glami_images.tar.gz (images for all items – distributed separately)
One image in the dataset is corrupted. You can download the correct version here: https://static.glami.hu/img/200x240bt/508912023.jpg

Available Features Per Item

Each item contains:

title
description
geo
price
image
department IDs
color tag IDs
label (hidden during competition except in the training set)
Total Statistics

~1.3M total items
217k distinct product labels
Competition Structure

Phase 1 – Duplicate Detection Within Groups

Task Description

You are given 15,000 predefined groups, each containing 5 items.

The groups are generated from items_phase_1.csv. Items may appear in multiple groups.

A positive group contains at least two items that share the same product label.

A negative group contains five different products.

Positive groups may contain:

2 identical items
3 identical items
4 identical items
5 identical items
Your Task

For each group, predict whether it contains at least one duplicate pair.

Let z_g = 1 denote that the group contains at least one pair of items with the same label, and z_g = 0 otherwise.

Your task is to predict z_g for all groups.

Phase 1 Evaluation

We evaluate binary classification performance across all 15,000 groups.

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 * Precision * Recall / (Precision + Recall)

Where:

TP – correctly predicted positive groups
FP – incorrectly predicted positive groups
FN – missed positive groups
Tie-breaking

If multiple participants obtain the same F1 score:

Higher Recall wins
If still tied, Higher Precision wins
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



# Phase 1 - ALREADY FINISHED

## Phase 2

Implement a clustering model in new notebook which will use my siamese model. This model will find neareset neighbors for each point and then use the siamese model to find duplicates inside those neighbors.