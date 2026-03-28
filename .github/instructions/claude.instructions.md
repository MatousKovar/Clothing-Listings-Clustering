---
applyTo: '**'
---

NEDIVEJ SE DO SLOZKY data/images - je velka, zabije te to


Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

tohle je zadani: 

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

NEDIVEJ SE DO SLOZKY data/images - je velka, zabije te to


potrebuju toto naimplementovat. Uz mam naimplementovany GlamiDatasetVocabulary GlamiItemDataset a GlamiSiamese dataset. O neco jsem se pokousel v explore_datasets.ipynb jsem mel impplementaci ale nefungovalo to velmi dobre, takze mas pripravene datsety a chop se toho ty - idealne pouzij stejne obrazkove embeddingy - clip_embeddings.pt. Tlac na to at se to rychle trenuje, nemusi to byt 99.99 procent accuracy ale chci to pohotove testovat. V task_1.csv je 80% negativnich a 20% pozitivnich - s tim pocitej. 


Takze tvuj ukol je nasledujici - rozdelit data nejak rozumne udelat siamese dataset. Natrenovat model udelat demo outputu - vezmu id groupy vyprintim 5 obrazku 5 popisku a info jestli bych predikoval duplicitu nebo ne. 

NEDIVEJ SE DO SLOZKY data/images - je velka, zabije te to