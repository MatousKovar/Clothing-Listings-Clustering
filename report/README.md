# Report

LaTeX source for the 4-page course report.

## Files

- `report.tex` — main paper (IEEEtran two-column conference style)
- `refs.bib` — bibliography (mostly placeholders; **replace `zalando_crossgeo` with the real citation before submitting**)

## Build

```sh
cd report
pdflatex report
bibtex report
pdflatex report
pdflatex report
```

Or with `latexmk`:

```sh
latexmk -pdf -bibtex report.tex
```

## TODOs before submission

- [ ] Replace `zalando_crossgeo` placeholder in `refs.bib` with the actual paper the course expects you to cite. Read it and update §II to reflect what they actually do/report.
- [ ] Fill in the `TBD` cells in Table~1: B2 (exact-title) and B3 (raw CLIP cosine) clustering F1 on the same 30k val slice. Code-wise these are one-cell diffs of `phase1/05_phase2_baseline.ipynb`.
- [ ] Fill in the Phase-1 leaderboard score (the actual number, not a placeholder).
- [ ] Add up to 2 figures if you have something to show — strong candidates: (a) the Siamese-score histogram for true-positive vs candidate edges, (b) a panel of three example clusters (one correct, one over-merge, one split). Both are easy to export from the existing notebooks.
- [ ] Verify page count after compiling; trim §I or §III if it spills past 4 pages.
- [ ] Double-check that the report is under 1 MB (PDF). If figures push it over, downsample images.

## Length budget (4 pages, two-column)

| Section            | Target length     |
|--------------------|-------------------|
| Abstract           | ~150 words        |
| Introduction       | ~½ column         |
| Related Work       | ~½ column         |
| Methodology        | ~1 column + algo  |
| Baselines          | ~⅓ column         |
| Experiments        | ~⅓ column         |
| Results            | ~½ column + table |
| Conclusion         | ~⅓ column         |
| Bibliography       | ~⅓ column         |

If something needs to grow, take it from Methodology or Results — those are
where the real content is. The course rubric explicitly calls out clarity of
contribution and comparison to baselines as the two highest-weight criteria.
