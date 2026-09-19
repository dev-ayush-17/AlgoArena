# Decisions Log

All key architectural and design decisions are recorded here with their rationale so that no decision gets re-litigated in future sessions. If you're an agent reading this file, treat every recorded decision as final unless Rahul explicitly says otherwise.

## Format

Each entry follows this structure:
- **Question:** The design question that was raised.
- **Decision:** What was decided.
- **Rationale:** Why this was chosen over alternatives.
- **Date:** When the decision was made.
- **PRD Reference:** Which section of the PRD this relates to.

---

### OQ1: Dataset Selection

- **Question:** Which dataset should Algorithm Arena use?
- **Decision:** **Online Shoppers Purchasing Intention** (UCI ML Repository, ID 468).
- **Rationale:** Goldilocks size (12,330 rows) — all 5 models train in seconds with no subsampling. Feature properties naturally expose model trade-offs: correlated features (BounceRates/ExitRates) penalize Naive Bayes, scaling disparities penalize un-scaled KNN/SVM while trees are immune, zero-inflated features reward non-linear models. E-commerce CRO domain is compelling for LinkedIn. Class imbalance (84.5/15.5) makes precision/recall trade-offs genuinely visible.
- **Date:** 2026-09-19
- **PRD Reference:** Section 4
- **Source:** [UCI Dataset 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) — CC BY 4.0 license.

_See PRD.md Section 14 (Open Questions) for items pending resolution. The following questions are awaiting Rahul's decision:_

- **OQ1:** ~~Dataset selection~~ ✅ RESOLVED — Online Shoppers Purchasing Intention (UCI ID 468)
- **OQ2:** Encoding strategy (OneHotEncoder for all vs. model-specific encoders)
- **OQ3:** Frontend technology (Vanilla HTML/CSS/JS vs. React/Vue/Svelte)
- **OQ4:** Subsampling strategy for large datasets
- **OQ5:** Naive Bayes variant (GaussianNB vs. BernoulliNB vs. CategoricalNB)
- **OQ6:** Extra features handling (ignore silently vs. reject with 400)
- **OQ7:** Float-to-int rounding for categorical fields
- **OQ8:** Artifact storage (git vs. Git LFS vs. cloud storage)
- **OQ9:** Deployment platform (Render vs. Railway vs. Fly.io)
- **OQ10:** LinkedIn format (post vs. article)
