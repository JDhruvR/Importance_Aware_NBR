# Qualitative Analysis Workplan
## *Not All Items Are Created Equal — Work Division*

---

## What We Can and Cannot Do Without Item Metadata

Since we do not have item names or category labels, any analysis that requires **interpreting what an item is** must be modified to work with **item IDs and statistical proxies** instead. The table below maps each planned section to its feasibility.

| Section | Original Plan | Without Metadata | Verdict |
|---|---|---|---|
| 2.1 Importance score intuition | Human-readable basket examples | Use item ID + alpha_IDF rank; show score variance patterns | **Modified** |
| 2.2 Alpha_IDF vs learned correlation | Fully numeric | No change needed | **Full** |
| 2.3 Same item, different contexts | Requires recognising "cooking oil" | Find high-frequency items, show score variance across basket types statistically | **Modified** |
| 2.4 IDF correction validation | Requires naming top items | Compare avg frequency rank of top-alpha items before/after IDF | **Modified** |
| 3.1–3.4 Gate analysis | Fully numeric | No change needed | **Full** |
| 4.1–4.3 Orthogonal projection | Fully numeric | No change needed | **Full** |
| 4.2 Which items live in each subspace | Needs item names to interpret | Use avg alpha_IDF of top-K items per subspace as proxy | **Modified** |
| 5.1 Core item precision | Fully numeric | No change needed | **Full** |
| 5.2 Fill coherence | Uses co-occurrence matrix (IDs only) | No change needed | **Full** |
| 5.3 Intra-list diversity | Uses embeddings (IDs only) | No change needed | **Full** |
| 5.4 Slot saturation prevention | Needs item names for examples | Classify by alpha_IDF threshold, show counts numerically | **Modified** |
| 5.5 Soft vs hard bridge | Fully numeric | No change needed | **Full** |
| 6. Repeat vs Explore | Fully numeric | No change needed | **Full** |
| 7. Popularity bias | Fully numeric | No change needed | **Full** |
| 8. Training dynamics | Fully numeric | No change needed | **Full** |
| 9. Ablation study | Fully numeric | No change needed | **Full** |
| 10.1 High-loss user profiles | Fully numeric | No change needed | **Full** |
| 10.2–10.3 Two-stage vs flat examples | Needs item names for showcase | Show item IDs with alpha_IDF scores; still illustrates structure | **Modified** |
| 11.1 Word2Vec cluster quality | Needs category labels to validate clusters | Skip t-SNE labelling; use nearest-neighbour co-occurrence agreement instead | **Modified** |
| 11.2 Embedding drift | Fully numeric | No change needed | **Full** |
| 11.3 Intent vs fill subspace clusters | Needs labels to interpret cluster meaning | Report avg alpha_IDF per cluster region; skip narrative labelling | **Modified** |
| 12. User segmentation | Fully numeric | No change needed | **Full** |
| 13. Dataset characteristics | Fully numeric | No change needed | **Full** |
| 14. Computational analysis | Fully numeric | No change needed | **Full** |

**Dropped entirely:** Nothing needs to be fully dropped. All sections are achievable in modified form or as-is.

---

## Modified Procedures for Metadata-Dependent Analyses

**2.1 (Importance score intuition):** Instead of named baskets, pick 10–15 baskets, display items as `item_<id>`, and show the alpha_IDF and learned alpha score next to each. The key thing to show is that score variance within a basket is high (std > 0.15), not that specific items are interpretable. One or two baskets where a single item has alpha > 0.7 while others are below 0.2 is sufficient.

**2.3 (Context-dependent scores):** Programmatically find items that appear in 50+ baskets with high variance in learned alpha across those baskets. Report: "Item X appears in N baskets; its learned importance ranges from α_min to α_max (std = y), confirming that the MLP assigns context-dependent scores rather than a fixed global value." No name needed.

**2.4 (IDF correction):** Sort items by raw delta (pre-IDF) and by alpha_IDF. Report average document frequency (df) of the top-20 items in each ranking. If the IDF-corrected top-20 has lower avg df than the uncorrected top-20, the correction is working. Fully numeric.

**10.2–10.3 (Showcase examples):** Show baskets as lists of item IDs. Annotate each with its alpha_IDF score. Even without names, a basket where K1=2 high-alpha items are predicted first and K2=8 low-alpha items follow demonstrates slot separation structurally.

**11.1 (Word2Vec clusters):** Replace category-labelled t-SNE with a nearest-neighbour agreement metric: for each item, check what fraction of its K nearest neighbours in embedding space also appear in the same training baskets. High agreement = co-occurrence is reflected in geometry = word2vec initialisation is meaningful.

---

## Work Division

Tasks are ordered **within each person's list** by combined low-effort + high-impact priority. Do them top to bottom.

---

### Team Member A

**Focus: Core model behaviour — importance head, orthogonal projection, training dynamics, ablation**

---

**A1 — Alpha_IDF vs Learned Importance Correlation (§2.2)**
*Effort: Low | Impact: High*

Compute Pearson correlation between alpha_IDF and the learned MLP output at end of Phase 2 and end of Phase 3. Expected: ~0.90 at Phase 2 end, ~0.60 at Phase 3 end. A drop in correlation is the evidence that the MLP is learning context beyond what the global score captures. One scatter plot, one number per phase.

```python
corr_phase2 = pearsonr(alpha_idf[item_ids], learned_scores_phase2)[0]
corr_phase3 = pearsonr(alpha_idf[item_ids], learned_scores_phase3)[0]
```

---

**A2 — Component Loss Curves (§8.1)**
*Effort: Low | Impact: High*

Plot L_intent, L_fill, L_orth, and total loss across training steps from logs. No new computation needed if losses were logged. If L_intent drops faster than L_fill, note that the soft bridge conditioning is the bottleneck. This goes directly in the paper as a training dynamics figure.

---

**A3 — Importance Head Score Variance (§2.1 modified)**
*Effort: Low | Impact: High*

Run `inspect_importance` on the validation set. For each basket, compute intra-basket std of alpha scores. Report mean and distribution of this std across all baskets. If mean std > 0.15, scores are differentiating meaningfully. Pick 10–15 baskets with highest intra-basket variance and display as a table (item_id | alpha_IDF | learned_alpha) — these are your showcase examples. No item names needed.

---

**A4 — Orthogonality Verification (§4.1)**
*Effort: Low | Impact: High*

Measure cosine similarity between h_intent and h_fill across the validation set. Plot per epoch alongside L_orth. Report mean cosine similarity at end of training. Should be near 0. If it stays high despite L_orth decreasing, flag it explicitly as the failure mode described in §4.1.

```python
cos_sim = F.cosine_similarity(intent_repr, fill_repr, dim=-1).mean()
```

---

**A5 — Context-Dependent Importance (§2.3 modified)**
*Effort: Low | Impact: Medium*

Find all items with df > 50 (appear in many baskets). Compute the std of their learned alpha across all basket appearances. Sort by std descending. The top 10 items by std are your "same item, different contexts" examples. For each, show a few basket appearances with their learned alpha. Report the range and std. This is the numeric substitute for the cooking-oil example in the original guide.

---

**A6 — Projection Stability (§4.3)**
*Effort: Low | Impact: Medium*

At each Gram-Schmidt re-orthonormalisation step, record `||P^T P - I||`. Plot over training. Should spike before each step and reset after. This is quick to add as a logged metric if not already tracked. Shows the re-orthonormalisation is necessary, not cosmetic.

---

**A7 — Component Ablation Table (§9.1)**
*Effort: High | Impact: High*

Run all ablation variants and fill in the table: GP-TopFreq baseline, flat BERT+GPT, + importance head, + alpha_IDF loss weighting, + two-stage decoding, + orthogonal projection, full model. Columns: Recall@10, NDCG@10, Repeat R@10, Explore R@10, Core Precision. This is the most important table in the paper and justifies every architectural component individually.

---

**A8 — Intent Dimension Sensitivity (§4.4)**
*Effort: High | Impact: Medium*

Sweep intent_dim ∈ {8, 16, 32, 64}. Report Recall@10, core precision, and orthogonality at each value. A clear sweet spot is a meaningful architectural finding and justifies the chosen value.

---

### Team Member B

**Focus: Decoding behaviour, repeat/explore, user analysis, dataset characterisation, gate analysis**

---

**B1 — Repeat vs Explore Breakdown (§6)**
*Effort: Low | Impact: High*

This is the single most important evaluation in NBR per Li et al. (2023). Run `repeat_explore_metrics` on the validation set for your full model and GP-TopFreq. Report Repeat Recall@10 and Explore Recall@10 side by side. The expected story: your model has higher explore recall than GP-TopFreq due to the IDF correction, even if overall recall is similar. This reframes any metric gap as a deliberate design trade-off, not a failure.

---

**B2 — Slot Saturation Prevention (§5.4 modified)**
*Effort: Low | Impact: High*

On 100 validation baskets, run flat top-10 and two-stage top-10. For each predicted item, check whether alpha_IDF > tau_alpha (core) or below (fill). Report average counts: how many core-class items appear in flat top-10 vs two-stage top-10. Expected: flat top-10 has 7–8 core-class items; two-stage has K1=2 by construction. This is the clearest structural argument for two-stage decoding and requires no item names.

---

**B3 — Dataset Characteristics (§13)**
*Effort: Low | Impact: Medium*

Compute and report: n_users, n_items, n_baskets, avg_basket_size, avg_user_history_length, repeat_rate, sparsity. Also compute alpha_IDF distribution (§13.2): plot histogram of alpha_IDF scores across all items. Expected: right-skewed. This contextualises all other results and is required boilerplate for any systems paper.

---

**B4 — Gate Trajectory Over Training (§3.1)**
*Effort: Low | Impact: Medium*

Plot mean gate value per epoch from training logs. Expected: starts near 1.0 (CLS dominant), decreases toward 0.4–0.5 as importance head matures. If this trajectory is observed it directly validates the gating mechanism behaves as theorised. One plot, very quick to produce from logged gate values.

---

**B5 — Core Item Precision (§5.1)**
*Effort: Low | Impact: Medium*

Compute core precision at inference: `|predicted_core ∩ ground_truth| / K1`. Compare against flat top-K1 baseline. If two-stage core precision > flat top-K1 precision, this is a direct win. Report as a single number with comparison baseline.

---

**B6 — Intra-List Diversity (§5.3)**
*Effort: Low | Impact: Medium*

Using the `intra_list_diversity` function from §15, measure pairwise cosine distance between predicted items for: flat top-K, stage-1 only (residual in intent subspace), full two-stage. Expected: diversity increases at each stage. This validates that the residual loop is suppressing redundant picks.

---

**B7 — Performance by History Length (§10.4)**
*Effort: Low | Impact: Medium*

Stratify validation users by number of baskets in history: 1–3, 4–7, 8–15, 15+. Report Recall@10 for each bin. If performance improves with history length, the GPT is doing real sequential modelling. If it's flat, the model is ignoring history — honest and important to know either way.

---

**B8 — Gate vs Basket Size Correlation (§3.3)**
*Effort: Low | Impact: Low-Medium*

Run `gate_basket_correlation` on the validation set. Report Pearson r between gate value and basket size. Expected: negative correlation (larger baskets → more intent structure → lower gate → trust importance stream more). Any significant correlation justifies adaptive gating over a fixed weighted sum.

---

**B9 — User Segmentation: Planner vs Opportunist (§12.1)**
*Effort: Medium | Impact: Medium*

Segment users by std of their per-basket alpha_IDF scores across history. Low std = planner (consistent basket structure). High std = opportunist (variable). Report Recall@10 separately for each segment. Expected: model performs better on planners. Pairs well with B7 to form a full user analysis subsection.

---

**B10 — Fill Coherence (§5.2)**
*Effort: Medium | Impact: Medium*

Build a co-occurrence matrix from training baskets. For each validation prediction, compute average co-occurrence between predicted core items and predicted fill items. Compare against the same metric for a flat top-K decoder. Higher coherence for two-stage means the fill conditioning is working.

---

## Summary Table

| ID | Task | Owner | Effort | Impact |
|---|---|---|---|---|
| A1 | Alpha_IDF vs learned correlation | A | Low | High |
| A2 | Component loss curves | A | Low | High |
| A3 | Importance score variance / showcase | A | Low | High |
| A4 | Orthogonality verification | A | Low | High |
| A5 | Context-dependent importance (modified 2.3) | A | Low | Medium |
| A6 | Projection stability | A | Low | Medium |
| A7 | Component ablation table | A | High | High |
| A8 | Intent dimension sensitivity | A | High | Medium |
| B1 | Repeat vs Explore breakdown | B | Low | High |
| B2 | Slot saturation prevention | B | Low | High |
| B3 | Dataset characteristics + alpha_IDF distribution | B | Low | Medium |
| B4 | Gate trajectory over training | B | Low | Medium |
| B5 | Core item precision | B | Low | Medium |
| B6 | Intra-list diversity | B | Low | Medium |
| B7 | Performance by history length | B | Low | Medium |
| B8 | Gate vs basket size correlation | B | Low | Low-Med |
| B9 | User segmentation (planner vs opportunist) | B | Medium | Medium |
| B10 | Fill coherence via co-occurrence | B | Medium | Medium |

**If time runs out**, A7 (ablation) and B1 (repeat/explore) are the two analyses that cannot be skipped. Everything else is supporting evidence. A1 and B2 are the next priority tier.
