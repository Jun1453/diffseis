# Consolidated Revision Plan

**Manuscript:** OBS wide-angle denoising with a conditional DDPM trained on diversity-stacked field data  
**Decision:** Major revision (all three reviewers; Associate Editor concurs)  
**Manuscript sections (for navigation):** Introduction → Data & Methods → Results → Discussion → Conclusions; supplementary examples in `supplementary.tex`

---

## Executive summary

Reviewers and the Associate Editor agree the topic is timely and the use of real OBS field data is a strength, but the manuscript overstates generalization and downstream utility (FWI/RTM) relative to the evidence presented. The central scientific issue is that **diversity-stacked sections are a surrogate target**, not independent ground truth—and for some pass counts the target may include the input pass—so performance metrics mainly show agreement with stacking rather than recovery of an unknown clean Earth response. Revisions should (1) clarify or fix the training target construction (ideally leave-one-pass-out), (2) soften claims and/or add more independent validation, (3) add meaningful baselines and reframing of the contribution, (4) address amplitude/waveform fidelity for waveform-based applications, (5) improve reproducibility of the DDPM setup, and (6) polish presentation (language, figures, captions, abbreviations). Addressing target independence, claims, baselines, and evaluation framing is essential before resubmission; language and figure polish can proceed in parallel but should not be deferred entirely.

---

## Major revisions (must address)

### A. Training target, leakage, and what the model actually learns


| #   | Action                                                                                                                                                                                                                                                    | Where (manuscript)                                         | Reviewers                                  |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------ |
| A1  | **State explicitly** whether each input shooting pass is excluded from the diversity stack used as its training target. If the 3-pass stack can include the denoised pass, acknowledge the leakage risk.                                                  | Data & Methods (stacking / training pairs, ~lines 290–294) | **R1** (major #1), **R2** (#1)             |
| A2  | **Strongly recommended:** Rebuild targets with **leave-one-pass-out (LOPO)** stacking and re-train or re-evaluate key results (validation metrics, representative figures). Report whether performance changes materially.                                | Methods + Results + Discussion                             | **R1** (major #1)                          |
| A3  | **Discuss bias/artifact transfer** from diversity stacking into the learned mapping (weights, failed stacks, residual stack noise).                                                                                                                       | Methods (stacking) + Discussion (limitations)              | **R2** (#1), **R1** (major #4)             |
| A4  | **Reframe contribution** explicitly: the model learns to **emulate or improve upon diversity stacking from a single pass**, unless external validation is added. Avoid language implying recovery of true subsurface response without independent checks. | Introduction, Discussion, Conclusions                      | **R1** (major #4), **R2** (#4), **Editor** |
| A5  | Clarify how **64 stacked training sections** relate to **5 shooting passes** and **82,880 tiles** (stations × passes × tiling).                                                                                                                           | Data & Methods                                             | **R1** (minor)                             |


### B. Validation design and generalization claims


| #   | Action                                                                                                                                                                                                                                                                                        | Where (manuscript)                                      | Reviewers                                               |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| B1  | **Soften** claims that the model “generalizes to different noise types and geological settings” unless supported by quantitative out-of-domain tests. Align Abstract, Discussion, and Conclusions with evidence.                                                                              | Abstract, Discussion (~368–370), Conclusions            | **R1** (major #2), **R2** (#2), **R3** (#2), **Editor** |
| B2  | Acknowledge that held-out validation stations are from the **same survey line and campaign** as training; describe this as in-domain validation, not fully independent generalization.                                                                                                        | Data & Methods (validation, ~302–324), Discussion       | **R1** (major #2), **R2** (#2), **R3** (#2)             |
| B3  | For **50-m Noto** and **NW Pacific** tests: explain **source-location mismatch** between input and reference and how that limits interpretation; do not over-interpret qualitative panels alone.                                                                                              | Results (test setup, ~~337–338), Discussion (~~377–378) | **R1** (major #2), **R2** (#3)                          |
| B4  | **Add stronger out-of-domain evidence** where feasible: e.g., quantitative metrics on test data with careful alignment, **manual/expert picking comparison**, repeat shots not in stacks, or a downstream tomography/FWI demonstration. If not possible, state as limitation and future work. | Results + Discussion (+ supplement)                     | **R1** (major #2, #4), **R3** (#1, #2)                  |
| B5  | Discuss **limits of generalization**: OBS spacing, shot spacing, dominant frequency, bathymetry, geology, acquisition parameters.                                                                                                                                                             | Discussion                                              | **R3** (#2)                                             |


### C. Baselines and method justification


| #   | Action                                                                                                                                                                                                                                                                | Where (manuscript)   | Reviewers                      |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ------------------------------ |
| C1  | Add comparisons beyond **raw prestack input** and **diversity stack target**, including where applicable: **deterministic supervised baseline** (U-Net or residual CNN on same input–target pairs), and **ablation** (diffusion vs simpler image-to-image regressor). | Results + Discussion | **R1** (major #3), **R3** (#4) |
| C2  | Justify **DDPM complexity** (~155M parameters, 2000 inference steps) versus baselines: quality, compute, and practical trade-offs.                                                                                                                                    | Methods + Discussion | **R1** (major #3), **R3** (#6) |

#### C1 — Code implementation (`baseline` branch)

| Method | Spatial context | Iterative? | Train | Inference | Output path |
| --- | --- | --- | --- | --- | --- |
| DeepDenoiser (pretrained / fine-tuned) | Trace-wise 1D (no neighbor traces) | No | `baseline/rebuild_deepdenoiser.py --model finetuned` (+ `export_finetune_npz` / `run_finetune`) | `Profiles.deepdenoiser_tracewise` → `baseline/deepdenoiser_bridge.py` | `results/baseline/deepdenoiser/{pretrained,finetuned}-*/` |
| Deterministic U-Net | 2D unit window (same tiling as DDPM) | No (single forward) | `train_direct.py` | `baseline/rebuild_unet_direct.py` | `results/baseline/unet-direct/` |
| DDPM (reference) | 2D unit window + diffusion sampling | Yes (~2000 steps) | `train.py` | `rebuild_oop.py` | `results/demultiple*/` |

Metrics script: `python baseline/evaluate.py --scan_defaults` (RMS vs diversity-stack GT in `rebuild.{inp,out,gt}`).


### D. Amplitude, waveform fidelity, and geophysical impact


| #   | Action                                                                                                                                                                                                                    | Where (manuscript)                           | Reviewers                                 |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ----------------------------------------- |
| D1  | Reconcile **trace-wise normalization** (first 0.2 s) and reported **amplitude mismatch** with claims about **FWI/RTM** suitability. State clearly what is preserved (phase, relative amplitude, spectra) and what is not. | Methods (~~290), Results (~~340), Discussion | **R1** (amplitude paragraph), **R3** (#3) |
| D2  | Expand **amplitude/spectral analysis** across offsets and noise regimes (beyond example traces in Fig. 3); tie to waveform-based inversion requirements.                                                                  | Results + Discussion                         | **R1**, **R3** (#3)                       |
| D3  | **Geophysical impact:** Explain whether picking/CC improvements translate to **tomography, initial models, FWI, or RTM**—or provide a downstream example or careful argument why metrics proxy that benefit.              | Discussion (+ Conclusions)                   | **R3** (#1), **R1** (major #4)            |
| D4  | **Qualify** the statement that direct waves are “not useful” for tomography/FWI (context-dependent; may matter in some workflows).                                                                                        | Discussion (~366)                            | **R1** (minor)                            |


### E. Evaluation metrics and failure modes


| #   | Action                                                                                                                                                                                                                                                                           | Where (manuscript)                    | Reviewers                                  |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------ |
| E1  | Clarify that **AIC detectability** and **CC alignment** are computed **against the stacked target**, so metrics measure agreement with stacking—not independent truth.                                                                                                           | Discussion (~361–383), Fig. 7 caption | **R1** (major #4), **R3** (#1)             |
| E2  | Report **main quantitative metrics in the Abstract** (e.g., AIC/CC RMS and misfit ratios on validation set).                                                                                                                                                                     | Abstract                              | **R2** (#6)                                |
| E3  | Add explicit discussion of **residual noise and failure cases** (long-duration noise, strong interference, large amplitude contamination, wide preceding-shot contamination, normalization failures). Classify success vs partial success with examples (Figs. 4–6, supplement). | Discussion + figure callouts          | **R3** (#5), **R1** (implicit in examples) |


### F. Reproducibility: model, training, and inference


| #   | Action                                                                                                                                                                                                                                    | Where (manuscript)               | Reviewers         |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ----------------- |
| F1  | Fix/clarify **DDPM exposition**: conditioning on noisy input vs target; whether U-Net predicts noise given **x_t** (not incorrectly “given x0” only); how **x_t**, conditioning pass, and stack target interact in training and sampling. | Data & Methods (~296–300)        | **R1** (major #6) |
| F2  | Add architecture detail: **conditioning mechanism**, down/up blocks, channels, normalization/attention if any, **inference sampler** schedule, **tile overlap blending**.                                                                 | Methods (+ supplement if needed) | **R1** (major #6) |
| F3  | **Quantify** stochastic variability between two denoising runs (currently “acceptably small” without numbers).                                                                                                                            | Results (~328), supplement       | **R1** (major #6) |
| F4  | Acknowledge **effective sample size** given **80% tile overlap** and discuss overfitting risk despite train/val loss similarity.                                                                                                          | Methods + Discussion             | **R1** (minor)    |
| F5  | Ensure **Open Research / Zenodo** entries list all inputs, validation data, trained weights, notebooks, and scripts needed to reproduce figures and metrics per AGU guidelines.                                                           | Open Research section            | **R3** (#7)       |


### G. Computational practicality


| #   | Action                                                                                                                                   | Where (manuscript)    | Reviewers                               |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | --------------------------------------- |
| G1  | Expand discussion of **training time, inference time per section/line**, GPU requirements, and feasibility for **large marine surveys**. | Discussion (~386–387) | **R3** (#6)                             |
| G2  | Discuss **faster diffusion sampling** (fewer steps, recent schedulers) as future or optional benchmark.                                  | Discussion            | **R3** (#6), **R1** (implicit via cost) |


---

## Minor revisions / clarifications


| #   | Action                                                                                                                                                                                                                                                                                                              | Where               | Reviewers                           |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ----------------------------------- |
| M1  | **English/language edit** throughout (grammar, typos). Fix known errors: *seperately* → separately (Data Availability); *artifical* → artificial (Discussion); *desipte* → despite; *diffence* → difference; *noralized* → normalized; *forth* → fourth (Results); “various process occurring” and similar wording. | Full manuscript     | **R1** (minor), **R3** (minor 1, 6) |
| M2  | **Define abbreviations at first use:** OBS, DDPM, FWI, RTM, AIC, CC, etc.                                                                                                                                                                                                                                           | Introduction onward | **R3** (minor 3)                    |
| M3  | **Terminology consistency:** prestack / pre-stacked, denoised output, ground truth vs target/reference.                                                                                                                                                                                                             | Full manuscript     | **R3** (minor 5)                    |
| M4  | **Figure 1 (workflow):** Remove distracting **background colors**; clarify paths for prestack input, diversity-stacked target, and DDPM output.                                                                                                                                                                     | Fig. 1 + caption    | **R2** (#5), **R3** (minor 4)       |
| M5  | **Expand and clean figure captions** (Figs. 4–6 and others); ensure in-text references match panels and datasets (validation vs 50-m test vs NW Pacific).                                                                                                                                                           | Results figures     | **R1** (minor), **R3** (minor 2)    |
| M6  | **Data Availability Statement:** fix spelling; ensure complete results linkage is clear.                                                                                                                                                                                                                            | Back matter         | **R1** (minor)                      |


---

## Optional / nice-to-have


| #   | Action                                                                                                                                                        | Where                     | Reviewers                            |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------ |
| O1  | Add a **compact summary table** of denoising/metric outcomes across all stations (if Zenodo holds full outputs), reducing reliance on selected examples only. | Results or supplement     | **R1** (minor)                       |
| O2  | Highlight cases where denoising **outperforms** diversity stacking (already mentioned briefly)—with clear caveats about target definition.                    | Discussion                | **R1**, **R2** (#4)                  |
| O3  | Broader literature additions if needed after reframing (recent deep-learning denoising for marine/OBS data).                                                  | Introduction / References | **R1** (referencing), **R2**, **R3** |


---

## Suggested order of work (phased checklist)

### Phase 1 — Scientific design (blocking)

- **A1–A2:** Document pass-inclusion in stacks; implement **LOPO targets** and re-run core evaluation.
- **A4–A5:** Rewrite framing and training-data accounting (sections vs passes vs tiles).
- **B1–B3:** Audit and soften generalization language; fix test/reference mismatch exposition.
- **E1:** Reframe metrics as stack-relative; update Discussion and Conclusions accordingly.

### Phase 2 — New analyses (high impact)

- **C1–C2:** Implement and report **baselines** + DDPM cost/ benefit comparison.
- **B4:** Add best feasible **independent validation** (picking study, downstream test, or explicit limitation).
- **D1–D3:** Amplitude/spectrum fidelity analysis; cautious FWI/RTM claims; geophysical impact paragraph.
- **E3:** Systematic **failure-case** discussion with figure references.

### Phase 3 — Methods transparency

- **F1–F3:** Correct DDPM/math description; architecture table/diagram; stochastic variability numbers.
- **F4:** Overlap / effective sample size note.
- **F5, G1–G2:** Reproducibility package check; compute practicality and faster-sampling outlook.

### Phase 4 — Presentation polish

- **E2, M1–M6:** Abstract metrics; language edit; abbreviations; Fig. 1 and captions; terminology.
- **D4:** Qualify direct-wave statement.
- **O1 (optional):** Summary table across stations.

### Phase 5 — Resubmission package

- Point-by-point **response letter** mapping each comment to Phase 1–4 changes.
- Verify supplement and Zenodo match revised Methods and figures.
- Final pass: Abstract, Key Points, and Conclusions aligned with softened claims and new baselines.

---

## Quick reference: themes × reviewers


| Theme                                   | R1  | R2  | R3  | Editor          |
| --------------------------------------- | --- | --- | --- | --------------- |
| Surrogate target / LOPO / stacking bias | ✓   | ✓   | —   | ✓ (via reviews) |
| Generalization & validation scope       | ✓   | ✓   | ✓   | ✓               |
| Baselines & DDPM justification          | ✓   | —   | ✓   | —               |
| Amplitude / FWI suitability             | ✓   | —   | ✓   | —               |
| Metrics vs stack target                 | ✓   | ✓   | ✓   | —               |
| Reproducibility (DDPM detail)           | ✓   | —   | ✓   | —               |
| Geophysical downstream impact           | —   | —   | ✓   | —               |
| Failure cases & compute cost            | —   | —   | ✓   | —               |
| Data/software availability              | —   | —   | ✓   | —               |
| Language, figures, abstract metrics     | ✓   | ✓   | ✓   | ✓ (quality)     |


---

*This plan synthesizes only comments recorded in `reviewers_comments.md` (Associate Editor summary + Reviewers 1–3). Line references in R1’s review correspond to the version cited in that file; verify against the current `agujournaltemplate.tex` after edits.*