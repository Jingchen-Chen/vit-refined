# `vit-refined` — Project Analysis & Next-Step Plan

**Date:** 2026‑04‑25
**Author of analysis:** Claude (for Jingchen Chen 陈京晨, X‑Talent Program, advisor 杨涛副教授)
**Project topic, as I now understand it:** Improving Vision-Transformer–based **multi-organ medical image segmentation**, building on the undergraduate thesis (HiFormer + Triplet Attention + PPA on the Synapse dataset) and extending it into the X‑Talent master's project (target completion June 2027).

---

## 0. Important context before reading

* **The domain is medical image segmentation, not generic image classification.** I had to ask once and you initially answered "image classification"; the thesis abstract you later pasted is actually multi-organ CT segmentation. The whole analysis below is rewritten on that basis. If you intend to *pivot* away from segmentation, ignore Sections 2–4 and we'll redo the plan.
* **No code in the repo yet.** The repository today contains only the proposal docx/pdf, the November 2025 progress report, the slides, and the undergrad thesis PDF. There is no `src/`, no checkpoint, no requirements file, no training script. After ~10 months of an 18‑month plan that is the single largest risk, ahead of any architectural concern.
* **The local sandbox could not parse the binary files this session**, so I worked from the abstract / intro / methods / results / proposal text you pasted directly. The numerical claims used below (Synapse DSC=79.73, HD=20.25 for HF+TA+PPA, etc.) are taken from your Section 5 results table verbatim.

---

## 1. Snapshot of the work as it stands

**Undergraduate thesis (深圳技术大学, 基于 ViT 的医学影像分割):**

* **Backbone:** HiFormer (Heidari et al., WACV 2023) — a CNN encoder + Swin Transformer + Dual-Layer Fusion (DLF) hybrid for medical segmentation.
* **Two add-ons you contributed:**
  * **PPA — Parallelized Patch-aware Attention**, lifted from HCF-Net (ICME 2024), which was originally proposed for **infrared small object detection**.
  * **TA — Triplet Attention** (Misra et al., WACV 2021), a cross-dimension attention module.
* **Dataset:** Synapse multi-organ CT (30 cases, 8 organs). 224×224 inputs.
* **Headline results (your Table 5.1):**
  * HiFormer baseline: 78.52 DSC / 18.56 HD95 / 25.51M params.
  * HF+PPA: 79.41 / 20.07 / 25.51M.
  * HF+TA: 79.64 / **14.74** / 25.51M.
  * HF+TA+PPA: **79.73** / 20.25 / 25.51M.

**X-Talent proposal (June 2025 → June 2027):** "ViT-improved medical image segmentation" with **four** proposed innovations stacked into one project:

1. **Hybrid-MambaFormer** — replace part of the Transformer attention with Mamba state-space blocks.
2. **KAN-augmented attention** — bring Kolmogorov–Arnold Networks into the attention module for non-linear function approximation.
3. **Structure-aware Prompt Adapter** — for cross-modal / cross-task transfer.
4. **Evidential Deep Learning + visualization** — for clinical-grade uncertainty quantification.

Plus datasets: Synapse, ACDC, BraTS, CHAOS. Target venues: *Medical Image Analysis*, *Computer Methods and Programs in Biomedicine*, *Computers in Biology and Medicine*.

---

## 2. Does the existing thesis "work"? Honest answer.

### Where it works

* **The model trains and produces sensible numbers.** HF+TA+PPA at 79.73 DSC on Synapse is in the ballpark of TransUNet (77.48), Swin-UNet (79.13), and the reported HiFormer baseline (78.52). It is *internally* a coherent improvement over your own baseline.
* **Parameter efficiency is genuinely good.** 25.51M is small for a hybrid CNN+Swin model. The thesis frames this correctly.
* **The empirical pipeline (Synapse, DSC + HD95, ablation per module) is methodologically sound** and repeatable, which is a real asset for the master's project.
* **Triplet Attention is the strongest of the two additions.** HF+TA improves DSC by +1.12 over baseline AND drops HD95 from 18.56 to 14.74 — a clean Pareto win. TA cost is negligible (the table shows the same 25.51M parameters because it's almost free).

### Where it does *not* fully work, and where reviewers will push back

These are not gentle critiques; they are the same things a Q1/Q2 journal reviewer will write. Better to face them now than at submission.

1. **Your numbers are below current SOTA on Synapse.** 79.73 DSC is a respectable thesis result but it is roughly 4–5 DSC points behind 2024–2025 published numbers on the same benchmark. Recent reports include:
   * **EMCAH-Net** (April 2025): 84.73 DSC on Synapse, +2.85 over its previous SOTA, with ~25% of TransUNet's params.
   * **DATTNet** (Oct 2024): 84.5 DSC on Synapse, +1.7 over TransUNet.
   * **EM-Net**: 83.95 DSC on Synapse, beating U-Mamba.
   * Several Mamba-based medical segmentation papers (VM-UNet, U-Mamba, SegMamba-V2, SCM-UNet) now sit at or near 84+ DSC.

   **Implication for your master's project:** the X-Talent proposal cannot claim "we beat existing methods on Synapse" by comparing only to TransUNet/Swin-UNet/HiFormer; you must include EMCAH-Net, DATTNet, U-Mamba, VM-UNet, and at least one nnU-Net configuration. Without those, reviewers will reject for inadequate baselines.

2. **HF+TA+PPA *worsens* HD95 vs. HF+TA alone (20.25 vs 14.74).** This is in your own table and your own discussion, but the thesis treats it as a small caveat. It actually means **PPA is hurting boundary precision**. If the PPA module is going forward into the master's project as-is, this needs a fix, not a footnote.

3. **PPA's transfer story is thin.** PPA was designed for **infrared small object detection** (HCF-Net, ICME 2024). The thesis justifies bringing it to organ segmentation by analogy ("small organs are like small objects"). That is a workable hook in an undergraduate thesis but is not a publishable contribution on its own — reviewers will ask why the multi-branch design choices that worked for IR small targets are right for soft, large, blurry organ boundaries. The HD95 regression from PPA is consistent with the suspicion that PPA's local branches sharpen detection at the cost of boundary smoothness.

4. **Synapse is a very small benchmark (30 cases)** and is increasingly seen as a "tweak-and-overfit" target. The X-Talent proposal already plans to add ACDC, BraTS, CHAOS — keep that. **Without 3+ datasets you will not pass review at *Medical Image Analysis* or *TMI*.**

5. **No nnU-Net comparison.** The 2024 NeurIPS-track paper "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation" (arXiv 2404.09556) is now widely cited by reviewers to reject claims of "we beat the U-Net family" that don't include nnU-Net. As of 2026 you cannot avoid this comparison.

6. **2D, not 3D.** Synapse is volumetric (CT slices); you train 2D. Most clinical segmentation work in 2025–2026 is 3D (or 2.5D). nnU-Net, SegMamba-V2, U-Mamba, VISTA3D are 3D. Staying purely 2D is a publishability drag.

7. **No self-supervised pretraining anywhere.** The thesis uses ImageNet-pretrained CNN+Swin weights. For medical CT this is suboptimal and is now considered a soft baseline. Domain SSL (SimMIM/MAE on CT, or DINOv2-distilled medical features) is one of the most cost-effective publishable improvements available to you.

### Net assessment

Your undergraduate thesis is a *valid baseline* for further work but is **not, by itself, a publishable paper in 2026** — the result is 4–5 DSC behind SOTA and the contribution (PPA + TA on top of HiFormer) is incremental, with one of the modules (PPA) actually hurting HD95. That doesn't mean the project is wrong; it means the master's project needs to do real new work, not just add more modules to the existing stack.

---

## 3. Critical view of the four proposed X‑Talent innovations

The proposal lists four major architectural/methodological additions on top of HF+TA+PPA. Each is independently plausible, but **stacking all four into one master's thesis is the central risk of the project**, and would also make the resulting paper unreviewable. Below is the honest state of each in 2026.

### 3.1 Hybrid Mamba-Former — *crowded but still publishable with a sharp angle*

**State of the field, April 2026:** Mamba in medical segmentation is no longer a frontier; it's a saturated subfield.

* **Already published:** U-Mamba (Jan 2024, the canonical first one), VM-UNet (Feb 2024, ACM TOMM 2025), SegMamba & **SegMamba-V2** (now in *IEEE TMI* / arXiv), Swin-UMamba, LightM-UNet, SCM-UNet (Elsevier 2025), MedMamba, H-VMUNet, "Vision Mamba and xLSTM-UNet" (Sci. Rep. 2025), "What can Mamba do for 3D Volumetric Medical Image Segmentation?" (OpenReview 2025), plus a comprehensive survey (arXiv 2410.02362v3) listing dozens of Mamba-medical models.
* **Implication:** "I added Mamba to a Transformer for medical segmentation" is already a paper template. You will need a *specific* angle. Concretely viable angles:
  * **Mamba where it provably helps, attention where it provably helps.** A per-block (or per-token) gating between SSM and attention, with an analysis of *when* each wins on different organs/scales. Most existing Mamba-medical models are uniform — they replace attention everywhere or stack Mamba in fixed positions. A learned, principled allocation is still under-explored.
  * **3D volumetric Mamba on Synapse + ACDC + BraTS at matched FLOPs vs. nnU-Net.** Strong and underdone.
  * **Mamba *for the small-organ / boundary problem* specifically.** Your existing HF+TA+PPA story already centers on small organs (gallbladder, pancreas) and HD95. Mamba's long-range modelling is more naturally suited to boundary continuity than to small-object emphasis — this is actually a *non-trivial* hypothesis worth testing.
* **What will not work:** "We replaced HiFormer's Swin block with a Mamba block." Reviewers will compare to U-Mamba, VM-UNet, and SegMamba-V2 directly.

### 3.2 KAN-augmented attention — *the riskiest of the four*

**State of the field, April 2026:** KAN was proposed in May 2024; the medical-segmentation flavours followed almost immediately. U-KAN, IKANet, MM-UKAN++, KANSeg, U-KABS, "Hybrid KAN", "Fully KAN deep model" already exist.

* **Documented limitation in the literature itself:** "deeply stacked KANs are practically impossible due to high training difficulties and substantial memory requirements." Most papers can only afford a few KAN layers, which dilutes the KAN-vs-MLP comparison.
* **Honest assessment of "KAN replaces MLP in attention":** The original KAN paper's interpretability claim is debated; on standard image benchmarks KAN typically loses on speed and ties on accuracy. The novelty argument for "KAN inside attention for medical segmentation" exists, but only if you can show *one* concrete benefit that B-spline/Bernstein activations enable that GELU/SiLU MLPs do not — e.g., better boundary fidelity, cleaner uncertainty calibration, or smaller models at matched accuracy.
* **My recommendation:** Treat KAN as a "stretch" experiment, not a thesis pillar. If after 4–6 weeks of trying KAN-augmented attention you do not see a clear, replicable win on at least one metric on at least two datasets, drop it. Do not write a paper around KAN unless the win is real.

### 3.3 Structure-aware Prompt Adapter — *dominated by MedSAM-Adapter*

**State of the field, April 2026:** The "Prompt Adapter for medical segmentation" niche is now occupied by:

* **MedSAM-Adapter** (Wu et al., *Medical Image Analysis* 2025, MIA 102:103547) — already in your own reference list as [5].
* **VISTA3D** (CVPR 2025) — unified 3D segmentation foundation model, also in your refs as [1].
* **Medical SAM 2** and several SAM-based adapter variants.

These are *strong* baselines. "I designed an adapter for medical segmentation" without explicit positioning vs. MedSAM-Adapter is unpublishable.

* **Where there's still room:** prompt adapters that operate *without* SAM/foundation-model checkpoints — i.e., adapters for your own HiFormer/Mamba backbone for cross-modal transfer between Synapse (CT, abdomen) and BraTS (MRI, brain) and ACDC (MRI, heart). That is a genuinely smaller and harder problem because you don't have SAM's general visual prior.
* **Honest assessment:** even with that positioning, the Prompt Adapter angle is a *medium-strength* contribution at best. It works as a secondary chapter of the thesis, not as the headline.

### 3.4 Evidential Deep Learning + uncertainty visualization — *useful but not novel by itself*

**State of the field, April 2026:**

* Region-based EDL for brain tumor segmentation (NCAA 2022) is the canonical reference.
* Recent extensions: "Uncertainty-Error correlations in EDL for biomedical segmentation" (arXiv 2410.18461, 2025); ESPD-Net (Frontiers in AI 2026, Dirichlet evidential theory for liver fibrosis nodules); "Deep evidential fusion with uncertainty quantification and reliability learning for multimodal medical image segmentation" (Information Fusion 2025, your ref [2]).
* **What this means:** EDL on top of an existing segmentation backbone is a known, well-studied recipe. As a sole contribution it is no longer fresh.
* **Where it can still be a real contribution:** *EDL combined with Mamba* is comparatively under-explored. Almost all EDL-segmentation papers use CNN/Swin backbones. A clean study of how Dirichlet-evidential heads behave on top of SSM features (where the receptive field structure is fundamentally different from attention) would be a publishable secondary contribution.

### 3.5 The single most important critique of the proposal

**Four parallel innovations is not a master's thesis, it is four half-finished papers.** This is the single biggest risk. The proposal as written — Mamba + KAN + Prompt Adapter + Evidential Uncertainty, all together, plus four datasets — cannot be executed at publishable depth in 14 remaining months by one student on one V100. Reviewers reading the resulting paper will see four mediocre contributions instead of one clean one and will reject for lack of focus.

**You need to cut at least two of the four.**

---

## 4. Recommendation: a focused 14-month plan

Below are three feasible packagings of the proposal, in descending order of how strongly I recommend them. Each is one paper, not four.

### Track A — *Mamba × Uncertainty for medical segmentation* (strongly recommended)

> "An efficient SSM-based hybrid architecture with evidential uncertainty for multi-organ segmentation, validated on Synapse, ACDC and BraTS."

* **Keep:** HiFormer skeleton, Triplet Attention (it's your strongest, almost-free win). 3D extension. Mamba-based long-range block.
* **Cut:** PPA (it hurts HD95), KAN, Prompt Adapter.
* **Add:** A Dirichlet/EDL head on top of the segmentation logits, with uncertainty-aware loss (Bayesian risk loss). Compare uncertainty maps to MC-Dropout and Deep Ensembles, not just visualize.
* **Why publishable:** Mamba × EDL on medical segmentation is the smallest, cleanest, most under-served gap in the four directions you proposed. The paper has a clear story ("SSM features change uncertainty calibration in this specific way"), strong baselines, and a clinically meaningful pitch.
* **Realistic target:** *Computers in Biology and Medicine*, *Knowledge-Based Systems*, *Information Fusion*, *Pattern Recognition*, or — with strong execution — *Medical Image Analysis* / *IEEE TMI*.

### Track B — *Adaptive attention/SSM allocation for boundary-aware segmentation* (medium risk, higher ceiling)

> "When does attention beat Mamba in medical segmentation, and a learned per-token gating that achieves the Pareto frontier of DSC and HD95 at matched FLOPs."

* **Keep:** HiFormer + TA. Datasets Synapse, ACDC, BraTS, CHAOS.
* **Cut:** PPA, KAN, Prompt Adapter, EDL.
* **Add:** A Mamba branch and a small gating network that decides per-block (or per-token) whether to route through attention or SSM. Compare to U-Mamba, VM-UNet, SegMamba-V2, EMCAH-Net at matched FLOPs.
* **Why publishable:** The "ViT vs. Mamba in medical segmentation" question is one of the few questions reviewers in 2026 still genuinely care about. A clean answer with a principled hybrid is a strong paper. **Risky** because you have to actually demonstrate the Pareto win — it might not exist.
* **Realistic target:** same as Track A; potentially *MICCAI 2027*.

### Track C — *Tightened version of the existing proposal* (least recommended)

If you do not want to deviate from the X-Talent proposal:

* **Keep at most two innovations.** Most defensible pair: Mamba (architecture) + EDL (uncertainty). Drop KAN and Prompt Adapter.
* **Fix the HD95 regression of PPA** (Section 2 issue #2) explicitly — either drop PPA, or restrict it to specific small-organ branches, or change its multi-scale fusion to preserve boundary smoothness.
* **Add nnU-Net comparison** — non-negotiable.
* **Move to 3D** on at least one of the datasets (Synapse or BraTS).

> **My recommendation: Track A.** It maximizes publishability under the actual constraints (one student, ~14 months, ~V100 budget), it preserves the meaningful parts of your undergrad thesis (the Triplet Attention win, the Synapse pipeline), it positions you on the smallest gap in the field, and it leaves room for a second/follow-up paper later if the first works.

---

## 5. Concrete next steps (in order)

You are roughly 10 months into an 18-month plan and the repository contains no code. The next 6 weeks should change that, not produce more documents.

### Week 0 (this week) — Lock the Track

Discuss Sections 3 and 4 of this document with Prof. Yang (杨涛). Decide between Track A, Track B, and Track C, and write **one paragraph** of hypothesis + **one paragraph** of "what success looks like" before any code.

### Weeks 1–2 — Recover and re-baseline the existing thesis code

* Get the HF + TA + PPA training script committed to this repo.
* Reproduce your own Table 5.1 numbers on Synapse from scratch on the current hardware. **If you cannot reproduce within ±0.3 DSC, the rest of the project is built on sand.** This is the single highest-value sanity check available.
* Create a clean repo layout:

```
vit-refined/
├── src/
│   ├── models/        (HiFormer, TA, PPA, plus baselines you'll add: U-Mamba, VM-UNet, nnU-Net wrapper)
│   ├── ssl/           (only if Track adds SSL pretraining)
│   ├── data/          (Synapse, ACDC, BraTS, CHAOS dataloaders; consistent splits)
│   ├── train.py
│   └── eval.py
├── configs/           (one YAML per experiment; under git)
├── scripts/           (reproducer scripts, FLOPs counter, uncertainty viz)
├── results/           (.csv logs only; never commit checkpoints)
├── pyproject.toml
└── README.md          (replace the one-line placeholder)
```

* Use the standard tooling — `monai` for 3D medical IO and metrics, `nnunetv2` package for nnU-Net runs, `timm` for ViT/Swin blocks, `mamba-ssm` for Mamba blocks, `wandb` for logging, `fvcore` for FLOPs. Don't reinvent any of these.

### Weeks 3–4 — Add the must-have baselines

Run, on Synapse and at least one other dataset (ACDC), the following baselines, *under your own preprocessing pipeline*, before introducing any new method:

* nnU-Net (default 3D config — use the official package, do not hand-implement).
* U-Mamba (or VM-UNet) on the same split.
* HiFormer + TA (your best 2024 result, reproduced).
* TransUNet and Swin-UNet (your existing baselines).

Why this matters: when reviewers ask "why didn't you compare to nnU-Net / U-Mamba / EMCAH-Net," you need numbers in your own pipeline, not just numbers cited from their papers. This is exactly the issue the "nnU-Net Revisited" paper raised.

### Weeks 5–14 — Implement the chosen Track and ablate

For Track A, the concrete shopping list is:

1. **3D HiFormer-Mamba block** — replace the deepest Swin stage with a Mamba block (start with bidirectional Mamba; consider VMamba's SS2D scan).
2. **Triplet Attention head** — keep, it's almost free.
3. **EDL head** — replace softmax with Dirichlet output and Bayesian risk loss; train end-to-end. Reference: Region-based EDL (NCAA 2022), Uncertainty-Error correlations (arXiv 2410.18461).
4. **Drop PPA** unless an ablation shows it improves HD95 in the new architecture. If you keep it, restrict it to the shallowest stage where the small-organ argument applies.
5. **Ablations** (mean ± std over **3 seeds**, single-seed numbers are no longer accepted at MIA/TMI):
   * Backbone: Swin-only vs. Swin+Mamba vs. Mamba-only.
   * Loss: CE+Dice vs. EDL (Dirichlet + Bayesian risk).
   * Head: with/without TA.
   * Dataset scale: Synapse-only vs. Synapse+ACDC vs. all four.
   * FLOPs match: report all numbers at fixed FLOPs, not fixed parameters.

### Weeks 12–18 — Draft the paper *while* finishing experiments

Start the related-work section the moment the first end-to-end run completes. Drafting the related work is the single best way to discover, before submission, that your contribution overlaps with a paper from 2024 you hadn't seen. It is much cheaper to discover that *during* writing than after.

### Months 18–24 — Submit, revise, second contribution

Realistic venue ladder, descending by difficulty, for one student in your situation:

1. *Medical Image Analysis* / *IEEE TMI* — possible if the Pareto win against U-Mamba and nnU-Net is real, with three datasets and rigorous calibration.
2. *Computer Methods and Programs in Biomedicine*, *Information Fusion*, *Knowledge-Based Systems*, *Pattern Recognition*, *Computers in Biology and Medicine*. **These are the realistic primary targets** for a tight Track-A paper.
3. *MICCAI 2027* main conference — if timing aligns and the submission is camera-ready.
4. MICCAI workshops (e.g., PRIME, where DAE-Former in your reference list was published) — a strong fallback.

---

## 6. Things you should sanity-check yourself

These are points the analysis depends on but that I cannot verify without your help.

1. **Does the November 2025 progress report already commit to a specific direction with the advisor?** If yes, my Track recommendation must be reconciled with that, not contradict it.
2. **What compute do you actually have?** A single V100 (32GB) is enough for 2D Synapse experiments at any scale, but Mamba on 3D BraTS at 128³ resolution will be tight; plan accordingly.
3. **Is the original undergraduate thesis code preserved?** If not, reproducing Table 5.1 will be more painful than expected and Week 1–2 may slip.
4. **Were the three-dataset training experiments you mention in the proposal already started?** If yes, share the logs; if no, that's a chunk of work to schedule.
5. **Does the X-Talent program have explicit publication requirements** (e.g., "must publish a Q1 paper")? That changes the venue ladder above.

If you share the November 2025 progress report content and your advisor's current preferred direction, I can do a second pass that reconciles this analysis with the institutional context.

---

## 7. References used

Backbone & baselines:

* [HiFormer (WACV 2023)](https://openaccess.thecvf.com/content/WACV2023/papers/Heidari_HiFormer_Hierarchical_Multi-Scale_Representations_Using_Transformers_for_Medical_Image_Segmentation_WACV_2023_paper.pdf)
* [TransUNet (arXiv 2102.04306)](https://arxiv.org/abs/2102.04306)
* [Swin-UNet (ECCV 2022 W)](https://arxiv.org/abs/2105.05537)
* [HCF-Net — origin of PPA (ICME 2024)](https://arxiv.org/abs/2403.10778)
* [Triplet Attention (WACV 2021)](https://arxiv.org/abs/2010.03045)
* [nnU-Net (Nature Methods 2021)](https://www.nature.com/articles/s41592-020-01008-z)
* [nnU-Net Revisited — A Call for Rigorous Validation in 3D Medical Image Segmentation (arXiv 2404.09556)](https://arxiv.org/abs/2404.09556)

Mamba in medical segmentation:

* [VM-UNet — Vision Mamba UNet (arXiv 2402.02491)](https://arxiv.org/abs/2402.02491)
* [SegMamba-V2 (PubMed)](https://pubmed.ncbi.nlm.nih.gov/40679879/)
* [What can Mamba do for 3D Volumetric Medical Image Segmentation? (OpenReview 2025)](https://openreview.net/forum?id=m3cKeqvC7z)
* [SCM-UNet — Spatial-channel Mamba UNet (Elsevier 2025)](https://www.sciencedirect.com/science/article/abs/pii/S105120042500572X)
* [A Comprehensive Survey of Mamba Architectures for Medical Image Analysis (arXiv 2410.02362)](https://arxiv.org/html/2410.02362v3)
* [Vision Mamba (arXiv 2401.09417)](https://arxiv.org/abs/2401.09417)
* [Spatial-Mamba (ICLR 2025)](https://openreview.net/forum?id=iDe1mtxqK5)
* [2DMamba (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_2DMamba_Efficient_State_Space_Model_for_Image_Representation_with_Applications_CVPR_2025_paper.pdf)

KAN in medical segmentation:

* [A hybrid Kolmogorov-Arnold network for medical image segmentation (arXiv 2602.07702)](https://arxiv.org/abs/2602.07702)
* [Fully Kolmogorov-Arnold Deep Model in Medical Image Segmentation](https://arxiv.org/html/2602.03156)
* [MM-UKAN++ (PubMed 40031744)](https://pubmed.ncbi.nlm.nih.gov/40031744/)
* [KANSeg — multi-organ KAN segmentation (Elsevier 2025)](https://www.sciencedirect.com/science/article/abs/pii/S1051200425004944)
* [IKANet — Interpretable Medical Image Segmentation (Springer 2025)](https://link.springer.com/chapter/10.1007/978-981-95-5634-2_14)

Foundation models / prompt adapters for medical segmentation:

* [Medical SAM Adapter (Medical Image Analysis 2025)](https://www.sciencedirect.com/science/article/abs/pii/S136184152500137X)
* [VISTA3D (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/He_VISTA3D_A_Unified_Segmentation_Foundation_Model_for_3D_Medical_Imaging_CVPR_2025_paper.pdf)

Evidential / uncertainty:

* [Region-based Evidential Deep Learning for Brain Tumor Segmentation (NCAA 2022)](https://link.springer.com/article/10.1007/s00521-022-08016-4)
* [Uncertainty-Error correlations in EDL for biomedical segmentation (arXiv 2410.18461)](https://arxiv.org/html/2410.18461)
* [ESPD-Net — Edge-Semantics Probabilistic Dirichlet Network (Frontiers in AI 2026)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1801342/full)
* [Deep evidential fusion with uncertainty quantification and reliability learning for multimodal medical image segmentation (Information Fusion 2025)](https://www.sciencedirect.com/science/article/pii/S1566253524004391)

Recent strong Synapse-multi-organ models (mandatory baselines):

* [EMCAH-Net (PMC 11994538, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11994538/)
* [Dual-attention transformer-based hybrid network — DATTNet (Nature Sci. Rep. 2024)](https://www.nature.com/articles/s41598-024-76234-y)
* [EM-Net (arXiv 2409.17675)](https://arxiv.org/pdf/2409.17675)

General reviews:

* [Vision Transformers on the Edge (arXiv 2503.02891, 2025)](https://arxiv.org/html/2503.02891v3)
* [Advances in attention mechanisms for medical image segmentation (Computer Science Review 2025)](https://www.sciencedirect.com/science/article/pii/S1574013724000844)
* [Medical image segmentation review: The success of u-net (IEEE TPAMI 2024)](https://ieeexplore.ieee.org/document/10643318)
