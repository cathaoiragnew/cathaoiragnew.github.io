---
layout: page
title: 3D Dental CBCT Segmentation
subtitle: How U-Mamba2, the winner of the ToothFairy3 challenge, labels the jaws, teeth, pulp and nerve canals in a CBCT scan
share-img: /assets/img/toothfairy3_hero.png
---

<style>
  /* The site's links are white like the body text; underline them on this page so they read as links. */
  main p a, main li a, main td a { text-decoration: underline; text-underline-offset: 2px; }
  /* The theme stripes table rows #f8f8f8, which hides white text; use a dark stripe instead,
     and let wide tables scroll sideways on phones. */
  main table { display: block; max-width: 100%; overflow-x: auto; margin-bottom: 1rem; }
  main table tr:nth-child(2n) { background-color: #1a1a19; }
  main table tr th, main table tr td { border-color: #4a4945; }
  /* "Segmentation" is too long for the theme's 50px title on phones and breaks mid-word. */
  @media (max-width: 575px) { .intro-header .page-heading h1 { font-size: 2.25rem; } }
</style>

<img src="{{ '/assets/img/toothfairy3_hero.png' | relative_url }}"
     alt="One CBCT scan shown three ways: an axial slice in greyscale, the same slice with its ground-truth labels coloured in, and all labelled structures in 3D"
     style="width:100%; max-width:900px;">

*One CBCT scan from ToothFairy3. Left: an axial slice through the lower jaw. Middle: the same slice with the dataset's ground-truth labels. Right: every labelled structure in 3D.*

My [last write-up](../teeth_segmentation/) looked at intra-oral scans: 3D surfaces of the teeth and gums, as seen by a scanner in the mouth. This time I wanted to go below the surface. A **CBCT (cone-beam computed tomography)** scan is a 3D X-ray volume, so it shows what an intra-oral scan can't: the jawbone, the tooth roots, the pulp inside each tooth, and the nerve canals running through the jaw.

The fundamentals are still the same:

**input → feature vectors → operations → outputs**

The winning solution, U-Mamba2, combines three things: **a strong, well-tuned baseline, one targeted change to the architecture, and a lot of domain knowledge built into the training.** This page walks through all three.

## The benchmark: ToothFairy3

[ToothFairy3](https://ditto.ing.unimore.it/toothfairy3/){:target="_blank"} contains **532 CBCT volumes** labelled with **77 classes**. It comes in three sets, named by the first letter of each file:

- **Set A** (P, 417 volumes) and **Set B** (F, 63 volumes) come from the same scanner and overlap with the earlier ToothFairy2 dataset. Set B has a wider field of view, which includes the complete upper teeth.
- **Set C** (S, 52 volumes) is new, acquired on a different machine.

Compared with ToothFairy2, it adds **35 labels**: the **pulp of all 32 teeth**, the **left and right incisive canals** and the **lingual canal**. The other 42 classes cover the upper and lower jawbones, the left and right inferior alveolar canals (which carry the main nerve of the lower jaw), the maxillary sinuses, the pharynx, bridges, crowns, implants and each of the 32 teeth by [FDI number](https://en.wikipedia.org/wiki/FDI_World_Dental_Federation_notation){:target="_blank"}. Scans have **0.3 mm isotropic voxels**, stored in Hounsfield units.

The dataset was used for the [ToothFairy3 challenge](https://toothfairy3.grand-challenge.org/){:target="_blank"} at the ODIN workshop at MICCAI 2025, with two tasks:

- **Task 1, fast multi-class segmentation** of every class, where speed counts too. Teams are ranked separately on Dice and HD95 for every class and on average runtime, and the ranks are averaged, with runtime weighted as much as all the classes together.
- **Task 2, interactive segmentation of the inferior alveolar canals**, where the model is given user clicks as prompts.

This page focuses on Task 1.

## Same recipe, new domain

Set against a standard 2D segmentation model, the structure is the same:

| Step | 2D image model | U-Mamba2 |
|---|---|---|
| **Input** | Pixels, 3 numbers each (RGB) | A 160 × 288 × 288 patch of voxels, 1 number each (X-ray density) |
| **Feature vectors** | CNN encoder → feature maps | 3D residual CNN encoder → feature maps, with a Mamba2 layer at the bottleneck |
| **Operations** | Decoder with skip connections, then a class for every pixel | 3D decoder with skip connections, then post-processing |
| **Output** | A class for every pixel | A class for every voxel: 46 structures plus background |

After point clouds, this felt like home ground. A CBCT scan **does** have a grid, so everything from 2D convolutional networks carries over with 3D kernels. The difficulty is size. The scan on this page is 512 × 512 × 262 voxels, which is **68.7 million voxels**, far too many to feed a 3D network in one go. So the network sees one patch at a time (13.3 million voxels in the final model), and the full scan is covered with a sliding window whose predictions are blended together.

## U-Mamba2 architecture

U-Mamba2 ([code on GitHub](https://github.com/zhiqin1998/U-Mamba2){:target="_blank"}, [paper](https://arxiv.org/abs/2509.12069){:target="_blank"}) is by Zhi Qin Tan, Xiatian Zhu, Owen Addison and Yunpeng Li (team TAIR Lab). The repository is a fork of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet){:target="_blank"} v2, and the network is nnU-Net's residual encoder U-Net with one addition:

<a href="{{ '/assets/img/umamba2_architecture.svg' | relative_url }}" target="_blank"><img src="{{ '/assets/img/umamba2_architecture.svg' | relative_url }}"
     alt="U-Mamba2 architecture: a seven-stage residual encoder shrinks a 160 by 288 by 288 patch to 5 by 9 by 9 with 320 channels; a Mamba2 layer processes the 405 bottleneck tokens; a six-stage decoder with skip connections outputs 47 class scores per voxel. Below: pretraining, loss, mirroring and post-processing"
     style="width:100%; max-width:760px;"></a>

*The Task 1 network as configured for the final submission in the repository's ToothFairy3 instructions. Click the diagram to open it full size.*

### The backbone: nnU-Net's residual encoder U-Net

The encoder has **seven stages**. The first keeps full resolution, the next five each halve every side, and the last keeps the size, so a 160 × 288 × 288 patch ends up as a **5 × 9 × 9 grid of 320-channel feature vectors**. Stages are built from residual blocks (1, 3, 4, 6, 6, 6 and 6 of them), with feature widths growing from 32 to 320. The decoder mirrors this: each stage upsamples with a transposed convolution, concatenates the matching encoder features (the skip connection) and refines the result with convolutions. The output is a score for each class at every voxel. The team merged the 32 pulp classes into one, giving 46 structures plus background. This is the "nnU-Net ResEnc" family from [nnU-Net Revisited](https://arxiv.org/abs/2404.09556){:target="_blank"} (Isensee et al., 2024), a paper whose main message is that a well-configured CNN U-Net is still very hard to beat in 3D medical segmentation.

### The addition: Mamba2 at the bottleneck

Convolutions are local. Even at the bottleneck, each feature vector has mostly been built from its own neighbourhood. U-Mamba2 adds one layer that mixes information across the whole patch:

1. Flatten the 5 × 9 × 9 grid into a **sequence of 405 tokens**, each a 320-dimensional feature vector.
2. Apply LayerNorm, then **Mamba2**.
3. Reshape back to 5 × 9 × 9 and hand it to the decoder.

[Mamba](https://arxiv.org/abs/2312.00752){:target="_blank"} (Gu & Dao, 2023) is a state space model. It reads a sequence one token at a time and carries a fixed-size state forward, updating it with each token, so its cost grows **linearly** with sequence length, where attention grows quadratically. [Mamba2](https://arxiv.org/abs/2405.21060){:target="_blank"} (Dao & Gu, 2024) restricts the state update to a simpler form, which lets the same computation be written as matrix multiplications that run fast on GPUs. That efficiency is the paper's motivation for using Mamba2. The paper also reports that putting the block only at the bottleneck gave the best results for 3D CT. It's also where the sequence is shortest: 405 tokens, compared with 13.3 million voxels at full resolution.

Two details that only show up in the code:

- **The Mamba2 layer makes one forward pass over the sequence**, in raster order (row by row, slice by slice). Each token's output can draw on every token *before* it, not after it.
- **There is no residual connection around the layer.** Its output replaces the bottleneck features rather than being added to them.

For Task 2, the same network gets an extra branch: user clicks are turned into embeddings by a SAM-style point encoder, and two cross-attention blocks fuse them with the Mamba2 output.

### Training with dental knowledge

Most of what makes the solution specific to teeth isn't in the network. It's in how it's trained and what happens after it:

**Self-supervised pretraining.** Before seeing any labels, the network is trained as a [disruptive autoencoder](https://arxiv.org/abs/2307.16896){:target="_blank"} (Valanarasu et al., 2023): parts of a scan are masked, downsampled or made noisy, and the network learns to reconstruct the original (L1 loss). On top of ToothFairy3, this uses 371 unlabelled CBCT scans from the STS-3D-Tooth dataset.

**Label smoothing between related structures.** Normally the training target for a voxel is 1 for its class and 0 for everything else. Here the true class gets 0.9 and the remaining 0.1 is shared by related classes, such as similar teeth or the connected nerve canals, so near-misses between them cost less.

**A heavier weight for the tiniest structures.** The two incisive canals and the lingual canal are thin and small. In this scan they cover 658, 819 and 136 voxels, against 1.43 million for the lower jawbone. Their loss weight is set to 10, so the large structures don't drown them out.

**Mirroring that respects left and right.** Flipping scans left to right is a standard nnU-Net augmentation, but it breaks labels that name a side: a flipped *left* canine looks exactly like a *right* canine. U-Mamba2 keeps the flip and swaps every left/right label with its partner whenever it happens (tooth 13 ↔ 23, left canal ↔ right canal, and so on). It does the same at test time, when predictions on flipped copies of the scan are averaged. The viewer below animates this.

It's a different answer to the problem ToothGroupNet faced in my last write-up. ToothGroupNet merged most left and right tooth types into one class and worked out the side from geometry afterwards. U-Mamba2 keeps separate classes and makes the augmentation respect them.

**Post-processing with anatomical sizes.** After prediction, any connected blob smaller than a per-class threshold is removed. Each threshold is the **0.5th percentile of that class's connected-component volumes in the training labels**, so it's set from the anatomy rather than from a particular model's mistakes.

## See it on the scan

Drag to rotate, scroll to zoom, and switch between the views. Hover over a structure to see its name.

<div class="cv" data-src="{{ '/assets/data/toothfairy3_F001.bin' | relative_url }}"></div>
<script type="module" src="{{ '/assets/js/cbct_viewer.js' | relative_url }}"></script>

*Everything shown comes from the dataset's ground-truth labels for one scan (ToothFairy3F_001), not from model predictions. Each structure is a surface extracted from its label with marching cubes and simplified for the web. The left–right flip applies the label pairs U-Mamba2 swaps during augmentation.*

## Results

The paper's validation results for Task 1 use a 90/10 split of ToothFairy3, stratified so both parts contain the same mix of data sources:

| Model | Dice | HD95 | Dice after post-processing | HD95 after post-processing | Time per scan (s) |
|---|---|---|---|---|---|
| SwinUNETR | 0.858 | 48.86 | 0.874 | 40.09 | 7.23 |
| nnU-Net ResEnc | 0.861 | 45.28 | 0.887 | 32.05 | 6.20 |
| U-Mamba | 0.865 | 42.06 | 0.896 | 25.88 | 6.98 |
| **U-Mamba2** | **0.873** | **41.08** | **0.908** | **21.35** | 6.81 |

Two things stand out. The post-processing step adds between 0.016 and 0.035 Dice to every model, more than the gap between the architectures. And the comparison isn't architecture alone: the paper notes that left–right mirroring was switched off in training for every model except U-Mamba2.

The ablation (Task 1 validation, without post-processing) shows where the dental knowledge helps most:

| Label smoothing | Weighted loss | Left–right mirroring | Dice (all classes) | Dice (incisive + lingual canals) |
|---|---|---|---|---|
|   |   |   | 0.867 | 0.617 |
| ✓ |   |   | 0.872 | 0.628 |
|   | ✓ |   | 0.870 | 0.635 |
|   |   | ✓ | 0.871 | 0.642 |
| ✓ | ✓ | ✓ | 0.873 | 0.646 |

On all classes together the gains are small, but on the three tiny canals the full set of changes lifts Dice from 0.617 to 0.646.

On the challenge's hidden test set, U-Mamba2 won both tasks. Task 1's podium, from the [challenge winners page](https://toothfairy3.grand-challenge.org/challenge-winners/){:target="_blank"}:

| Place | Team | Dice | HD95 | Runtime (s) | Mean rank |
|---|---|---|---|---|---|
| 1 | TAIR-Lab (U-Mamba2) | 0.84 | 38.17 | 40.58 | 3.1 |
| 2 | sjtu_eiee_2-426lab | 0.77 | 94.77 | 17.46 | 3.7 |
| 3 | Black_Myth | 0.85 | 33.23 | 90.04 | 3.8 |

Black_Myth had a slightly higher mean Dice and lower mean HD95, but took more than twice as long per scan. Because the ranking works per class and weights runtime heavily, U-Mamba2's balance of accuracy and speed came out on top.

## Knowing the anatomy is part of the model

What struck me most was how much of this winning solution is knowledge about teeth rather than new machinery. The architecture change is a single layer. Around it, each trick encodes one fact about the anatomy:

- **Some structures are related**, so near-misses between them are penalised less (label smoothing).
- **Some structures are tiny**, so they're weighted up (class weights).
- **The head is left–right symmetric, but the labels aren't**, so flips swap the labels (mirroring).
- **Every structure has a typical size**, so implausibly small blobs are removed (post-processing).

Each one is cheap to add. The network gets you most of the way; knowing the domain gets you the rest.

## References

- Tan, Zhu, Addison & Li, *U-Mamba2: Scaling State Space Models for Dental Anatomy Segmentation in CBCT*, ODIN workshop, MICCAI 2025. [arXiv:2509.12069](https://arxiv.org/abs/2509.12069){:target="_blank"} · [doi:10.1007/978-3-032-20711-1_12](https://doi.org/10.1007/978-3-032-20711-1_12){:target="_blank"} · [github.com/zhiqin1998/U-Mamba2](https://github.com/zhiqin1998/U-Mamba2){:target="_blank"}
- ToothFairy3 dataset, AImageLab, University of Modena and Reggio Emilia. [ditto.ing.unimore.it/toothfairy3](https://ditto.ing.unimore.it/toothfairy3/){:target="_blank"} · [challenge page](https://toothfairy3.grand-challenge.org/){:target="_blank"}
- Ma, Li & Wang, *U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation*, 2024. [arXiv:2401.04722](https://arxiv.org/abs/2401.04722){:target="_blank"}
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752){:target="_blank"}
- Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*, ICML 2024. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060){:target="_blank"}
- Isensee et al., *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation*, MICCAI 2024. [arXiv:2404.09556](https://arxiv.org/abs/2404.09556){:target="_blank"}
- Valanarasu et al., *Disruptive Autoencoders: Leveraging Low-level features for 3D Medical Image Pre-training*, 2023. [arXiv:2307.16896](https://arxiv.org/abs/2307.16896){:target="_blank"}

<small>The scan shown is `ToothFairy3F_001` from the ToothFairy3 dataset, licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/){:target="_blank"}. The images and 3D data on this page are derived from it and shared under the same licence.</small>
