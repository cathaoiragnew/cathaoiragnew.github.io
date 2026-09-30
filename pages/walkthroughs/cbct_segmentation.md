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
  /* Let the wide diagrams grow past the text column on big screens, centred on it. */
  .wide-fig { width: min(var(--fig-w), calc(100vw - 48px)); position: relative; left: 50%;
    transform: translateX(-50%); margin: 1rem 0; }
  .wide-fig img { width: 100%; display: block; }
</style>

<img src="{{ '/assets/img/toothfairy3_hero.png' | relative_url }}"
     alt="One CBCT scan shown three ways: an axial slice in greyscale, the same slice with its ground-truth labels coloured in, and all labelled structures in 3D"
     style="width:100%; max-width:900px;">

*One CBCT scan from ToothFairy3. Left: an axial slice through the lower jaw. Middle: the same slice with the dataset's ground-truth labels. Right: every labelled structure in 3D.*

My [last write-up](../teeth_segmentation/) looked at intra-oral scans: 3D surfaces of the teeth and gums, as seen by a scanner in the mouth. This time I wanted to go below the surface. A **CBCT (cone-beam computed tomography)** scan is a 3D X-ray volume, so it shows what an intra-oral scan can't: the jawbone, the tooth roots, the pulp inside each tooth, and the nerve canals running through the jaw.

The winning solution, U-Mamba2, combines three things: **a strong, well-tuned baseline, one targeted change to the architecture, and a lot of domain knowledge built into the training.** This page walks through all three.

## The benchmark: ToothFairy3

[ToothFairy3](https://ditto.ing.unimore.it/toothfairy3/){:target="_blank"} contains **532 CBCT volumes** labelled with **77 classes**. It comes in three sets, named by the first letter of each file:

- **Set A** (P, 417 volumes) and **Set B** (F, 63 volumes) come from the same scanner and overlap with the earlier ToothFairy2 dataset. Set B has a wider field of view, which includes the complete upper teeth.
- **Set C** (S, 52 volumes) is new, acquired on a different machine.

Compared with ToothFairy2, it adds **35 labels**: the **pulp of all 32 teeth**, the **left and right incisive canals** and the **lingual canal**. The other 42 classes cover the upper and lower jawbones, the left and right inferior alveolar canals (which carry the main nerve of the lower jaw), the maxillary sinuses, the pharynx, bridges, crowns, implants and each of the 32 teeth by [FDI number](https://en.wikipedia.org/wiki/FDI_World_Dental_Federation_notation){:target="_blank"}. Scans have **0.3 mm isotropic voxels**, stored in Hounsfield units.

The dataset was used for the [ToothFairy3 challenge](https://toothfairy3.grand-challenge.org/){:target="_blank"} at the ODIN workshop at MICCAI 2025, with two tasks:

- **Task 1, fast multi-class segmentation** of every class, where speed counts too. Teams are ranked separately on Dice and HD95 for every class and on average runtime, and the ranks are averaged, with runtime weighted as much as all the classes together.
- **Task 2, interactive segmentation of the inferior alveolar canals**, where the model is given user clicks as prompts.

This page focuses on Task 1. The challenge scores all 32 pulps as a single "pulp" class, so Task 1 is scored on **46 classes**: the 77 labels, with the 32 pulp labels counted as one (77 − 32 + 1 = 46).

Two scores measure accuracy. Both are computed for each class in each scan:

- **Dice** measures overlap between the predicted mask and the true mask: twice the overlapping volume, divided by the two volumes added together. 1 is a perfect match and 0 means no overlap.
- **HD95** (the 95th percentile Hausdorff distance) measures how far the predicted boundary strays from the true one. For every point on the boundary of one mask, take the distance to the nearest point on the boundary of the other, in both directions. HD95 is the 95th percentile of those distances, so the worst 5% (a few stray voxels) don't dominate. Lower is better. In the challenge's evaluation code it's measured in voxels, and a structure that is missed completely, or predicted where there is none, scores the length of the scan's diagonal. One miss therefore adds a lot to the average.

## Same recipe, new domain

Set against a standard 2D segmentation model, the structure is the same:

| Step | 2D image model | U-Mamba2 |
|---|---|---|
| **Input** | An RGB image, 3 × H × W (3 channels: red, green, blue) | A CBCT patch, 1 × 160 × 288 × 288 (1 channel: X-ray density) |
| **Feature vectors** | CNN encoder → feature maps | 3D residual CNN encoder → feature maps, with a Mamba2 layer at the bottleneck |
| **Operations** | Decoder with skip connections, then a class for every pixel | 3D decoder with skip connections, then post-processing |
| **Output** | Class scores, C × H × W → argmax over the C channels → an H × W label map | Class scores, 47 × 160 × 288 × 288 → argmax over the 47 channels → a 160 × 288 × 288 label map (46 structures plus background) |

After point clouds, this felt like home ground. A CBCT scan **does** have a grid, so everything from 2D convolutional networks carries over with 3D kernels. The difficulty is size.

### Scans in, patches through the network

The network never sees a whole scan. It looks through a fixed 160 × 288 × 288 window, the patch size nnU-Net chose when it planned the model, and that window is smaller than the scan on every axis:

| | Slices | Rows | Columns | Voxels | Size at 0.3 mm |
|---|---|---|---|---|---|
| The scan on this page | 262 | 512 | 512 | 68.7 million | 79 × 154 × 154 mm |
| One patch | 160 | 288 | 288 | 13.3 million | 48 × 86 × 86 mm |

One patch covers about a fifth of this scan. Feeding the whole scan at once would need about five times the memory for the network's activations, and one patch is already large: the first encoder level alone holds 32 × 13.3 million ≈ 425 million numbers. Scans in the dataset also differ in size, and a fixed patch size handles them all the same way.

- **Training:** every step crops a random 160 × 288 × 288 patch from a training scan, so the network learns from many different patches of each scan.
- **Inference:** the window slides across the scan. U-Mamba2 sets the step to at most 95% of the patch size (nnU-Net's default is 50%), and nnU-Net spaces the positions evenly so the last one ends at the scan's edge. For this scan that gives two positions along each axis: slices 0–159 and 102–261, and rows and columns 0–287 and 224–511. So **2 × 2 × 2 = 8 patches** cover the whole scan, overlapping by 58 slices and by 64 rows and columns. Fewer patches means less computation, which matters here because runtime counts in the Task 1 ranking. Each patch is also predicted on flipped copies (see mirroring below).
- **Stitching:** each patch gives 47 × 160 × 288 × 288 class scores. These are added into one 47 × 262 × 512 × 512 score volume, weighted by a Gaussian that favours each patch's centre, where the network has context on all sides. The argmax over the 47 channels is taken once on that volume, giving the final 262 × 512 × 512 label map.

This scan is already at the dataset's 0.3 mm spacing, so nnU-Net doesn't resample it before cutting it into patches.

## U-Mamba2 architecture

U-Mamba2 ([code on GitHub](https://github.com/zhiqin1998/U-Mamba2){:target="_blank"}, [paper](https://arxiv.org/abs/2509.12069){:target="_blank"}) is by Zhi Qin Tan, Xiatian Zhu, Owen Addison and Yunpeng Li (team TAIR Lab). The repository is a fork of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet){:target="_blank"} v2, and the network is nnU-Net's residual encoder U-Net with one addition: a Mamba2 layer at the bottleneck.

### The backbone: nnU-Net's residual encoder U-Net

<div class="wide-fig" style="--fig-w: 1200px;"><a href="{{ '/assets/img/umamba2_architecture.svg' | relative_url }}" target="_blank"><img src="{{ '/assets/img/umamba2_architecture.svg' | relative_url }}"
     alt="U-Mamba2 architecture drawn as 3D blocks in a U shape. Encoder: a CBCT patch of 1 by 160 by 288 by 288 goes through a stem convolution and residual blocks with 32, 64, 128, 256, 320 and 320 channels, halving every side at each stage down to 5 by 9 by 9. Bottleneck: a seventh stage of 320 channels at 5 by 9 by 9, then a Mamba2 layer. Decoder: at each level, a transposed convolution doubles every side, the matching encoder features are concatenated and one 3 by 3 by 3 convolution follows. A final 1 by 1 by 1 convolution gives class scores of 47 by 160 by 288 by 288, and the argmax over the 47 channels gives a 160 by 288 by 288 map of class labels."></a></div>

*The Task 1 network as configured for the final submission in the repository's ToothFairy3 instructions, for one 160 × 288 × 288 patch. Numbers above the blocks are channels; the sizes on the green arrows are the feature-map size at each level. At inference the argmax is taken once for the whole scan, after the scores from all patches are combined (see [Scans in, patches through the network](#scans-in-patches-through-the-network)). Click the diagram to open it full size.*

Following one patch through the network:

- **In:** a 1 × 160 × 288 × 288 patch, one channel of X-ray density.
- **Stem:** one 3 × 3 × 3 convolution lifts it to 32 channels. As everywhere in nnU-Net, the convolutions use instance normalisation and LeakyReLU activations.
- **Encoder:** seven stages of **residual blocks** (1, 3, 4, 6, 6, 6 and 6 of them). A residual block is two 3 × 3 × 3 convolutions plus a shortcut that adds the block's input back to its output. Stages 2 to 6 start with a block whose first convolution has stride 2, which halves every side: 160 × 288 × 288 → 80 × 144 × 144 → 40 × 72 × 72 → 20 × 36 × 36 → 10 × 18 × 18 → 5 × 9 × 9. At the same time the channels grow: 32 → 64 → 128 → 256 → 320, then stay at 320. Stage 7 keeps the 5 × 9 × 9 size.
- **Bottleneck:** the patch is now a **5 × 9 × 9 grid of 320-channel feature vectors**. Neighbouring vectors are 32 voxels (about 1 cm) apart in the scan. This is where the Mamba2 layer sits.
- **Decoder:** six stages mirror the encoder. Each one doubles every side with a transposed convolution, concatenates the encoder features from the same level (the skip connection, for example 128 + 128 = 256 channels at 40 × 72 × 72) and applies one 3 × 3 × 3 convolution to bring the channels back down.
- **Out:** a 1 × 1 × 1 convolution turns the 32 channels at every voxel into **47 class scores**, so the output is 47 × 160 × 288 × 288: 46 structures plus background. The 46 match the challenge's classes, because the team merged the 32 pulp labels into one, just as the challenge scores them.
- **Label:** the class at each voxel is the **argmax** over the 47 channels, the one with the highest score. That gives a 160 × 288 × 288 map with one class number per voxel. At inference, U-Mamba2 applies the argmax to the raw scores (logits) directly, with no softmax: softmax keeps the order of the scores, so it wouldn't change which one wins. Softmax is only used during training, inside the loss.
- **Whole scan:** at inference the argmax isn't taken per patch. The scores from all 8 overlapping patches are combined into one score volume first, and the argmax is taken once on that, as described [above](#scans-in-patches-through-the-network).

This is the "nnU-Net ResEnc" family from [nnU-Net Revisited](https://arxiv.org/abs/2404.09556){:target="_blank"} (Isensee et al., 2024), a paper whose main message is that a well-configured CNN U-Net is still very hard to beat in 3D medical segmentation.

### The addition: Mamba2 at the bottleneck

Convolutions are local. Even at the bottleneck, each feature vector has mostly been built from its own neighbourhood. U-Mamba2 adds one layer that mixes information across the whole patch:

1. Flatten the 5 × 9 × 9 grid into a **sequence of 405 tokens**, each a 320-dimensional feature vector.
2. Apply LayerNorm, then **Mamba2**.
3. Reshape back to 5 × 9 × 9 and hand it to the decoder.

[Mamba](https://arxiv.org/abs/2312.00752){:target="_blank"} (Gu & Dao, 2023) is a state space model. Here is what that means in practice:

<div class="wide-fig" style="--fig-w: 1000px;"><a href="{{ '/assets/img/mamba2_bottleneck.svg' | relative_url }}" target="_blank"><img src="{{ '/assets/img/mamba2_bottleneck.svg' | relative_url }}"
     alt="What the Mamba2 layer does. One: the 5 by 9 by 9 bottleneck grid with 320 channels is flattened into 405 tokens. Two: the scan. A fixed-size state h is carried from token to token. At each token the old state is scaled by a, between 0 and 1, the token is written in through B, and an output y is read out through C. Three: the steps inside U-Mamba2's layer: LayerNorm, a linear layer from 320 to 640 channels, a short convolution over 4 tokens, the scan with 8 heads of 80 channels and state size 32, gating and RMSNorm, and a linear layer back to 320, reshaped to 5 by 9 by 9."></a></div>

*What the Mamba2 layer does to the bottleneck features. Click the diagram to open it full size.*

- **A small memory, carried along.** The layer reads the 405 tokens in order and keeps a **state**: a fixed-size memory (in U-Mamba2, 8 heads, each holding 80 × 32 numbers). At each token it does three things: it fades the old state a little (**forget**), adds the new token into it (**write**) and reads an output from it (**read**).
- **The token decides.** How much to forget, what to write and what to read are all computed from the token itself. This is what makes Mamba *selective*: it can hold on to what matters and let the rest fade. Older state space models used the same fixed update for every token.
- **Linear cost.** The state never grows, so the cost grows **linearly** with the number of tokens. Attention compares every token with every other token, so its cost grows with the square: 405 × 405 pairs here.
- **What Mamba2 changes.** [Mamba2](https://arxiv.org/abs/2405.21060){:target="_blank"} (Dao & Gu, 2024) makes the "forget" step a single number per head. That simpler form lets the scan be computed with matrix multiplications instead of Mamba's specialised scan, which parallelises better. That efficiency is the paper's motivation for using Mamba2.

The paper also reports that putting the block only at the bottleneck gave the best results for 3D CT. It's also where the sequence is shortest: 405 tokens, compared with 13.3 million voxels at full resolution.

For Task 2, the same network gets an extra branch: user clicks are turned into embeddings by a SAM-style point encoder, and two cross-attention blocks fuse them with the Mamba2 output.

### Training with dental knowledge

Most of what makes the solution specific to teeth isn't in the network. It's in how it's trained and what happens after it:

**Self-supervised pretraining.** Before seeing any labels, the whole network is trained as a [disruptive autoencoder](https://arxiv.org/abs/2307.16896){:target="_blank"} (Valanarasu et al., 2023): the input is a disrupted patch, and the target is the original patch.

1. **Disrupt** each 1 × 160 × 288 × 288 training patch, with intensities normalised to zero mean and unit variance, in three steps:
   - add Gaussian noise (standard deviation about 0.32),
   - downsample by 4 on each side and upsample back with nearest-neighbour interpolation, which removes fine detail,
   - mask 30% of its 16 × 16 × 16-voxel blocks by setting them to −2.
2. **Predict:** pass the disrupted patch through U-Mamba2. For this stage the output layer has a single channel instead of 47, so the output is 1 × 160 × 288 × 288: a predicted intensity for every voxel.
3. **Compare** the prediction with the original patch using the **L1 loss**, the mean absolute error: \|predicted − original\| at every voxel, averaged over the patch.

It's a regression on voxel intensities. L1 penalises errors in proportion to their size, where L2 (mean squared error) squares them and so punishes the few large errors much more. To reconstruct masked and low-resolution regions, the network has to learn the typical shapes and intensities of jaws, teeth and canals, which is a good starting point for segmentation. No labels are needed, so on top of ToothFairy3 this stage also uses 371 unlabelled CBCT scans from the STS-3D-Tooth dataset. The segmentation training then starts from these weights.

**The segmentation loss.** As in standard nnU-Net, the training loss adds two terms. **Cross-entropy** scores the predicted class probabilities at every voxel against the target. **Dice loss** rewards a high Dice score (see above) for each structure, averaged over the structures, so a small structure counts as much as a large one. The next two tricks change the targets and weights inside this loss.

**Label smoothing between related structures.** Normally the training target for a voxel is 1 for its class and 0 for everything else. Here the true class gets 0.9 and the remaining 0.1 is shared by related classes, so near-misses between them cost less. A tooth shares with its neighbours on the same side of the same jaw and with the pulp, each central incisor with the one across the midline, each inferior alveolar canal with the incisive canal on the same side, and the two sinuses with each other.

**A heavier weight for the tiniest structures.** The two incisive canals and the lingual canal are thin and small. In this scan they cover 658, 819 and 136 voxels, against 1.43 million for the lower jawbone. In the cross-entropy term their weight is set to 10, against 1 for every other class, so the large structures don't drown them out.

**Mirroring that respects left and right.** Flipping scans left to right is a standard nnU-Net augmentation, but it breaks labels that name a side: a flipped *left* canine looks exactly like a *right* canine. U-Mamba2 keeps the flip and swaps every left/right label with its partner whenever it happens (tooth 13 ↔ 23, left canal ↔ right canal, and so on). It does the same at test time, when predictions on flipped copies of the scan are averaged.

It's a different answer to the problem ToothGroupNet faced in my last write-up. ToothGroupNet merged most left and right tooth types into one class and worked out the side from geometry afterwards. U-Mamba2 keeps separate classes and makes the augmentation respect them.

**Post-processing with anatomical sizes.** After prediction, any connected blob smaller than a per-class threshold is removed. Each threshold is the **0.5th percentile of that class's connected-component volumes in the training labels**, so it's set from the anatomy rather than from a particular model's mistakes.

## See it on the scan

Drag to rotate, scroll to zoom, and switch between the views. Hover over a structure to see its name.

<div class="cv" data-src="{{ '/assets/data/toothfairy3_F001.bin' | relative_url }}"></div>
<script type="module" src="{{ '/assets/js/cbct_viewer.js' | relative_url }}"></script>

*Everything shown comes from the dataset's ground-truth labels for one scan (ToothFairy3F_001), not from model predictions. Each structure is a surface extracted from its label with marching cubes and simplified for the web.*

## Results

For its validation results, the paper trains on 90% of ToothFairy3 and holds back the other 10% for testing. The split is **stratified by data source**: each of the three sets (A, B and C above, which differ in field of view and scanner) is split 90/10 on its own. That way the validation scans have the same mix of scanners and fields of view as the training scans. A plain random split could, by chance, put too many of the new scanner's scans on one side and skew the results.

The paper's Task 1 validation results (higher Dice is better, lower HD95 is better):

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

What struck me most was how much of this winning solution is knowledge about teeth rather than new networks. The architecture change is a single layer. Around it, each trick encodes one fact about the anatomy:

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
