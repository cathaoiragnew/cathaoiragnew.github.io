---
layout: page
title: 3D Teeth Segmentation
subtitle: How ToothGroupNet, the winner of the 3DTeethSeg'22 challenge, labels every tooth in an intra-oral scan
share-img: /assets/img/teeth_seg_hero.png
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

<img src="{{ '/assets/img/teeth_seg_hero.png' | relative_url }}"
     alt="An upper-jaw intra-oral scan shown twice: plain on the left, and on the right with each tooth coloured by its ground-truth tooth type"
     style="width:100%; max-width:900px;">

*One upper-jaw scan from Teeth3DS+. Left: the raw mesh. Right: the dataset's ground-truth labels, coloured by tooth type.*

Out of curiosity, I wanted to explore a new domain in healthcare tech, so I picked up a public benchmark of 3D dental scans and the solution that won its challenge. The vocabulary was new (gingiva, occlusal surfaces, FDI tooth numbers), but the fundamentals don't change from domain to domain:

**input → feature vectors → operations → outputs**

The winning solution also relies on a trick I keep running into, including in my own research: **get a coarse answer first, then refine it where it matters.** This page walks through the architecture, then comes back to that coarse-to-fine idea.

## The benchmark: Teeth3DS+

[Teeth3DS+](https://crns-smartvision.github.io/teeth3ds/){:target="_blank"} contains **1,800 intra-oral 3D scans from 900 patients**. The upper and lower jaws are scanned separately. Each scan is a triangle mesh (`.obj`) with a JSON file giving **a tooth label and an instance ID for every vertex**. The dataset was used for the [3DTeethSeg'22 challenge](https://arxiv.org/abs/2305.18277){:target="_blank"} at MICCAI 2022.

## Same recipe, new domain

Set against a standard image segmentation model, the structure is the same:

| Step | Image model | ToothGroupNet |
|---|---|---|
| **Input** | Pixels, 3 numbers each (RGB) | 24,000 points sampled from the mesh, 6 numbers each (xyz + surface normal) |
| **Feature vectors** | CNN or ViT backbone → feature map | Point Transformer backbone → a feature vector per point |
| **Operations** | Heads for class, box and mask | Heads for tooth class and offset to the tooth centre, then clustering, cropping and resampling |
| **Output** | A mask and label per object | A tooth number (FDI) and tooth instance for every vertex |

The one real difference is that **a point cloud has no grid**. In an image, the neighbours of a pixel are simply the pixels next to it. In a point cloud, neighbours have to be computed (the *k* nearest points), and the network has to cope with points that are unevenly spaced. A lot of what makes point-cloud networks look unfamiliar comes down to that.

## ToothGroupNet architecture

ToothGroupNet ([code on GitHub](https://github.com/limhoyeon/ToothGroupNetwork){:target="_blank"}) works in two stages:

<a href="{{ '/assets/img/teeth_seg_pipeline.svg' | relative_url }}" target="_blank"><img src="{{ '/assets/img/teeth_seg_pipeline.svg' | relative_url }}"
     alt="ToothGroupNet pipeline: stage 1 samples 24,000 points with farthest point sampling and labels them with a Point Grouping Module and a Tooth Cropping Module; stage 2 finds boundaries from the stage-1 labels, resamples them densely and labels them with a second network; the labels are merged onto every mesh vertex"
     style="width:100%; max-width:760px;"></a>

*The two-stage pipeline, as described in the challenge paper and implemented in the released code. Click the diagram to open it full size.*

### The backbone: Point Transformer

Both stages are built on the [Point Transformer](https://arxiv.org/abs/2012.09164){:target="_blank"} (Zhao et al., 2021). This is self-attention applied locally: each point attends to its *k* nearest neighbours, and the attention also takes their relative positions into account. The network is shaped like a U-Net. The encoder repeatedly downsamples the cloud while widening the features, and the decoder upsamples back so that **every input point ends up with its own feature vector**. In the released stage-1 configuration there are five levels, keeping a quarter of the points at each downsampling step, with feature widths growing from 32 to 512.

To keep features sharp at the edges of teeth, training adds [contrastive boundary learning](https://arxiv.org/abs/2203.05272){:target="_blank"} (Tang et al., 2022). Near a boundary, points with the same label are pulled towards similar features and points with different labels are pushed apart.

### Stage 1: coarse, with farthest point sampling

The mesh is first reduced to **24,000 points by farthest point sampling (FPS)**. FPS starts from one point and repeatedly adds whichever point is farthest from everything chosen so far, so the samples spread evenly over the surface. These points go into the **Tooth Group Network**, which has two parts.

**Point Grouping Module.** This works much like [PointGroup](https://arxiv.org/abs/2004.01658){:target="_blank"} (Jiang et al., 2020). Two heads sit on the backbone features:

- a **classification head** predicts each point's tooth type;
- a **regression head** predicts a 3D **offset from the point to the centre of its tooth**.

Adding each point's offset to its position pulls all the points of one tooth towards the same spot. Gingiva points are dropped, and **DBSCAN** clusters what remains into one group per tooth. The paper notes this works well because each tooth is a compact, roughly cylindrical shape that is easy to group. In the code, the offset head is trained so that shifted points land on their tooth's centre, point towards it, and end up much closer to one tooth centre than to any other.

**Tooth Cropping Module.** For each predicted tooth centre, the 3,072 nearest points are cropped and passed to a second Point Transformer. This one makes a single binary call: tooth or gingiva. Its mask has the final say on which points count as tooth, which cleans up the grouping result. This is the **proposal → crop → mask** pattern from two-stage detectors, in 3D: find the object roughly, then look closely at a crop around it.

A detail that only shows up in the code: **the stage-1 classifier mostly doesn't separate left from right.** Apart from the two central incisors, each tooth type is one class for both sides (the left and right canines share a class, for example). The side is assigned afterwards from geometry, relative to the midline between the central incisors. Left and right teeth of the same type are close to mirror images, so my read is that this lets the network focus on tooth type and leaves the side to geometry.

### Stage 2: refine, with boundary aware point sampling

Stage 1 labels the 24,000 sampled points, but the output has to cover *every* mesh vertex. Each vertex takes the label of its nearest sampled point, and that is where the edges suffer. The challenge paper explains that scanner meshes are densest near tooth boundaries. FPS spreads its points evenly, so boundary regions, where the vertices are dense and the labels change, get relatively few sampled points.

On the sample scan, using the ground-truth labels, **16% of the mesh's vertices are boundary vertices, but only 12% of the FPS points land on them.**

**Boundary Aware Point Sampling** fixes this:

1. Use the stage-1 labels to find the boundaries. In the code, a vertex counts as a boundary vertex if fewer than 70% of its 40 nearest sampled points share its label.
2. Draw a new set of 24,000 points: up to **20,000 boundary points**, with the rest filled in by FPS.
3. Run a **second Tooth Group Network** on this sample. It only decides which tooth (or gingiva) each point belongs to, grouping the points into the number of teeth stage 1 found. Its tooth-type predictions aren't used.

On the sample scan, **82% of the new sample sits on boundaries.**

<img src="{{ '/assets/img/teeth_seg_sampling.png' | relative_url }}"
     alt="The front teeth of the scan sampled two ways. Left, farthest point sampling: boundary points form thin dotted outlines. Right, boundary-aware sampling: boundary points form thick, dense outlines around every tooth"
     style="width:100%; max-width:900px;">

*The same front teeth, sampled both ways. Bright points lie on a tooth–tooth or tooth–gingiva boundary. Boundaries here come from the ground-truth labels, whereas ToothGroupNet finds them from its stage-1 predictions.*

Finally, the two are merged. Only stage 2's boundary points are kept, and each of its clusters is matched to a stage-1 tooth by a nearest-neighbour majority vote. Every mesh vertex then takes the label of its nearest sampled point from either set. Each tooth's number is a majority vote of the **stage-1** class predictions inside it. So stage 2 refines where the edges are, not which tooth is which.

How the two stages are trained is worth noting. Stage 1 is trained first. Stage 2's training loads the finished stage-1 model and keeps it **frozen**, using it only to decide where to sample. The stage-2 network is a **separate, smaller model** (two levels, feature widths 16 and 32) trained on those boundary-heavy samples. So the refinement doesn't come from fine-tuning the same weights. It comes from a second, specialist model whose input is mostly the hard regions.

## See it on the scan

Drag to rotate, scroll to zoom, and switch between the steps above. Hover over a point to see which tooth it belongs to.

<div class="tv" data-src="{{ '/assets/data/teeth3ds_01F4JV8X_upper.bin' | relative_url }}"></div>
<script type="module" src="{{ '/assets/js/teeth_viewer.js' | relative_url }}"></script>

*All colours come from the dataset's ground-truth labels, not from model predictions. "Shift to centres" uses the true tooth centres, which is what a perfect offset head would produce. "Boundary-aware sample" uses the true labels in place of stage-1 predictions. The sampling follows the settings in ToothGroupNet's [inference pipeline](https://github.com/limhoyeon/ToothGroupNetwork/blob/main/inference_pipelines/inference_pipeline_tgn.py){:target="_blank"}: 24,000 FPS points, a boundary test over 40 neighbours at 70%, and up to 20,000 boundary points.*

## Coarse → fine is everywhere

What struck me most was how familiar the overall shape felt. The same refinement pattern keeps turning up across tasks:

- **Two-stage object detectors** ([Faster R-CNN](https://arxiv.org/abs/1506.01497){:target="_blank"}): a region proposal network suggests rough boxes, then a second head crops features for each one, classifies it and refines the box.
- **[D-FINE](https://arxiv.org/abs/2410.13842){:target="_blank"}**: a DETR-style detector that treats box regression as iterative refinement. Each decoder layer refines a probability distribution over the box edges from the previous layer, and the most refined predictions are distilled back into the earlier layers.
- **ToothGroupNet** does it twice. The Tooth Cropping Module refines the Point Grouping Module's result (locate, then decide the exact extent), and stage 2 refines stage 1 (label everything, then resample and relabel the boundaries).
- **My own paper**, [Pretraining instance segmentation models with bounding box annotations](https://www.sciencedirect.com/science/article/pii/S2667305324001285){:target="_blank"}, applies the same idea to *labels*. Ground-truth bounding boxes are turned into coarse polygon masks ("weak annotations"). Instance segmentation models (SOLOv2, Mask R-CNN and Mask2Former) are pretrained on those, then fine-tuned on finely annotated masks. On COCO, the best model (Mask2Former with a Swin-L backbone) reached **97.5%, 100.4% and 101.3% of fully supervised performance** using just **1%, 5% and 10%** of the instance segmentation annotations.

This work reminded me of that paper. Each of these applies coarse → fine to something different: predictions across layers (D-FINE), image regions (Faster R-CNN), where to sample (ToothGroupNet) and annotations (my paper). The logic is the same, though. A cheap, approximate first pass gets you most of the way, and the expensive part (compute, model capacity or annotation time) is saved for the refinement.

## References

- Ben-Hamadou et al., *Teeth3DS+: An Extended Benchmark for Intraoral 3D Scans Analysis*. [arXiv:2210.06094](https://arxiv.org/abs/2210.06094){:target="_blank"} · [dataset page](https://crns-smartvision.github.io/teeth3ds/){:target="_blank"}
- Ben-Hamadou et al., *3DTeethSeg'22: 3D Teeth Scan Segmentation and Labeling Challenge*. [arXiv:2305.18277](https://arxiv.org/abs/2305.18277){:target="_blank"}
- Lim & Kim (team CGIP), *ToothGroupNetwork*. [github.com/limhoyeon/ToothGroupNetwork](https://github.com/limhoyeon/ToothGroupNetwork){:target="_blank"}
- Zhao et al., *Point Transformer*, ICCV 2021. [arXiv:2012.09164](https://arxiv.org/abs/2012.09164){:target="_blank"}
- Jiang et al., *PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation*, CVPR 2020. [arXiv:2004.01658](https://arxiv.org/abs/2004.01658){:target="_blank"}
- Tang et al., *Contrastive Boundary Learning for Point Cloud Segmentation*, CVPR 2022. [arXiv:2203.05272](https://arxiv.org/abs/2203.05272){:target="_blank"}
- Ren et al., *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497){:target="_blank"}
- Peng et al., *D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement*. [arXiv:2410.13842](https://arxiv.org/abs/2410.13842){:target="_blank"}
- Agnew et al., *Pretraining instance segmentation models with bounding box annotations*, Intelligent Systems with Applications 24 (2024) 200454. [doi:10.1016/j.iswa.2024.200454](https://doi.org/10.1016/j.iswa.2024.200454){:target="_blank"}

<small>The scan shown is `01F4JV8X_upper` from Teeth3DS+, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/){:target="_blank"}. The images and point data on this page are derived from it and shared under the same licence.</small>
