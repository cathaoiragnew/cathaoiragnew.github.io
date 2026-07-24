---
layout: page
title: Paddle Scout
---

*Natural-language search over a padel match - reconstructed, described, and indexed
entirely from ground-truth annotations.*

<video
    autoplay
    loop
    muted
    playsinline
    style="width: 100%; height: auto;">
    <source src="{{ '/assets/img/preview_app.webm' | relative_url }}" type="video/webm">
</video>

I wanted to be able to search a padel match the way you'd describe it to a friend -
*"the long rally that ended in a smash"*, *"the point where she got pulled right out to
the corner"* - and get the actual moment back. This project does that. You type a
description in plain English and it returns the matching rallies, each one replayed as a
reconstruction built entirely from the dataset's ground-truth annotations.

## Pipeline

**1 · The data.** PadelTracker100 provides hand-verified **ground-truth annotations** at
30 fps - ball position, 17-keypoint player **pose**, per-frame **homography**, and
frame-level **shot labels** (serve / forehand / backhand / smash / dropshot). No footage:
the source WPT broadcasts are now private. These are *labels, not pixels* - and nothing
downstream cares where they come from, so swapping them for model **predictions**
(detector + pose + homography + shot classifier) would run the same pipeline on live matches.

**2 · Pixels → court metres.** The **homography** maps any pixel to a real position on the
10 × 20 m court (verified to a few cm). 

<img src="{{ '/assets/img/homography_exp.png' | relative_url }}"
     alt="Homography example"
     style="width:100%; max-width:900px;">

**3 · Rally segmentation.** Collapse per-frame shot labels into discrete shots, then split
into rallies on a serve or a >2 s gap → 60 rallies for the women's final, ~7.6 shots each.

**4 · Structured features.** Exact per-rally stats - shot sequence, duration, pace, court
width covered. Ground truth, no guessing: the reliable half of search. The coverage
heatmap already shows the tell - players cluster at the net and the baseline, with a dead
band across the net.


<img src="{{ '/assets/img/highlights.png' | relative_url }}"
     alt="Highlights example"
     style="width:100%; max-width:900px;">

**5 · Reconstruction.** Redraw each rally from the labels: stick-figure players (coloured
by side of the net), a ball trail, and the court in perspective via the **inverse
homography** → a short clip that plays in the browser. Solves the no-video
problem.

**6 · Captioning (VLM).** **SmolVLM2** (2.2 B, bf16) commentates each reconstruction,
grounded by injecting the ground-truth facts into the prompt. Honest ceiling: a small
model on a schematic gives usable phrases ("powerful smash", "defensive scramble"), not
broadcast commentary. 


<img src="{{ '/assets/img/VLM_exp.png' | relative_url }}"
     alt="VLM Caption example"
     style="width:100%; max-width:900px;">

**7 · Search.** This is the core, so it's worth unpacking how a sentence of English
becomes a ranked list of rallies.

*Text → a vector (embedding).* Computers can't compare meaning directly, so each rally's
description (the VLM caption + its structured facts) is passed through a **text-embedding
model** - `BAAI/bge-base-en-v1.5` - which turns it into a **768-number vector**. Under the
hood the text is split into sub-word tokens, a transformer encoder produces a
context-aware vector per token, those are mean-pooled into one vector, and it's
L2-normalised to unit length. The model was trained (contrastively) so that texts with
*similar meaning* end up pointing in *similar directions* - even with no words in common,
"frantic scramble at the net" lands near "hectic exchange up front". Meaning becomes
geometry.

*Comparing vectors.* Similarity is the **cosine** of the angle between two vectors: `+1`
means same direction (near-identical meaning), `0` unrelated. Because every vector is
unit-length, cosine is just the dot product - cheap to compute.

*Queries vs. documents.* BGE is trained for *asymmetric* search, so the rally descriptions
are embedded as-is, but a query is first given a short instruction prefix
(*"Represent this sentence for searching relevant passages:"*) before embedding. That
nudges the query's vector toward the description that answers it rather than one that
merely repeats its words.

*Storing and finding them.* Each rally becomes a row in **LanceDB** (a columnar, on-disk
vector store): its 768-d vector plus every metadata field (`n_shots`, `last_shot`,
`court_width_m`, `clip_path`, …). At query time I embed the query and ask LanceDB for the
rows whose vectors have the highest cosine similarity. With 60 rallies that's an exact
brute-force scan - score all 60, sort, take the top-k. At a million rallies you'd instead
build an **approximate-nearest-neighbour index** (IVF / HNSW) that finds the closest
vectors in sub-linear time for a small accuracy cost; LanceDB adds one with a single call,
so the same code scales.

*Hybrid: exact filters + fuzzy ranking.* A query is parsed two ways at once. Hard
constraints ("more than 8 shots", "ends with a smash") become a SQL-style `WHERE` clause
that **pre-filters** the rows; the semantic vector search then *ranks* whatever survives.
So the exact conditions are guaranteed while the embedding handles the fuzzy intent.
Counting questions ("how many smashes?") skip vectors entirely - they're a direct
aggregation over a metadata column.

**8 · Extras.** Two things fall straight out of having vectors + a spectacle score:
**auto-highlights** (rank every rally by `0.5·length + 0.3·width + 0.2·pace`, normalised)
and **"more like this"** (take a rally's own vector and search for its nearest
neighbours). One click each.

## Where it goes next: an agent

Every capability above is already a function - which is exactly a tool-using **LLM agent's**
toolbox. Wrapped as one, the search box becomes a padel analyst that *plans* multi-step
questions ("compare the teams' shot selection, with examples") instead of running a single
query.

```text
   "Compare the teams' shot selection, with examples"
                          |
                          v
             +---------------------------+
             |    Padel-analyst agent    |   plans the steps,
             |       (LLM planner)       |   reads results, replies
             +-------------+-------------+
          _______________ | _______________
         |                |                |
         v                v                v
  court_analytics   search_rallies   top_highlights /
     (team)            (query)       similar_rallies
         |________________|________________|
                          v
             +---------------------------+     describe_clip()
             |  rally index: stats +     | <-- VLM re-checks a
             |  vectors + reconstructions|     rally to verify
             +-------------+-------------+
                          v
      Narrated answer  +  evidence reconstruction clips
```

It orchestrates existing capabilities — it doesn't invent new ground truth.


## Scope — a proof of concept

This is a **proof of concept**, not a production system. It exists to show the whole idea
working end-to-end - ground-truth annotations → court geometry → reconstruction → VLM
captions → hybrid search - on a single match of 60 rallies. Every stage is deliberately the
simplest thing that demonstrates the approach. A production version would need:

- **Perception models in front.** Here the annotations are ground truth. A live system
  would run its own ball/player detection, pose estimation, homography calibration and shot
  classification, each adding error   the pipeline would have to tolerate.
- **A stronger captioner.** SmolVLM2-2.2B on a schematic gives rough, occasionally wrong
  descriptions; a larger or fine-tuned VLM - ideally on real footage - would sharpen
  retrieval a lot.
- **Scale and robustness.** Brute-force vector search is fine for 60 rallies but untuned for
  millions (no ANN index, no latency budget); the Gradio app is a single-user demo with no
  auth, monitoring, or hardened error handling.
- **Validated heuristics.** The spectacle score and the serve-fault flag are hand-tuned
  hints, not calibrated against real outcomes.

And some limits are the data's, not the build's: there are no score labels, so "who won the
point" is only inference, and fine trick shots (a between-the-legs) are beyond
both the labels and a small model reading a diagram.

## Stack

Homography geometry (NumPy) · rally segmentation · OpenCV + ffmpeg reconstructions ·
**SmolVLM2** captioning · **BGE** embeddings · **LanceDB** vector store · **Gradio**. Heavy
work runs once offline; the demo serves on CPU.
