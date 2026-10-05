---
layout: page
title: Projects
permalink: /projects/
description: Here are some ongoing research directions of the group.  Openings for curious students at the undergraduate and graduate levels!
nav: true
nav_order: 2
display_categories: 
horizontal: true
---

<style>
.projects .project { 
  display: flow-root; /* contains the float */
  margin-bottom: 2.25rem;
}
.projects figure {
  float: right;
  margin-left: 20px; margin-bottom: 10px;
  text-align: center; width: 350px;
}
.projects img {
  max-width: 100%; height: auto; border-radius: 6px;
}
.projects figcaption {
  font-size: 0.9em; color: #555; margin-top: 6px;
}
@media (max-width: 700px) {
  .projects figure { float: none; width: 100%; margin: 0 0 10px 0; }
}

.projects a:hover {
  text-decoration: underline;
}

</style>

<div class="projects" markdown="1">

## Information flows and mechanistic interpretability

<div class="project" markdown="1">
<figure style="float:right; margin-left: 20px; margin-bottom: 10px; text-align:center; width:250px;" >
  <img src="/assets/img/publication_preview/stoch_norm.png" alt="Stochastic residual stream reads for measurable information flows" width="350" loading="lazy">
  <figcaption style="font-size:0.9em; color:#555;">
    An easy modification to pre-trained transformers: add isotropic noise inside the normalization operation and then fine-tune for rate-limited information processing.
  </figcaption>
</figure>

Transformers aggregate information from many pieces -- language fragments, image patches, temporal intervals, etc. -- into a global representation of the whole.
Most commonly, they operate on point-based representations and no limitation on the amount of information passing through each latent space. 
**What if we want to know what information is processed by each component?** 

We convert specific junctures in the transformer into probabilistic representation spaces and restrict the flow of information.
One point-based processing path from input to output then becomes an information flow, with volume and mistakes and rates contributed by different components.
We're particularly interested in light-touch modifications to pre-trained models -- see our recent stochastic normalization work below -- and then in applying methods from mechanistic interpretability to characterize the flows. 

**Research highlights:**
- [*Tracing distinguishability through transformer processing with stochastic LayerNorm,*](https://arxiv.org//abs/2608.30720) arXiv 2026.
- [*From independent patches to coordinated attention: Controlling information flow in vision transformers,*](https://arxiv.org/abs/2602.04784) arXiv 2026, [ICML 2026 workshop on Mechanistic Interpretability](https://openreview.net/forum?id=nJmy7iULF7).

<div class="project" markdown="1">
<figure style="float:left; margin-right: 20px; margin-bottom: 10px; text-align:center; width:250px;" >
  <img src="/assets/img/publication_preview/schematic_vit_ib.png" alt="Vision transformer schematic with information bottlenecks before writes to the residual stream" width="350" loading="lazy">
  <figcaption style="font-size:0.9em; color:#555;">
    A modified vision transformer where attention heads pass through information bottlenecks before writing to the residual stream.
  </figcaption>
</figure>

</div>

---

## Information decomposition to reveal structure in data

<div class="project" markdown="1">

<img src="/assets/img/publication_preview/reveal_ig.gif" alt="pixel attribution for resnet in swimming koi video" width="250" loading="lazy" style="float:right; margin-right: 20px; margin-bottom: 10px;">

**Research highlights:**
- [*Attribution via Distributional Paths for Information Revelation,*](https://arxiv.org/abs/2606.03885) arXiv 2026.
- [*Where is the information in data?,*](https://kieranamurphy.com/information_explorable/) IEEE VIS 2024 workshop, \href{https://visxai.io/}{``Visualization for AI Explainability (visXAI)''}.
- [*Surveying the Space of Descriptions of a Composite System with Machine Learning,*](https://journals.aps.org/prl/abstract/10.1103/gxrh-2xsv) Physical Review Letters 2025.
- [*Information decomposition in complex systems via machine learning,*](https://www.pnas.org/doi/10.1073/pnas.2312988121) PNAS 2024.
- [*Interpretability with full complexity by constraining feature information,*](https://openreview.net/forum?id=R_OL5mLhsv) ICLR 2023.


</div>

---

## Probabilistic representation learning: engineering how information is stored in latent spaces

<div class="project" markdown="1">

<img src="/assets/img/rand_net.png" alt="abstract neural network visualization" width="250" loading="lazy" style="float:left; margin-right: 20px; margin-bottom: 10px;">

Deep learning layers transformations on top of one another, turning data into representations that twist and contort into something useful (hopefully).
We'd love to be able to measure how similar the representations of two networks are, just as we'd love to have more control over the nature of representations.
It turns out that both become easier if you force representations to be probability distributions, and view latent spaces as communication channels. 

**Research highlights:**
- [*Comparing the information content of probabilistic representation spaces,*](https://openreview.net/forum?id=adhsMqURI1) TMLR 2025.

</div>

</div>