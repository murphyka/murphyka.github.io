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

## How does information flow through transformers?

<div class="project" markdown="1">
<figure style="float:right; margin-left: 20px; margin-bottom: 10px; text-align:center; width:250px;" >
  <img src="/assets/img/publication_preview/stoch_norm.png" alt="Stochastic residual stream reads for measurable information flows" width="350" loading="lazy">
  <figcaption style="font-size:0.9em; color:#555;">
    An easy modification to pre-trained transformers: add isotropic noise inside the normalization operation and then fine-tune for rate-limited information processing.
  </figcaption>
</figure>


Transformers aggregate information from many pieces -- language fragments, image patches, temporal intervals, etc. -- into a global representation of the whole.
Most commonly, they operate on point-based representations and no limitation on the amount of information passing through each latent space. 
**What if we want to know what information is processed by each component at each location?** 

In this project, we convert specific junctures in the transformer into probabilistic representation spaces and restrict the transmission of information.
One point-based processing path from input to output then becomes an information flow, with volume and mistakes and rates contributed by each chokepoint.
We're particularly interested in light-touch modifications to pre-trained models -- see our recent stochastic normalization work below -- and then in applying methods from mechanistic interpretability for a new perspective on previously identified circuitry. 

**Research highlights:**
- [*Tracing distinguishability through transformer processing with stochastic LayerNorm,*](https://arxiv.org//abs/2608.30720) arXiv 2026.
- [*From independent patches to coordinated attention: Controlling information flow in vision transformers,*](https://arxiv.org/abs/2602.04784) arXiv 2026, [ICML 2026 workshop on Mechanistic Interpretability](https://openreview.net/forum?id=nJmy7iULF7).

<figure style="float:left; margin-right: 20px; margin-bottom: 10px; text-align:center; width:250px;" >
  <img src="/assets/img/publication_preview/schematic_vit_ib.png" alt="Vision transformer schematic with information bottlenecks before writes to the residual stream" width="350" loading="lazy">
  <figcaption style="font-size:0.9em; color:#555;">
    A modified vision transformer where attention heads pass through information bottlenecks before writing to the residual stream.
  </figcaption>
</figure>

</div>

---

## What's the structure of information in data?

<div class="project" markdown="1">

<img src="/assets/img/publication_preview/reveal_ig.gif" alt="pixel attribution for resnet in swimming koi video" width="250" loading="lazy" style="float:right; margin-right: 20px; margin-bottom: 10px;">

Data comes packaged in groups of features, generally with rich multivariate structure concerning their shared variation and the information relevant to downstream tasks.
What if, while optimizing a model to process data, we also optimized the selection of information from the features?
With a finite budget, we'd find an intuitive notion of **the most important information** for a task.  
We could also probe the specific way that information is contained within the features -- whether they are columns in a tabular dataset or pixels in an image.
In the gif to the right, our recent [*Reveal-IG*](https://arxiv.org/abs/2606.03885) method turns sensitivity to information revelation into a heatmap of evidence for a ResNet-50 to classify each frame as a goldfish; the method is a hybrid between [Shapley additive explanations (SHAP)](https://arxiv.org/abs/1705.07874) and [Integrated Gradients (IG)](https://arxiv.org/abs/1703.01365), two of the most popular methods in Explainable AI.  

**Research highlights:**
- [*Attribution via Distributional Paths for Information Revelation,*](https://arxiv.org/abs/2606.03885) arXiv 2026.
- [*Surveying the Space of Descriptions of a Composite System with Machine Learning,*](https://journals.aps.org/prl/abstract/10.1103/gxrh-2xsv) Physical Review Letters 2025.
- [*Information decomposition in complex systems via machine learning,*](https://www.pnas.org/doi/10.1073/pnas.2312988121) PNAS 2024.
- [*Where is the information in data?,*](https://kieranamurphy.com/information_explorable/) IEEE VIS 2024 workshop, [Visualization for AI Explainability (visXAI)](https://visxai.io/).
- [*Interpretability with full complexity by constraining feature information,*](https://openreview.net/forum?id=R_OL5mLhsv) ICLR 2023.


</div>

---

## How is information organized in probabilistic representation spaces? 

<div class="project" markdown="1">

<img src="/assets/img/rand_net.png" alt="abstract neural network visualization" width="250" loading="lazy" style="float:left; margin-right: 20px; margin-bottom: 10px;">

We clearly like probabilistic representation spaces in this group.
They can make information transmission finite and measurable, and they provide a notion of representational similarity that's grounded in downstream computation (through the data processing inequality).
In this project, our questions are more fundamental about the organization of information in such spaces.
One curiosity arose from our study about [compressing chaotic dynamical systems' state spaces](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.132.197201): although the neural network was optimized with a soft latent space, the eventual solution often hardened into a compression scheme equivalent to a discrete codebook.  
Hard compression schemes are generally more practical and far more interpretable, while the space of soft schemes is far larger and amenable to optimization.
Can we get the best of both worlds, through a better understanding of the space of compression schemes? 

**Research highlights:**
- [*Comparing the information content of probabilistic representation spaces,*](https://openreview.net/forum?id=adhsMqURI1) TMLR 2025.
- [*Machine-Learning Optimized Measurements of Chaotic Dynamical Systems via the Information Bottleneck*](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.132.197201), Physical Review Letters 2024.

</div>