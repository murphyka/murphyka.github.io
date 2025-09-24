---
layout: page
title: projects
permalink: /projects/
description: Here are some ongoing research directions of the group.  Openings for curious students at all levels!
nav: true
nav_order: 2
display_categories: 
horizontal: true
---

### How do transformers process information?
<figure style="float:right; margin-left: 20px; margin-bottom: 10px; text-align:center; width:350px;">
  <img src="/assets/img/vit.png" alt="ViT schematic" width="350">
  <figcaption style="font-size:0.9em; color:#555;">
    Schematic from the original Vision Transformer paper, <a href="https://arxiv.org/abs/2010.11929">Dosovitskiy et al. (2021)</a>.
  </figcaption>
</figure>

Transformers aggregate information from many pieces -- language fragments, image patches, temporal intervals, etc. -- into a global representation of the whole.
In this project, we restrict the flow of information between pieces as they are processed by a transformer, and gain an entirely new perspective on the way transformers build up from local to global.

This project builds upon several recent publications that use the distributed information bottleneck to restrict and monitor information in composite systems:
- [*Surveying the Space of Descriptions of a Composite System with Machine Learning,*](https://journals.aps.org/prl/abstract/10.1103/gxrh-2xsv) Physical Review Letters 2025.
- [*Information decomposition in complex systems via machine learning,*](https://www.pnas.org/doi/10.1073/pnas.2312988121) PNAS 2024.
- [*Interpretability with full complexity by constraining feature information,*](https://openreview.net/forum?id=R_OL5mLhsv) ICLR 2023.


---

### Probabilistic representation learning: engineering how information is stored in latent spaces
<img src="/assets/img/rand_net.png" alt="abstract neural network visualization" width="250" style="float:left; margin-right: 20px; margin-bottom: 10px;">

Deep learning layers transformations on top of one another, turning data into representations that twist and contort into something useful (hopefully).
We'd love to be able to measure how similar the representations of two networks are, just as we'd love to have more control over the nature of representations.
It turns out that both become easier if you force representations to be probability distributions, and view latent spaces as communication channels. 

**Research highlights:**
- [*Comparing the information content of probabilistic representation spaces,*](https://openreview.net/forum?id=adhsMqURI1) TMLR 2025.

---

### Information games
<figure style="float:right; margin-left: 20px; margin-bottom: 10px; text-align:center; width:200px;">
  <img src="/assets/img/pawn_chess.png" alt="artist rendition of information games" width="200">
</figure>

What happens in multi-agent scenarios when information is not a means to some other end, but rather the ultimate objective itself?
Do effective strategies for deceit and for efficient sensing arise naturally?
Using tools we've developed to characterize the nature of distributed information, we are studying how agents acquire information and deceive their opponents.

---
