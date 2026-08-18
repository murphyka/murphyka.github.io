---
layout: about
title: About
permalink: /
subtitle: <em>Information-processing empiricism</em>

profile:
  align: right
  image: prof_pic.jpg
  image_circular: true # crops the image to make it circular
  position: inline

selected_papers: false # includes a list of papers marked as "selected={true}"
social: false # includes social icons at the bottom of the page

announcements:
  enabled: false # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 5 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: false
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

Hello and welcome!  

**We are a research group that designs and studies information processing systems using tools from information theory.**
We study representation learning in deep neural networks, and build algorithms that distill high-dimensional data into reduced descriptions for the express purpose of interpretability.

**If you're curious about how to design and understand AI systems, consider joining!** 
[Please reach out](mailto:kieran.murphy@njit.edu).

<br>
<hr>
<br>

Visualization can be a powerful route to building intuition around how complex systems work.  Below is a visualization of a randomly initialized neural network that warps two-dimensional space.  The input starts as a square and then what you're seeing is the square after passing through the network.  Try varying the number of layers (64 units each) and the activation function!

<br>

<div style="text-align: center">
<canvas id="canvas" width="600" height="600" style="text-align: center"></canvas>

<br>
<button onclick="draw()">Generate</button>
<br>
Number of layers: <input type="text" id="textInput" value="2" style="width: 15px;border:none">
<input type="range" min="1" max="8" value="2" class="slider" id="num_layers_slider" onchange="updateTextInput(this.value);" style="margin-left:2em"> 
<form>Nonlinearity: 
    <input type="radio" name="act_fn_btns" value="tanh" id="tanh_btn" checked=true> tanh
    <input type="radio" name="act_fn_btns" value="relu" id="relu_btn"> relu
    <input type="radio" name="act_fn_btns" value="hardSigmoid" id="hardSigmoid_btn"> hard_sigmoid
    <input type="radio" name="act_fn_btns" value="elu" id="elu_btn"> elu
    <input type="radio" name="act_fn_btns" value="softsign" id="softsign_btn"> softsign
</form>
</div>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
<script src="/assets/js/km/random_network_visualizer.js"></script>
<script>
  function updateTextInput(val) {
    document.getElementById('textInput').value = val;
  }
  function draw() {
    var actFn = 'tanh';
    var checked = document.querySelector('input[name="act_fn_btns"]:checked');
    if (checked) actFn = checked.value;
    drawNN(document.getElementById('canvas'), {
      size: 450,
      numLayers: parseInt(document.getElementById('num_layers_slider').value),
      actFn: actFn,
    });
  }
  document.addEventListener('themeChanged', draw);
  document.addEventListener('DOMContentLoaded', draw);
</script>

