<div style="text-align: center">
<canvas id="canvas" width="600" height="600"></canvas>

<br>
<button onclick="draw()">Generate</button>
<br>
Number of layers: <input type="text" id="textInput" value="2" style="width: 10px;border:none"></input> <input type="range" min="1" max="8" value="2" class="slider" id="num_layers_slider" onchange="updateTextInput(this.value);"> 
<form>Nonlinearity: 
    <input type="radio" name="act_fn_btns" value="tanh" id="tanh_btn" checked=true> tanh
    <input type="radio" name="act_fn_btns" value="relu" id="relu_btn"> relu
    <input type="radio" name="act_fn_btns" value="hardSigmoid" id="hardSigmoid_btn"> hard_sigmoid
    <input type="radio" name="act_fn_btns" value="elu" id="elu_btn"> elu
    <input type="radio" name="act_fn_btns" value="softsign" id="softsign_btn"> softsign
</form>
</div>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
<script>
function updateTextInput(val) {
          document.getElementById('textInput').value=val; 
        }
</script>
<script type="application/javascript">
  if (screen.width > 900) {
  	N = 120;
    lw = 0.5;
  } else {
    N = 80;
    lw = 0.8;
  }
  d = 1.;
  x = tf.linspace(-d, d, N);
  y = tf.linspace(-d, d, N);
  var xx = tf.matMul( tf.ones  ([N, 1]), x.reshape([1, N]) )
  var yy = tf.matMul( y.reshape([N, 1]), tf.ones  ([1, N]) );
  xx = tf.reshape(xx, [-1]);
  yy = tf.reshape(yy, [-1]);
  const k_std = 0.5;
  const b_std = 0.5;
  const num_units = 64;
  function draw() {tf.tidy(() => {
    spacerBlock = document.getElementById("block-yui_3_17_2_1_1618842604252_4599");
  	const h = spacerBlock.clientWidth-10;
    var canvas = document.getElementById("canvas");
    canvas.width = h;
    canvas.height = h;
    var num_layers = document.getElementById("num_layers_slider").value;
    if (document.getElementById('tanh_btn').checked) {
      var act_fn = 'tanh';
    }else if (document.getElementById('relu_btn').checked) {
      var act_fn = 'relu';
    }else if (document.getElementById('hardSigmoid_btn').checked) {
      var act_fn = 'hardSigmoid';
    }else if (document.getElementById('elu_btn').checked) {
      var act_fn = 'elu';
    }else if (document.getElementById('softsign_btn').checked) {
      var act_fn = 'softsign';
    }
    const model = tf.sequential();
    k_init = tf.initializers.randomNormal(0.);
    b_init = tf.initializers.randomNormal(0.);
    k_init.stddev = k_std;
    b_init.stddev = b_std;

    model.add(tf.layers.dense({
      units: num_units, 
      inputShape: [2], 
      useBias: true, 
      activation: act_fn, 
      kernelInitializer: k_init, 
      biasInitializer: b_init,
    }))
    for (var i = 1; i < num_layers; i++) {
      layer = tf.layers.dense({
        units: num_units, 
        inputShape: [num_units], 
        useBias: true, 
        activation: act_fn,
        kernelInitializer: k_init, 
        biasInitializer: b_init,
      });
      model.add(layer);
    }
    model.add(tf.layers.dense({units: 2}));

    var v_out = model.apply(tf.stack([xx, yy], -1));
    v_out = tf.div(tf.sub(v_out, v_out.min(0)), tf.sub(v_out.max(0), v_out.min(0)));
    v_out = tf.reshape(v_out, [N, N, 2]);
    if (canvas.getContext) {
      var ctx = canvas.getContext('2d');
      ctx.lineWidth = lw;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      v_out.array().then(arr => {
        for (var i = 0; i < N; i++) {
          path = new Path2D();
          path.moveTo(h*arr[i][0][0], h*arr[i][0][1]);
          for (var j = 0; j < N; j++) {
            path.lineTo(h*arr[i][j][0], h*arr[i][j][1]);
          }
          ctx.stroke(path);
        }
      } )
  } 
  } )
}
</script>
<body onload="draw()">