// Draws a random neural network field visualization onto a canvas element.
// canvasEl: HTMLCanvasElement
// options.size: pixel dimensions (square)
// options.numLayers: number of hidden layers
// options.actFn: TF.js activation name
// options.dots: if true, render points instead of lines
function drawNN(canvasEl, { size = 120, numLayers = 2, actFn = 'tanh', dots = false } = {}) {
  if (typeof tf === 'undefined') return;

  const N = size >= 300 ? 120 : 60;
  const lw = size >= 300 ? 0.5 : 0.8;
  const num_units = 64;
  const k_std = 0.5;
  const b_std = 0.5;

  tf.tidy(() => {
    const d = 1.0;
    const x = tf.linspace(-d, d, N);
    const y = tf.linspace(-d, d, N);
    let xx = tf.matMul(tf.ones([N, 1]), x.reshape([1, N]));
    let yy = tf.matMul(y.reshape([N, 1]), tf.ones([1, N]));
    xx = tf.reshape(xx, [-1]);
    yy = tf.reshape(yy, [-1]);

    const k_init = tf.initializers.randomNormal(0.);
    const b_init = tf.initializers.randomNormal(0.);
    k_init.stddev = k_std;
    b_init.stddev = b_std;

    const model = tf.sequential();
    model.add(tf.layers.dense({
      units: num_units, inputShape: [2], useBias: true,
      activation: actFn, kernelInitializer: k_init, biasInitializer: b_init,
    }));
    for (let i = 1; i < numLayers; i++) {
      model.add(tf.layers.dense({
        units: num_units, inputShape: [num_units], useBias: true,
        activation: actFn, kernelInitializer: k_init, biasInitializer: b_init,
      }));
    }
    model.add(tf.layers.dense({ units: 2 }));

    canvasEl.width = size;
    canvasEl.height = size;

    let v_out = model.apply(tf.stack([xx, yy], -1));
    v_out = tf.div(tf.sub(v_out, v_out.min(0)), tf.sub(v_out.max(0), v_out.min(0)));
    v_out = tf.reshape(v_out, [N, N, 2]);

    if (canvasEl.getContext) {
      const ctx = canvasEl.getContext('2d');
      ctx.lineWidth = lw;
      ctx.clearRect(0, 0, size, size);
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      ctx.strokeStyle = isDark ? '#ffffff' : '#000000';

      v_out.array().then(arr => {
        if (dots) {
          const r = size / 180;
          ctx.fillStyle = ctx.strokeStyle;
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              ctx.beginPath();
              ctx.arc(size * arr[i][j][0], size * arr[i][j][1], r, 0, 2 * Math.PI);
              ctx.fill();
            }
          }
        } else {
          for (let i = 0; i < N; i++) {
            const path = new Path2D();
            path.moveTo(size * arr[i][0][0], size * arr[i][0][1]);
            for (let j = 0; j < N; j++) {
              path.lineTo(size * arr[i][j][0], size * arr[i][j][1]);
            }
            ctx.stroke(path);
          }
        }
      });
    }
  });
}
