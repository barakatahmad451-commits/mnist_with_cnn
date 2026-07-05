// model.js — pure-JS forward pass of the extracted Keras CNN.
// No TensorFlow.js dependency: this reimplements conv2d / maxpool / dense
// directly against the raw weight arrays pulled from mnist_model.h5.

const Model = (() => {
  let W = null; // decoded weight tensors, keyed by name

  function base64ToFloat32Array(b64) {
    const binary = atob(b64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  function load() {
    const flat = base64ToFloat32Array(WEIGHTS_B64);
    W = {};
    for (const [name, info] of Object.entries(WEIGHTS_META)) {
      W[name] = {
        shape: info.shape,
        data: flat.subarray(info.offset, info.offset + info.length),
      };
    }
    return W;
  }

  // input: Float32Array length 28*28, values 0..1 (H*W row-major)
  // returns { probs: Float32Array(10) }
  function predict(input28x28) {
    // ---- reshape to [28][28][1] (already flat in H*W*C order) ----
    const H0 = 28, W0 = 28;
    const img = input28x28;

    // ---- conv2d #1: 32 filters, 3x3, valid, relu ----
    const c1 = conv2d(img, H0, W0, 1, W.conv1_k.data, W.conv1_b.data, 3, 32);
    // c1: [26][26][32]

    // ---- maxpool 2x2 ----
    const p1 = maxpool2(c1.data, c1.h, c1.w, c1.c);
    // p1: [13][13][32]

    // ---- conv2d #2: 64 filters, 3x3, valid, relu ----
    const c2 = conv2d(p1.data, p1.h, p1.w, p1.c, W.conv2_k.data, W.conv2_b.data, 3, 64);
    // c2: [11][11][64]

    // ---- maxpool 2x2 ----
    const p2 = maxpool2(c2.data, c2.h, c2.w, c2.c);
    // p2: [5][5][64] -> flatten = 1600

    // ---- flatten (H,W,C row-major, matching Keras channels_last Flatten) ----
    const flat = p2.data; // already stored in H*W*C order

    // ---- dense 1: 1600 -> 128, relu ----
    const d1 = dense(flat, W.dense1_k.data, W.dense1_b.data, 1600, 128, true);

    // ---- dense 2: 128 -> 10, softmax ----
    const logits = dense(d1, W.dense2_k.data, W.dense2_b.data, 128, 10, false);
    const probs = softmax(logits);

    return { probs };
  }

  // input flattened as [h][w][cin] row-major (index = (y*w+x)*cin+ci)
  // kernel flattened as [kh][kw][cin][cout] row-major (Keras Conv2D kernel layout)
  function conv2d(input, h, w, cin, kernel, bias, k, cout) {
    const oh = h - k + 1;
    const ow = w - k + 1;
    const out = new Float32Array(oh * ow * cout);

    for (let oy = 0; oy < oh; oy++) {
      for (let ox = 0; ox < ow; ox++) {
        for (let oc = 0; oc < cout; oc++) {
          let sum = bias[oc];
          for (let ky = 0; ky < k; ky++) {
            const iy = oy + ky;
            for (let kx = 0; kx < k; kx++) {
              const ix = ox + kx;
              const inBase = (iy * w + ix) * cin;
              const kBase = ((ky * k + kx) * cin) * cout;
              for (let ic = 0; ic < cin; ic++) {
                sum += input[inBase + ic] * kernel[kBase + ic * cout + oc];
              }
            }
          }
          // relu
          out[(oy * ow + ox) * cout + oc] = sum > 0 ? sum : 0;
        }
      }
    }
    return { data: out, h: oh, w: ow, c: cout };
  }

  function maxpool2(input, h, w, c) {
    const oh = Math.floor(h / 2);
    const ow = Math.floor(w / 2);
    const out = new Float32Array(oh * ow * c);
    for (let oy = 0; oy < oh; oy++) {
      for (let ox = 0; ox < ow; ox++) {
        for (let ch = 0; ch < c; ch++) {
          let m = -Infinity;
          for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
              const iy = oy * 2 + dy;
              const ix = ox * 2 + dx;
              const v = input[(iy * w + ix) * c + ch];
              if (v > m) m = v;
            }
          }
          out[(oy * ow + ox) * c + ch] = m;
        }
      }
    }
    return { data: out, h: oh, w: ow, c };
  }

  // kernel flattened [nIn][nOut] row-major (Keras Dense kernel layout)
  function dense(input, kernel, bias, nIn, nOut, relu) {
    const out = new Float32Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let sum = bias[o];
      for (let i = 0; i < nIn; i++) {
        sum += input[i] * kernel[i * nOut + o];
      }
      out[o] = relu ? (sum > 0 ? sum : 0) : sum;
    }
    return out;
  }

  function softmax(logits) {
    let max = -Infinity;
    for (const v of logits) if (v > max) max = v;
    let sum = 0;
    const exps = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      exps[i] = Math.exp(logits[i] - max);
      sum += exps[i];
    }
    for (let i = 0; i < exps.length; i++) exps[i] /= sum;
    return exps;
  }

  return { load, predict };
})();
