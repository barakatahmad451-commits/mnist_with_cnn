(() => {
  const drawCanvas = document.getElementById('drawCanvas');
  const dctx = drawCanvas.getContext('2d', { willReadFrequently: true });
  const gridPreview = document.getElementById('gridPreview');
  const gctx = gridPreview.getContext('2d');
  const scanlineEl = document.getElementById('scanlineEl');
  const readBtn = document.getElementById('readBtn');
  const clearBtn = document.getElementById('clearBtn');
  const predictedDigitEl = document.getElementById('predictedDigit');
  const confidenceValueEl = document.getElementById('confidenceValue');
  const barsEl = document.getElementById('bars');
  const inferTimeEl = document.getElementById('inferTime');
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');

  const SIZE = 280; // on-screen canvas size

  // ---------- init drawing surface ----------
  function resetCanvas() {
    dctx.fillStyle = '#000';
    dctx.fillRect(0, 0, SIZE, SIZE);
    dctx.lineCap = 'round';
    dctx.lineJoin = 'round';
    dctx.strokeStyle = '#fff';
    dctx.lineWidth = 18;
  }
  resetCanvas();

  // ---------- build bar chart rows ----------
  const barFills = [];
  const barPcts = [];
  for (let d = 0; d < 10; d++) {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <span class="bar-digit mono">${d}</span>
      <div class="bar-track"><div class="bar-fill" id="barFill${d}"></div></div>
      <span class="bar-pct mono" id="barPct${d}">0%</span>
    `;
    barsEl.appendChild(row);
    barFills.push(document.getElementById(`barFill${d}`));
    barPcts.push(document.getElementById(`barPct${d}`));
  }

  // ---------- drawing interaction ----------
  let drawing = false;
  let hasInk = false;
  let lastX = 0, lastY = 0;
  let debounceTimer = null;

  function pos(e) {
    const rect = drawCanvas.getBoundingClientRect();
    const scaleX = SIZE / rect.width;
    const scaleY = SIZE / rect.height;
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
    return [cx * scaleX, cy * scaleY];
  }

  function startDraw(e) {
    e.preventDefault();
    drawing = true;
    hasInk = true;
    [lastX, lastY] = pos(e);
    dctx.beginPath();
    dctx.moveTo(lastX, lastY);
    dctx.lineTo(lastX + 0.01, lastY + 0.01);
    dctx.stroke();
  }

  function moveDraw(e) {
    if (!drawing) return;
    e.preventDefault();
    const [x, y] = pos(e);
    dctx.beginPath();
    dctx.moveTo(lastX, lastY);
    dctx.lineTo(x, y);
    dctx.stroke();
    [lastX, lastY] = [x, y];
  }

  function endDraw(e) {
    if (!drawing) return;
    drawing = false;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      if (hasInk) runInference();
    }, 250);
  }

  drawCanvas.addEventListener('mousedown', startDraw);
  drawCanvas.addEventListener('mousemove', moveDraw);
  window.addEventListener('mouseup', endDraw);
  drawCanvas.addEventListener('touchstart', startDraw, { passive: false });
  drawCanvas.addEventListener('touchmove', moveDraw, { passive: false });
  drawCanvas.addEventListener('touchend', endDraw);

  clearBtn.addEventListener('click', () => {
    resetCanvas();
    hasInk = false;
    predictedDigitEl.textContent = '–';
    predictedDigitEl.classList.add('empty');
    confidenceValueEl.textContent = '—';
    inferTimeEl.textContent = '—';
    barFills.forEach(f => { f.style.width = '0%'; f.classList.remove('top'); });
    barPcts.forEach(p => p.textContent = '0%');
    gctx.fillStyle = '#000';
    gctx.fillRect(0, 0, 28, 28);
  });

  readBtn.addEventListener('click', () => {
    if (hasInk) runInference();
  });

  // ---------- preprocessing: crop to bounding box, center in 28x28 ----------
  function preprocess() {
    const imgData = dctx.getImageData(0, 0, SIZE, SIZE).data;
    // brightness channel (drawing is white on black => use red channel)
    let minX = SIZE, minY = SIZE, maxX = -1, maxY = -1;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const v = imgData[(y * SIZE + x) * 4]; // R channel
        if (v > 20) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) return null; // nothing drawn

    const boxW = maxX - minX + 1;
    const boxH = maxY - minY + 1;

    // crop into a temp canvas
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = boxW;
    cropCanvas.height = boxH;
    const cctx = cropCanvas.getContext('2d');
    cctx.drawImage(drawCanvas, minX, minY, boxW, boxH, 0, 0, boxW, boxH);

    // scale so the longer side becomes 20px (standard MNIST-style margin)
    const target = 20;
    const scale = target / Math.max(boxW, boxH);
    const newW = Math.max(1, Math.round(boxW * scale));
    const newH = Math.max(1, Math.round(boxH * scale));

    const scaledCanvas = document.createElement('canvas');
    scaledCanvas.width = newW;
    scaledCanvas.height = newH;
    const sctx = scaledCanvas.getContext('2d');
    sctx.imageSmoothingEnabled = true;
    sctx.drawImage(cropCanvas, 0, 0, newW, newH);

    // paste centered into 28x28
    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = 28;
    finalCanvas.height = 28;
    const fctx = finalCanvas.getContext('2d');
    fctx.fillStyle = '#000';
    fctx.fillRect(0, 0, 28, 28);
    const offX = Math.floor((28 - newW) / 2);
    const offY = Math.floor((28 - newH) / 2);
    fctx.drawImage(scaledCanvas, offX, offY);

    const finalData = fctx.getImageData(0, 0, 28, 28).data;
    const arr = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      arr[i] = finalData[i * 4] / 255; // R channel, normalized 0..1
    }

    // draw the 28x28 preview (scaled up visually by CSS, canvas stays 28x28)
    gctx.putImageData(fctx.getImageData(0, 0, 28, 28), 0, 0);

    return arr;
  }

  // ---------- inference + UI update ----------
  function runInference() {
    const input = preprocess();
    if (!input) return;

    // trigger the signature scanline sweep
    scanlineEl.classList.remove('sweep');
    void scanlineEl.offsetWidth; // restart animation
    scanlineEl.classList.add('sweep');

    const t0 = performance.now();
    const { probs } = Model.predict(input);
    const t1 = performance.now();

    let best = 0;
    for (let i = 1; i < 10; i++) if (probs[i] > probs[best]) best = i;

    predictedDigitEl.textContent = String(best);
    predictedDigitEl.classList.remove('empty');
    confidenceValueEl.textContent = (probs[best] * 100).toFixed(1) + '%';
    inferTimeEl.textContent = (t1 - t0).toFixed(1) + ' ms';

    for (let d = 0; d < 10; d++) {
      const pct = probs[d] * 100;
      barFills[d].style.width = pct.toFixed(1) + '%';
      barFills[d].classList.toggle('top', d === best);
      barPcts[d].textContent = pct.toFixed(1) + '%';
    }
  }

  // ---------- pipeline diagram ----------
  function buildPipeline() {
    const stages = [
      { name: 'input', dim: '28×28×1', viz: vizGrid(false) },
      { name: 'conv 3×3 · 32', dim: '26×26×32', viz: vizGrid(true) },
      { name: 'max pool 2×2', dim: '13×13×32', viz: vizGrid(true, 3) },
      { name: 'conv 3×3 · 64', dim: '11×11×64', viz: vizStack() },
      { name: 'max pool 2×2', dim: '5×5×64', viz: vizStack(true) },
      { name: 'dense · relu', dim: '1600 → 128', viz: vizDots() },
      { name: 'dense · softmax', dim: '128 → 10', viz: vizOutput() },
    ];
    const pipeline = document.getElementById('pipeline');
    stages.forEach(s => {
      const div = document.createElement('div');
      div.className = 'stage';
      div.innerHTML = `<div class="stage-viz">${s.viz}</div>
        <div class="stage-name">${s.name}</div>
        <div class="stage-dim mono">${s.dim}</div>`;
      pipeline.appendChild(div);
    });
  }

  function vizGrid(activated, sizeOverride) {
    let cells = '';
    for (let i = 0; i < 25; i++) {
      const on = activated && Math.random() > 0.55;
      cells += `<div class="${on ? 'on' : ''}"></div>`;
    }
    return `<div class="viz-grid">${cells}</div>`;
  }

  function vizStack(small) {
    const bars = [14, 26, 18, 34, 22, 30, 16];
    let html = '';
    bars.forEach(h => {
      const height = small ? h * 0.55 : h;
      html += `<div style="height:${height}px;"></div>`;
    });
    return `<div class="viz-stack" style="height:${small ? 20 : 36}px;">${html}</div>`;
  }

  function vizDots() {
    let html = '';
    for (let i = 0; i < 6; i++) html += '<div></div>';
    return `<div class="viz-dot-col">${html}</div>`;
  }

  function vizOutput() {
    return `<div class="viz-output">?</div>`;
  }

  buildPipeline();

  // ---------- boot ----------
  try {
    Model.load();
    statusDot.classList.add('ready');
    statusText.textContent = 'model ready — draw a digit';
  } catch (err) {
    statusText.textContent = 'failed to load weights';
    console.error(err);
  }
})();
