/* fly-balatro viewer client.
 *
 * Two halves that never talk to each other:
 *   BrainView  - three.js point cloud of 123,930 somata, one shader, glow decays
 *                every frame so a 50 ms window plays back as a propagating wave.
 *   PanelView  - the narrow column beside it: the mushroom body (drive,
 *                dopamine, synapses), the odour, and the population rates.
 *                Plain DOM plus three small canvases. No cards: the game is on
 *                the real Balatro window and this page never redraws it.
 *
 * The server sends everything for one action in a single burst; playback timing
 * lives here so a GIL stall in the numba kernel never shows up on screen.
 */
'use strict';

// ---------------------------------------------------------------- constants
// Same order as flybalatro.viewer.soma.CATEGORY_NAMES.
var CATEGORIES = [
  { key: 'sensory',   label: 'sensory',        rgb: [0.30, 0.74, 0.86], size: 0.0165, css: '#4cbcd9' },
  { key: 'ALPN',      label: 'ALPN (AL out)',  rgb: [0.24, 0.78, 0.70], size: 0.0215, css: '#3ec7b3' },
  { key: 'KC',        label: 'Kenyon cells',   rgb: [0.86, 0.62, 0.22], size: 0.0140, css: '#dc9e38' },
  { key: 'MBON',      label: 'MBON (MB out)',  rgb: [0.97, 0.78, 0.34], size: 0.0260, css: '#f8c757' },
  { key: 'CX',        label: 'central complex',rgb: [0.56, 0.52, 0.84], css: '#8f85d6', size: 0.0125 },
  { key: 'DN',        label: 'descending',     rgb: [0.93, 0.44, 0.22], size: 0.0215, css: '#ed7038' },
  { key: 'ascending', label: 'ascending',      rgb: [0.44, 0.48, 0.52], size: 0.0105, css: '#707a85' },
  { key: 'optic',     label: 'optic lobe',     rgb: [0.20, 0.25, 0.31], size: 0.0088, css: '#333f4d' },
  { key: 'central',   label: 'central brain',  rgb: [0.31, 0.35, 0.41], size: 0.0100, css: '#4f5865' }
];

var PIPELINE = ['ORN', 'ALPN', 'KC', 'MBON', 'DN'];

// Bit-strip tinting. The server sends the block layout in `hello` (it depends on
// the encoding version the loaded readout needs); this is the v1 fallback.
var BLOCK_COLORS = {
  best_type:        { on: '#f0b429', off: '#2a2008' },
  best_mask:        { on: '#f8c757', off: '#2a2008' },
  selected_type:    { on: '#e08a3c', off: '#241a0c' },
  selected_is_best: { on: '#ffd479', off: '#2a2008' },
  score_vs_needed:  { on: '#d99a2b', off: '#221b0e' },
  hand:             { on: '#4cbcd9', off: '#141c22' },
  hand_selected:    { on: '#d99a2b', off: '#221b0e' },
  selected:         { on: '#3ec7b3', off: '#111f1d' },
  context:          { on: '#8f85d6', off: '#17161f' }
};
var DEFAULT_BLOCK = { on: '#4cbcd9', off: '#141c22' };
var BIT_BLOCKS = [
  { name: 'hand',          label: 'hand cards',       from: 0,   to: 144 },
  { name: 'hand_selected', label: 'selected flags',   from: 144, to: 152 },
  { name: 'selected',      label: 'selected cards',   from: 152, to: 242 },
  { name: 'context',       label: 'counters / stage', from: 242, to: 283 }
];
var N_BITS = 283;
var GLOMERULI = null;

// Live-learning mode highlights four populations in the point cloud, because
// they are what its panel is about: the two MBON pools the decision is the
// difference of, and the two dopamine clusters that gate the two halves of the
// plasticity rule. Green/red for reward/punishment matches the flash.
// `size` overrides the category point size: these are a few hundred somata among
// 123,930, so at their default size the populations the panel is about are not
// findable on screen.
var HILITE = {
  approach: { rgb: [1.00, 0.83, 0.47], css: '#ffd479', size: 0.034, label: 'approach MBONs' },
  avoid:    { rgb: [0.70, 0.53, 1.00], css: '#b388ff', size: 0.034, label: 'avoid MBONs' },
  pam:      { rgb: [0.20, 0.82, 0.48], css: '#33d17a', size: 0.028, label: 'PAM (reward DANs)' },
  ppl1:     { rgb: [0.94, 0.27, 0.27], css: '#ef4444', size: 0.034, label: 'PPL1 (punishment DANs)' }
};

var GLOW_TAU_MS = 260;      // e-folding time of a spike flash
var GLOW_FLOOR = 0.004;     // below this a point leaves the active list
var SWAY_RATE = 0.16;       // rad/s of the slow azimuth sway
var SWAY_AZ = 0.38;         // +/- radians about the resting view; a full orbit
var SWAY_EL = 0.10;         // spends most of its time looking at the brain end-on

function fmt(n) { return (n === null || n === undefined) ? '—' : Number(n).toLocaleString('en-US'); }
function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
function el(id) { return document.getElementById(id); }

/* `play` -> PLAY, `discard` -> DIG, `select_card[3]` -> SELECT 3. The same
 * three words flybalatro/realgame/overlay.py's action_label produces, so the
 * embedded panel and the NSWindow overlay say the same thing. */
function action_word(name) {
  if (!name) return '';
  if (name === 'play') return 'PLAY';
  if (name === 'discard') return 'DIG';
  if (name === 'cash_out') return 'CASH OUT';
  var m = /^select_card\[(\d+)\]$/.exec(name);
  if (m) return 'SELECT ' + m[1];
  return name.replace(/_/g, ' ').toUpperCase();
}

// --------------------------------------------------------------- brain view
function BrainView(host) {
  this.host = host;
  this.ready = false;
  this.frames = 0;
  this.fps = 0;
  // Resting view: anterior, tipped down onto the dorsal surface, framed so the
  // two optic lobes sit left and right with the central brain between them.
  this.azimuth = 0.0;
  this.elevation = 0.70;
  this.radius = 2.45;
  this.phase = 0;
  this.dragging = false;
  this.activeIdx = [];
  this.activeFlag = null;
  this.lastTime = performance.now();
  this.fpsMark = this.lastTime;
  this._initGL();
  this._initInput();
}

BrainView.prototype._initGL = function () {
  var w = Math.max(1, this.host.clientWidth);
  var h = Math.max(1, this.host.clientHeight);
  this.scene = new THREE.Scene();
  this.camera = new THREE.PerspectiveCamera(42, w / h, 0.05, 40);
  this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: 'high-performance' });
  this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  this.renderer.setSize(w, h, false);
  this.renderer.setClearColor(0x000000, 0);
  this.host.appendChild(this.renderer.domElement);
  this.renderer.domElement.style.width = '100%';
  this.renderer.domElement.style.height = '100%';
  var self = this;
  window.addEventListener('resize', function () { self.resize(); });
  if (typeof ResizeObserver !== 'undefined') {
    this._ro = new ResizeObserver(function () { self.resize(); });
    this._ro.observe(this.host);
  }
};

BrainView.prototype._initInput = function () {
  var self = this, last = null;
  var dom = this.host;
  dom.addEventListener('pointerdown', function (e) {
    self.dragging = true; last = [e.clientX, e.clientY];
    dom.classList.add('dragging');
    dom.setPointerCapture(e.pointerId);
  });
  dom.addEventListener('pointermove', function (e) {
    if (!self.dragging || !last) return;
    self.azimuth -= (e.clientX - last[0]) * 0.006;
    self.elevation = clamp(self.elevation + (e.clientY - last[1]) * 0.005, -1.45, 1.45);
    last = [e.clientX, e.clientY];
  });
  function stop(e) {
    self.dragging = false; last = null;
    dom.classList.remove('dragging');
    if (e && e.pointerId !== undefined && dom.hasPointerCapture && dom.hasPointerCapture(e.pointerId)) {
      dom.releasePointerCapture(e.pointerId);
    }
  }
  dom.addEventListener('pointerup', stop);
  dom.addEventListener('pointercancel', stop);
  dom.addEventListener('wheel', function (e) {
    e.preventDefault();
    self.radius = clamp(self.radius * (1 + e.deltaY * 0.0012), 0.9, 9.0);
  }, { passive: false });
};

BrainView.prototype.resize = function () {
  var w = Math.max(1, this.host.clientWidth);
  var h = Math.max(1, this.host.clientHeight);
  this.camera.aspect = w / h;
  this.camera.updateProjectionMatrix();
  this.renderer.setSize(w, h, false);
  this.renderer.domElement.style.width = '100%';
  this.renderer.domElement.style.height = '100%';
  if (this.material) this.material.uniforms.uSizeScale.value = this._sizeScale();
};

BrainView.prototype._sizeScale = function () {
  // three's own sizeAttenuation convention: half the drawing-buffer height.
  return this.renderer.domElement.height * 0.5;
};

/* neurons.bin: <IIII> magic, version, nPoints, nNeurons | float32 xyz | uint8 code */
BrainView.prototype.load = function (buffer) {
  var head = new Uint32Array(buffer, 0, 4);
  if (head[0] !== 0x464C5931) throw new Error('neurons.bin: bad magic 0x' + head[0].toString(16));
  var n = head[2];
  var xyz = new Float32Array(buffer, 16, n * 3);
  var code = new Uint8Array(buffer, 16 + n * 12, n);

  var color = new Float32Array(n * 3);
  var base = new Float32Array(n);
  var glow = new Float32Array(n);
  for (var i = 0; i < n; i++) {
    var cat = CATEGORIES[code[i]] || CATEGORIES[CATEGORIES.length - 1];
    color[i * 3] = cat.rgb[0]; color[i * 3 + 1] = cat.rgb[1]; color[i * 3 + 2] = cat.rgb[2];
    base[i] = cat.size;
  }

  var geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
  geom.setAttribute('aColor', new THREE.BufferAttribute(color, 3));
  geom.setAttribute('aBase', new THREE.BufferAttribute(base, 1));
  geom.setAttribute('aGlow', new THREE.BufferAttribute(glow, 1));
  geom.computeBoundingSphere();

  this.material = new THREE.ShaderMaterial({
    uniforms: { uSizeScale: { value: this._sizeScale() } },
    vertexShader: [
      'attribute vec3 aColor;',
      'attribute float aBase;',
      'attribute float aGlow;',
      'uniform float uSizeScale;',
      'varying vec3 vColor;',
      'varying float vAlpha;',
      'void main() {',
      '  vec4 mv = modelViewMatrix * vec4(position, 1.0);',
      '  float g = clamp(aGlow, 0.0, 1.0);',
      '  float size = aBase * (1.0 + 0.35 * g) + g * 0.026;',
      '  gl_PointSize = size * uSizeScale / max(0.15, -mv.z);',
      '  gl_Position = projectionMatrix * mv;',
      '  vec3 hot = mix(aColor, vec3(1.0, 0.94, 0.82), 0.38) * 1.85;',
      '  vColor = mix(aColor, hot, g);',
      '  vAlpha = 0.15 + 0.70 * g;',
      '}'
    ].join('\n'),
    fragmentShader: [
      'varying vec3 vColor;',
      'varying float vAlpha;',
      'void main() {',
      '  vec2 d = gl_PointCoord - vec2(0.5);',
      '  float r2 = dot(d, d);',
      '  if (r2 > 0.25) discard;',
      '  float f = 1.0 - r2 * 3.6;',
      '  gl_FragColor = vec4(vColor, vAlpha * f);',
      '}'
    ].join('\n'),
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    depthTest: false
  });

  this.points = new THREE.Points(geom, this.material);
  this.scene.add(this.points);
  this.glow = glow;
  this.glowAttr = geom.attributes.aGlow;
  this.activeFlag = new Uint8Array(n);
  this.nPoints = n;
  this.ready = true;
  return n;
};

/* Recolour a named population in place. The colour lives in a vertex attribute,
 * so this is a one-off buffer write and costs nothing per frame; the shader's
 * hot-white mix still applies on top, so a recoloured point still reads as
 * "spiking" when it flashes. */
BrainView.prototype.recolor = function (indices, rgb, size) {
  if (!this.ready || !indices || !indices.length) return 0;
  var attr = this.points.geometry.attributes.aColor;
  var sattr = this.points.geometry.attributes.aBase;
  var arr = attr.array, sarr = sattr.array, n = this.nPoints, hit = 0;
  for (var k = 0; k < indices.length; k++) {
    var i = indices[k];
    if (i < 0 || i >= n) continue;
    arr[i * 3] = rgb[0]; arr[i * 3 + 1] = rgb[1]; arr[i * 3 + 2] = rgb[2];
    if (size) sarr[i] = size;
    hit++;
  }
  attr.needsUpdate = true;
  if (size) sattr.needsUpdate = true;
  return hit;
};

BrainView.prototype.flash = function (indices) {
  if (!this.ready) return 0;
  var glow = this.glow, flag = this.activeFlag, active = this.activeIdx, n = this.nPoints;
  var hit = 0;
  for (var k = 0; k < indices.length; k++) {
    var i = indices[k];
    if (i >= n) continue;
    glow[i] = 1.0;
    if (!flag[i]) { flag[i] = 1; active.push(i); }
    hit++;
  }
  this.glowAttr.needsUpdate = true;
  return hit;
};

BrainView.prototype.clearFlashes = function () {
  if (!this.ready) return;
  var active = this.activeIdx;
  for (var k = 0; k < active.length; k++) { this.glow[active[k]] = 0; this.activeFlag[active[k]] = 0; }
  this.activeIdx = [];
  this.glowAttr.needsUpdate = true;
};

BrainView.prototype.tick = function (now) {
  var dt = Math.min(0.1, (now - this.lastTime) / 1000);
  this.lastTime = now;
  this.frames++;
  if (now - this.fpsMark > 1000) {
    this.fps = Math.round(this.frames * 1000 / (now - this.fpsMark));
    this.frames = 0; this.fpsMark = now;
  }
  // _freeze is set from the console/devtools to hold a frame for a screenshot.
  if (!this.dragging && !this._freeze) this.phase += SWAY_RATE * dt;
  var az = this.azimuth + SWAY_AZ * Math.sin(this.phase);
  var elv = clamp(this.elevation + SWAY_EL * Math.sin(this.phase * 0.63), -1.45, 1.45);

  var ce = Math.cos(elv), se = Math.sin(elv);
  this.camera.position.set(
    this.radius * ce * Math.sin(az),
    this.radius * se,
    this.radius * ce * Math.cos(az)
  );
  this.camera.up.set(0, 1, 0);
  this.camera.lookAt(0, 0, 0);

  if (this.ready) {
    var active = this.activeIdx;
    if (active.length) {
      var decay = Math.exp(-(now - (this._decayMark || now)) / GLOW_TAU_MS);
      this._decayMark = now;
      var glow = this.glow, flag = this.activeFlag, w = 0;
      for (var k = 0; k < active.length; k++) {
        var i = active[k];
        var g = glow[i] * decay;
        if (g < GLOW_FLOOR) { glow[i] = 0; flag[i] = 0; continue; }
        glow[i] = g; active[w++] = i;
      }
      active.length = w;
      this.glowAttr.needsUpdate = true;
    } else {
      this._decayMark = now;
    }
    this.renderer.render(this.scene, this.camera);
  }
};

// --------------------------------------------------------------- table view
function PanelView() {
  this.barNodes = [];
  this.pipeNodes = {};
  this.pipeMax = {};
  this.plastic = null;
  this.bucketNodes = {};
  this.bitsCtx = el('bits').getContext('2d');
  this._buildBars();
  this._buildPipe();
}

PanelView.prototype._buildBars = function () {
  var host = el('ro-bars');
  host.textContent = '';
  for (var i = 0; i < 5; i++) {
    var d = document.createElement('div');
    d.className = 'bar';
    d.innerHTML = '<div class="fill"><i></i><span>—</span></div><div class="val">—</div>';
    host.appendChild(d);
    this.barNodes.push({
      root: d, fill: d.querySelector('i'), name: d.querySelector('span'), val: d.querySelector('.val')
    });
  }
};

PanelView.prototype._buildPipe = function () {
  var host = el('pipe');
  host.textContent = '';
  for (var i = 0; i < PIPELINE.length; i++) {
    if (i > 0) {
      var a = document.createElement('div');
      a.className = 'arw'; a.textContent = '›';
      host.appendChild(a);
    }
    var key = PIPELINE[i];
    var d = document.createElement('div');
    d.className = 'p';
    d.innerHTML = '<div class="nm">' + key + '</div>'
      + '<div class="hz">—<i> Hz</i></div>'
      + '<div class="m"><i></i></div>'
      + '<div class="cnt">—</div>';
    host.appendChild(d);
    this.pipeNodes[key] = {
      hz: d.querySelector('.hz'), meter: d.querySelector('.m > i'), cnt: d.querySelector('.cnt')
    };
    this.pipeMax[key] = 20;
  }
};

PanelView.prototype.setPopulations = function (pops) {
  for (var i = 0; i < PIPELINE.length; i++) {
    var key = PIPELINE[i];
    if (this.pipeNodes[key] && pops[key] !== undefined) {
      this.pipeNodes[key].cnt.textContent = fmt(pops[key]) + ' cells';
    }
  }
};

/* One frame's sensory input. The board that came with it is deliberately not
 * drawn: the cards are on the real Balatro window, and a second, prettier copy
 * of them here would be this page's own invention. */
PanelView.prototype.state = function (frame) {
  el('bits-sub').textContent = frame.n_bits_on + ' of ' + (frame.n_bits || N_BITS)
    + ' bits on';
  this.bits(frame.bits);
  this.glomeruli(frame.bits);
  this.staleReadout();
};

// One chip per relay bit under the glomerular (v3) encoding: that bit drives
// every olfactory receptor neuron of one named glomerulus and nothing else, so
// the chip grid *is* the input the brain gets. Bits 32+ are not sent at all.
PanelView.prototype.setGlomeruli = function (glomeruli, nBits, nToBrain) {
  GLOMERULI = (glomeruli && glomeruli.length) ? glomeruli : null;
  var wrap = el('glom-wrap');
  // Under a glomerular encoding the chip grid *is* the input, and the bit strip
  // below it would be the same 32 bits a second time. One picture of the odour.
  el('bits-wrap').hidden = !!GLOMERULI;
  if (!GLOMERULI) { wrap.hidden = true; return; }
  wrap.hidden = false;
  var grid = el('glom');
  grid.textContent = '';
  GLOMERULI.forEach(function (g) {
    var d = document.createElement('div');
    d.className = 'gl';
    d.id = 'gl-' + g.bit;
    d.title = 'bit ' + g.bit + ' \u00b7 ' + g.feature + ' \u2192 ' +
      g.glomerulus + ' (' + g.n_orn + ' receptor neurons)';
    d.innerHTML = g.glomerulus.replace(/^ORN_/, '') +
      '<span class="n">' + g.n_orn + '</span>';
    grid.appendChild(d);
  });
  // The readout modes encode 315 bits and send 32; the plastic harness computes
  // only the 32, so there is no "other" block there and the note must not
  // invent one.
  var dropped = nBits - nToBrain;
  el('glom-note').textContent =
    GLOMERULI.length + ' relay bits \u2192 ' + GLOMERULI.length +
    ' whole ORN glomeruli, 30 mV each.' + (dropped > 0
      ? ' The other ' + dropped + ' game-state bits are not sent to the brain'
        + ' under this encoding; they are drawn below in grey for reference.'
      : ' That is the entire input \u2014 the hand analysis, computed outside'
        + ' the brain, and nothing else about the board.');
};

PanelView.prototype.glomeruli = function (bits) {
  if (!GLOMERULI) return;
  for (var i = 0; i < GLOMERULI.length; i++) {
    var d = el('gl-' + GLOMERULI[i].bit);
    if (d) d.className = 'gl' + (bits[GLOMERULI[i].bit] ? ' on' : '');
  }
};

PanelView.prototype.setBlocks = function (blocks, nBits) {
  if (blocks && blocks.length) BIT_BLOCKS = blocks;
  if (nBits) N_BITS = nBits;
  var key = el('bits-key');
  key.textContent = '';
  BIT_BLOCKS.forEach(function (b) {
    var c = BLOCK_COLORS[b.name] || DEFAULT_BLOCK;
    var s = document.createElement('span');
    s.innerHTML = '<i style="background:' + c.on + '"></i>' +
      b.from + '\u2013' + (b.to - 1) + ' ' + (b.label || b.name) +
      (b.relay ? ' *' : '') + (b.sent === false ? ' \u2298' : '');
    key.appendChild(s);
  });
  if (BIT_BLOCKS.some(function (b) { return b.relay; })) {
    var note = document.createElement('span');
    note.textContent = '* relayed hand analysis, computed outside the brain';
    key.appendChild(note);
  }
  if (BIT_BLOCKS.some(function (b) { return b.sent === false; })) {
    var n2 = document.createElement('span');
    n2.textContent = '\u2298 not sent to the brain';
    key.appendChild(n2);
  }
};

PanelView.prototype.bits = function (bits) {
  var ctx = this.bitsCtx, cols = 45, cw = 13, ch = 12, pad = 2;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  for (var i = 0; i < bits.length; i++) {
    var block = null;
    for (var b = 0; b < BIT_BLOCKS.length; b++) {
      if (i >= BIT_BLOCKS[b].from && i < BIT_BLOCKS[b].to) { block = BIT_BLOCKS[b]; break; }
    }
    var c = (block && BLOCK_COLORS[block.name]) || DEFAULT_BLOCK;
    // A block the encoding does not send is drawn faint whether it is on or off:
    // the screen must not suggest the brain is reading it.
    ctx.globalAlpha = (block && block.sent === false) ? 0.28 : 1.0;
    ctx.fillStyle = bits[i] ? c.on : c.off;
    ctx.fillRect((i % cols) * cw, Math.floor(i / cols) * ch, cw - pad, ch - pad);
  }
  ctx.globalAlpha = 1.0;
};

// ------------------------------------------------------------- mushroom body
/* The panel that is shown whenever the fly decides for itself: this process's
 * own learning loop, or a real game it is following. Everything on it is either
 * a rate the fly's own neurons produced, a weight of a real synapse, or a count
 * of what the game paid. Nothing here is fitted.
 *
 * Two loops write it -- flybalatro/viewer/plastic.py and, via
 * flybalatro/viewer/realgame_source.py, flybalatro/realgame/plastic.py -- and
 * the second is mapped onto the first's names server-side, so there is one
 * schema here and one render path. What still differs between them is what each
 * one *knows*: the real-game log has no per-hand outcome sentence and no game
 * counter, so those read as absent rather than as zero. */
PanelView.prototype.setPlastic = function (plastic) {
  this.plastic = plastic || null;
  el('ro-sect').hidden = !!plastic;
  el('mb-sect').hidden = !plastic;
  if (!plastic) return;

  el('mb-tag').textContent = plastic.homeostasis === null || plastic.homeostasis === undefined
    ? plastic.label
    : plastic.version + (plastic.homeostasis ? ' · KC homeostasis' : ' · no homeostasis');
  // The operating point belongs to whichever loop produced the record, so the
  // formula waits for a frame rather than being asserted from setup.
  this.formula(null);

  var host = el('mb-buckets');
  host.textContent = '';
  this.bucketNodes = {};
  var self = this;
  (plastic.buckets || []).forEach(function (b) {
    var d = document.createElement('div');
    d.className = 'bk';
    d.innerHTML = '<div class="col"><i></i></div><div class="pv">—</div>'
      + '<div class="lb">' + b.label + '</div><div class="n">—</div>';
    d.title = b.bucket + ': best available hand scores '
      + (b.bucket === 'ge1.0' ? 'at least' : 'less than') + ' '
      + ({ 'lt0.25': 'a quarter of', 'lt0.5': 'half of', 'lt1.0': 'all of',
           'ge1.0': 'all of' }[b.bucket] || '') + ' what is still needed';
    host.appendChild(d);
    self.bucketNodes[b.bucket] = {
      fill: d.querySelector('i'), pv: d.querySelector('.pv'), n: d.querySelector('.n')
    };
  });
  el('mb-bk-note').textContent =
    'The bucket is 4 of the 32 relay bits — the one ordered axis the odour '
    + 'carries. Four separated bars = a valence learned per odour; four equal '
    + 'bars = one number learned for every odour.';
  el('mb-w-sub').textContent = fmt(plastic.pools.kc_mbon_edges) + ' synapses · floor '
    + plastic.weight_floor + '× original';
  this.mbStale();
};

/* The drive rule, written out with this run's own two constants. */
PanelView.prototype.formula = function (m) {
  var p = this.plastic;
  if (!p) return;
  var bias = m && m.bias !== null && m.bias !== undefined ? m.bias : p.bias;
  var temp = m && m.temperature !== null && m.temperature !== undefined
    ? m.temperature : p.temperature;
  if (bias === null || bias === undefined || temp === null || temp === undefined) {
    el('mb-formula').innerHTML = 'play_drive = mean rate(' + p.n_approach
      + ' approach MBONs) − mean rate(' + p.n_avoid + ' avoid MBONs) + bias';
    return;
  }
  var floor = p.explore_floor;
  var soft = (floor === null || floor === undefined)
    ? 'σ(drive / ' + temp.toFixed(2) + ')'
    : (floor / 2).toFixed(2) + ' + ' + (1 - floor).toFixed(2)
      + ' · σ(drive / ' + temp.toFixed(2) + ')';
  el('mb-formula').innerHTML =
    'play_drive = mean rate(' + p.n_approach + ' approach) − mean rate('
    + p.n_avoid + ' avoid) ' + (bias < 0 ? '− ' : '+ ')
    + Math.abs(bias).toFixed(2) + ' Hz &nbsp;·&nbsp; P(play) = ' + soft;
};

PanelView.prototype.mbStale = function () {
  el('mb-drive').className = 'stale';
  el('mb-verdict').className = 'stale';
};

function pct(v, digits) {
  return (v === null || v === undefined) ? '—' : (v * 100).toFixed(digits || 0) + '%';
}

/* One hand's worth of mushroom body.
 *
 * Two records can land here. A decision carries the whole thing: the rates its
 * MBONs produced, the call it made, and the state of the synapses at that
 * moment. An outcome carries only what the pulse changed -- it is not a window,
 * nothing fired for it, and there are no rates on it. So the drive and the
 * verdict are rendered only when the record has them, and otherwise are left
 * showing the decision they belong to rather than being blanked or, worse,
 * filled with the previous hand's numbers under a new heading. */
PanelView.prototype.mb = function (m) {
  if (!this.plastic) return;
  this.trace(m);
  if (m.record === 'outcome' || m.approach_hz === null || m.approach_hz === undefined) return;
  el('mb-drive').className = '';
  el('mb-verdict').className = '';
  this.formula(m);

  var scale = Math.max(8, m.approach_hz, m.avoid_hz);
  el('mb-app').textContent = m.approach_hz.toFixed(2) + ' Hz';
  el('mb-avo').textContent = m.avoid_hz.toFixed(2) + ' Hz';
  el('mb-app-bar').style.width = clamp(m.approach_hz / scale * 100, 0, 100) + '%';
  el('mb-avo-bar').style.width = clamp(m.avoid_hz / scale * 100, 0, 100) + '%';

  // P(play) is this hand's probability where the record has it. Runs recorded
  // before the two numbers were split carry only the rolling play rate, and the
  // bar says which one it is showing rather than passing one off as the other.
  var p = (m.p_play === null || m.p_play === undefined) ? m.p_play_rolling : m.p_play;
  var rolling = m.p_play === null || m.p_play === undefined;
  el('mb-p').textContent = pct(p);
  el('mb-p-bar').style.width = (p === null || p === undefined ? 0 : clamp(p * 100, 0, 100)) + '%';
  el('mb-sect').querySelector('.dr.net .k').textContent = rolling ? 'play rate' : 'P(play)';
  el('mb-drive-note').textContent = 'play_drive ' + (m.play_drive >= 0 ? '+' : '')
    + m.play_drive.toFixed(2) + ' Hz · ' + fmt(m.kc_active) + ' of '
    + fmt(this.plastic.n_kc) + ' Kenyon cells fired';
  el('mb-sub').textContent = m.hands ? 'hand ' + fmt(m.hands) : 'first hand';

  var dig = m.action !== 'play';
  el('mb-act').textContent = dig ? 'DIG' : 'PLAY';
  el('mb-act').className = 'act ' + (dig ? 'dig' : 'play')
    + (m.explored ? ' explored' : '');
  var h = m.hand;
  el('mb-outcome').textContent = m.outcome !== undefined && m.outcome !== null
    ? m.outcome
    : (dig ? 'dug — threw the worst cards back' : 'played the best subset it was told about');
  el('mb-hand').textContent = h.best_type + ' · ' + fmt(h.best_score)
    + ' of ' + fmt(h.needed) + ' needed (' + h.bucket_label + ') · '
    + h.plays + ' plays, ' + h.discards + ' discards left'
    + (m.explored ? ' · exploration' : '')
    + (h.discard_ok ? '' : ' · no discard left, play forced');

};

/* The half of the panel a dopamine pulse can change on its own: the pulse, the
 * synapse state after it, and the rolling picture of what the fly is doing. */
PanelView.prototype.trace = function (m) {
  if (!this.plastic) return;
  // The real game logs the pulse on its own record, one beat after the decision
  // it pays for, so a decision record there says nothing about dopamine and
  // must not be allowed to wipe the pulse that is still being shown. The
  // headless loop resolves the hand inside the decision, and does carry it.
  if (!(m.source === 'realgame' && m.record === 'decision')) this.dopamine(m);

  var w = m.weights;
  var stats = el('mb-wstats');
  stats.textContent = '';
  [['changed', pct(w.frac_changed, 1), fmt(w.n_changed) + ' of ' + fmt(w.n_edges)],
   ['mean weight', pct(w.mean_ratio, 1), 'of original'],
   ['at the floor', fmt(w.n_at_floor), pct(w.frac_at_floor, 2)]
  ].forEach(function (row) {
    var d = document.createElement('div');
    d.className = 'ws';
    d.innerHTML = '<span class="k">' + row[0] + '</span><span class="v">' + row[1]
      + '</span><span class="s">' + row[2] + '</span>';
    stats.appendChild(d);
  });
  var ref = m.weight_ref !== undefined ? m.weight_ref : w.mean_w0;
  this.spark('mb-wspark', m.weight_history, '#dc9e38', ref);

  (m.buckets || []).forEach(function (b) {
    var node = this.bucketNodes[b.bucket];
    if (!node) return;
    var has = b.p_play !== null && b.p_play !== undefined;
    node.fill.style.height = (has ? clamp(b.p_play * 100, 1.5, 100) : 0) + '%';
    node.pv.textContent = has ? pct(b.p_play) : '—';
    node.n.textContent = b.n ? 'n=' + b.n : '—';
  }, this);

  this.spark('mb-rspark', m.reward_history, '#3ec7b3', null);
  var window_ = m.rolling_window || 50;
  el('mb-rr-sub').textContent = (m.reward_rate === null || m.reward_rate === undefined)
    ? 'no hands yet'
    : pct(m.reward_rate) + ' of the last ' + Math.min(window_, m.hands) + ' hands'
      + (m.hands > window_ ? ' · ' + fmt(m.hands) + ' total' : '')
      + (m.games === undefined ? '' : ' · ' + fmt(m.games) + ' games');
};

/* Back to no pulse, at the start of the next hand rather than at the end of
 * this one: the colour fades over a quarter second, and clearing it the instant
 * the next record lands leaves a frame reading "no pulse" on a green row. */
PanelView.prototype.dopamineClear = function () {
  if (!this.plastic) return;
  el('mb-da').className = 'da none';
  el('mb-da-dot').className = 'dot none';
  el('mb-da-text').textContent = 'no pulse';
  // Keep the running totals -- still true -- and drop the "N synapses
  // depressed" tail, which belonged to the pulse that has just ended.
  var ct = el('mb-da-count');
  ct.textContent = ct.textContent.split(' \u00b7 ')[0];
};

/* The pulse, and the running total of pulses. Split out because in the real
 * game it arrives on its own record, one hand after the decision it pays for. */
PanelView.prototype.dopamine = function (m) {
  if (!this.plastic) return;
  var kind = m.dopamine || 'none';
  el('mb-da').className = 'da ' + kind;
  el('mb-da-dot').className = 'dot ' + kind;
  el('mb-da-text').textContent =
    kind === 'reward' ? 'PAM pulse — reward'
      : (kind === 'punish' ? 'PPL1 pulse — punishment'
        : (m.learning === false ? 'learning off' : 'no pulse'));
  el('mb-da-count').textContent = fmt(m.pulses.reward) + ' reward / '
    + fmt(m.pulses.punish) + ' punish'
    + (kind !== 'none' && m.dopamine_changed
        ? ' · ' + fmt(m.dopamine_changed) + ' synapses depressed' : '');
};

/* A strip chart with no axes: the shape is the message, and the numbers that
 * matter are printed beside it. `ref` draws a dashed baseline (the original mean
 * weight), so "down from where it started" is visible rather than asserted. */
PanelView.prototype.spark = function (id, values, css, ref) {
  var cv = el(id);
  if (!cv) return;
  var ctx = cv.getContext('2d'), w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);
  if (!values || values.length < 2) return;
  var lo = Infinity, hi = -Infinity, i;
  for (i = 0; i < values.length; i++) { lo = Math.min(lo, values[i]); hi = Math.max(hi, values[i]); }
  if (ref !== null && ref !== undefined) { lo = Math.min(lo, ref); hi = Math.max(hi, ref); }
  var span = (hi - lo) || Math.abs(hi) || 1;
  lo -= span * 0.12; hi += span * 0.12; span = hi - lo;
  function y(v) { return h - 3 - (v - lo) / span * (h - 6); }
  function x(k) { return 1 + k / (values.length - 1) * (w - 2); }
  if (ref !== null && ref !== undefined) {
    ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = 'rgba(190,200,215,0.38)';
    ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, y(ref)); ctx.lineTo(w, y(ref));
    ctx.stroke(); ctx.restore();
  }
  ctx.beginPath();
  ctx.moveTo(x(0), y(values[0]));
  for (i = 1; i < values.length; i++) ctx.lineTo(x(i), y(values[i]));
  ctx.strokeStyle = css; ctx.lineWidth = 1.6; ctx.lineJoin = 'round';
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x(values.length - 1), y(values[values.length - 1]), 2.2, 0, Math.PI * 2);
  ctx.fillStyle = css; ctx.fill();
};

PanelView.prototype.staleReadout = function () {
  if (this.plastic) { this.mbStale(); return; }
  el('ro-bars').className = 'bars stale';
};

PanelView.prototype.clearReadout = function () {
  el('ro-bars').className = 'bars';
  for (var i = 0; i < this.barNodes.length; i++) {
    var n = this.barNodes[i];
    n.root.className = 'bar';
    n.fill.style.width = '0%';
    n.name.textContent = '—';
    n.val.textContent = '—';
  }
};

PanelView.prototype.decision = function (frame, hasLogits) {
  el('ro-bars').className = 'bars';
  var top = frame.top || [];
  var lo = Infinity, hi = -Infinity;
  for (var i = 0; i < top.length; i++) { lo = Math.min(lo, top[i].logit); hi = Math.max(hi, top[i].logit); }
  var span = (hi - lo) || 1;
  for (var j = 0; j < this.barNodes.length; j++) {
    var node = this.barNodes[j], row = top[j];
    if (!row) {
      node.root.className = 'bar'; node.fill.style.width = '0%';
      node.name.textContent = '—'; node.val.textContent = '—';
      continue;
    }
    node.root.className = 'bar' + (row.chosen ? ' chosen' : '');
    node.fill.style.width = clamp(10 + 90 * (row.logit - lo) / span, 6, 100) + '%';
    node.name.textContent = row.name;
    node.val.textContent = hasLogits ? row.logit.toFixed(2) : (row.chosen ? 'pick' : '—');
  }
  for (var k = 0; k < PIPELINE.length; k++) {
    var key = PIPELINE[k], hz = frame.rates[key], n = this.pipeNodes[key];
    if (!n || hz === undefined) continue;
    this.pipeMax[key] = Math.max(this.pipeMax[key] * 0.995, hz, 5);
    n.hz.innerHTML = hz.toFixed(1) + '<i> Hz</i>';
    n.meter.style.width = clamp(hz / this.pipeMax[key] * 100, 0, 100) + '%';
  }
};


// ------------------------------------------------------------------- client
/* ------------------------------------------------------------------------- */
/* The real game, inside this page                                           */
/* ------------------------------------------------------------------------- */
/* The server captures the Balatro window and pushes JPEG frames down the same
 * websocket the spikes come on, tagged so they cannot be mistaken for a spike
 * sub-step. This draws them, and draws the fly's highlight on top of them.
 *
 * The boxes are NOT re-fitted here. The server sends them in the game's own
 * canvas units -- the same units the captured frame is a picture of, because
 * the frame is that canvas cropped of its title bar and downscaled -- so the
 * whole transform is one ratio, `frame.width / canvas[0]`. Both the rects and
 * the pixels come out of the same game, which is the reason this is more
 * accurate than a transparent window floating over the real screen. */
var GAME_TAG = 0x80000001;
var GAME_COLORS = { best: '#f0b429', selected: '#3ec7b3', card: '#7d8894' };
/* How long a decision's card rects can be trusted against a live frame. One
 * Balatro select-and-play animation is comfortably inside this. */
var BOX_FRESH_MS = 1200;

function GameView() {
  this.canvas = el('game-canvas');
  this.ctx = this.canvas.getContext('2d');
  this.bitmap = null;
  this.units = null;       // [w, h] of the LOVE canvas the boxes live in
  this.pov = null;         // boxes + headline for the decision being shown
  this.povAt = 0;          // when it arrived; the boxes describe that instant only
  this.frameAt = 0;        // performance.now() of the last decoded frame
  this.interval = 1000 / 12;
  this.decoding = false;
  this.pending = null;     // newest undecoded frame; older ones are dropped
  this.state = 'off';
  this.on = false;
  this.staleShown = false;
}

GameView.prototype.enable = function (info) {
  var on = !!info;
  if (on !== this.on) {
    this.on = on;
    el('game-sect').hidden = !on;
    document.body.classList.toggle('gameview', on);
  }
  if (on) this.status(info);
};

/* What the capture says about itself: how fast, how expensive, and -- when
 * there is no picture -- why not, in a sentence rather than an empty box. */
GameView.prototype.status = function (info) {
  this.state = info.state || 'off';
  if (info.fps_target) this.interval = 1000 / info.fps_target;
  el('game-sub').textContent = info.state === 'live'
    ? (info.fps || 0).toFixed(1) + ' fps · ' + Math.round(info.capture_ms + info.encode_ms)
      + ' ms · ' + Math.round((info.bytes || 0) / 1024) + ' KB/frame'
    : (info.state || '—');
  if (info.state === 'live' && !this.bitmap) {
    // Coming back from "not running": the old sentence must not sit over the
    // panel until the first frame decodes, because by then it is false.
    el('game-msg').textContent = 'waiting for the first frame\u2026';
  }
  if (info.state !== 'live') {
    // No picture at all: drop the frame rather than keep the last one on
    // screen behind a message, because a message over a live-looking frame
    // reads as a caption, not as "this is not the game".
    this.bitmap = null;
    this.frameAt = 0;
    var msg = el('game-msg');
    msg.hidden = false;
    msg.textContent = info.text || info.state || 'no game view';
    el('game-stale').hidden = true;
    this._clear();
  }
};

/* Which rects, which decision they are the rects of, and how old they are. The
 * frame under them is live and the record is not: Balatro raises a selected
 * card and animates a played one, so between the record and the next capture
 * the game can move a card out from under its box. Naming the decision and the
 * age is how the panel says the boxes are a snapshot and the picture is not. */
GameView.prototype._align = function (pov, ageMs) {
  el('game-align').textContent = pov
    ? (pov.align === 'game' ? 'game rects' : 'fitted model')
      + (pov.index ? ' \u00b7 decision ' + pov.index : '')
      + (pov.moving ? ' \u00b7 cards moving' : '')
      + (ageMs > BOX_FRESH_MS
          ? ' \u00b7 boxes ' + (ageMs / 1000).toFixed(1) + ' s old'
          : '')
    : '\u2014';
};

GameView.prototype._clear = function () {
  if (!this.canvas.width) return;
  this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
};

/* One binary frame off the socket. Only the newest undecoded one is kept:
 * createImageBitmap is asynchronous and a queue would put the panel behind the
 * brain it is meant to sit beside. */
GameView.prototype.onFrame = function (buffer) {
  var view = new DataView(buffer);
  var w = view.getUint32(16, true), h = view.getUint32(20, true);
  var jpeg = new Uint8Array(buffer, 24);
  this.pending = { w: w, h: h, blob: new Blob([jpeg], { type: 'image/jpeg' }) };
  this._decode();
};

GameView.prototype._decode = function () {
  if (this.decoding || !this.pending) return;
  var self = this, next = this.pending;
  this.pending = null;
  this.decoding = true;
  createImageBitmap(next.blob).then(function (bmp) {
    if (self.bitmap && self.bitmap.close) self.bitmap.close();
    self.bitmap = bmp;
    self.frameAt = performance.now();
    el('game-msg').hidden = true;
    self.draw();
  }).catch(function (err) {
    console.error('game frame decode failed', err);
  }).then(function () {
    self.decoding = false;
    self._decode();
  });
};

/* The highlight for the decision now on screen, in game canvas units. */
GameView.prototype.setPov = function (pov) {
  this.pov = pov || null;
  this.povAt = pov ? performance.now() : 0;
  if (pov && pov.canvas) this.units = pov.canvas;
  this._lastBoxAge = -1;
  this._align(pov, 0);
  if (pov) {
    var hl = el('game-headline');
    hl.textContent = pov.headline || '—';
    hl.className = 'hl' + (pov.ok ? ' ok' : '');
    el('game-note').textContent = pov.note || '';
  }
  this.draw();
};

/* PLAY / DIG, held back until the 50 ms wave has played out in the cloud --
 * the game panel must not announce the decision before the brain that made it
 * has finished firing. */
GameView.prototype.setVerdict = function (text) {
  var act = el('game-act');
  act.textContent = text || '—';
  act.className = 'act' + (text ? '' : ' idle');
};

/* A hand that has already resolved: the game is animating those cards away and
 * dealing their replacements, so the rects in the record no longer describe
 * anything on screen. Boxes go, the headline stays. */
GameView.prototype.clearBoxes = function () {
  if (this.pov) this.pov = Object.assign({}, this.pov, { boxes: [] });
  this.draw();
};

GameView.prototype.draw = function () {
  var bmp = this.bitmap;
  if (!bmp) return;
  var c = this.canvas, ctx = this.ctx;
  if (c.width !== bmp.width || c.height !== bmp.height) {
    c.width = bmp.width;
    c.height = bmp.height;
  }
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(bmp, 0, 0);
  var pov = this.pov;
  if (!pov || !pov.boxes || !pov.boxes.length || !this.units) return;
  var s = c.width / this.units[0];
  ctx.save();
  ctx.lineJoin = 'miter';
  // The rects are the card positions at the instant of the record; the frame
  // under them is live. Balatro raises a card the moment it is selected and
  // flies a played hand off the table, so within about a second of a decision
  // the boxes stop describing anything on screen. They fade rather than keep
  // asserting themselves, and the tag under the frame prints their age.
  ctx.globalAlpha = (performance.now() - this.povAt) > BOX_FRESH_MS ? 0.3 : 1.0;
  for (var i = 0; i < pov.boxes.length; i++) {
    var b = pov.boxes[i];
    var x = b.x * s, y = b.y * s, w = b.w * s, h = b.h * s;
    ctx.save();
    // Rotated about the rect's own centre, exactly as the NSWindow overlay
    // does it, so the outline sits on the card's border instead of being an
    // upright box oversized to cover a card tilted by up to 5.5 degrees.
    ctx.translate(x + w / 2, y + h / 2);
    if (b.angle) ctx.rotate(b.angle);
    var col = GAME_COLORS[b.kind] || GAME_COLORS.card;
    if (b.fill) {
      ctx.fillStyle = col;
      ctx.fillRect(-w / 2, -h / 2, w, h);
    } else {
      ctx.strokeStyle = col;
      ctx.lineWidth = Math.max(1.5, 0.0035 * c.width);
      ctx.strokeRect(-w / 2, -h / 2, w, h);
    }
    ctx.restore();
  }
  ctx.restore();
};

/* A frame that stopped arriving is labelled, never quietly left up. The game
 * can be minimised, the capture can fail, the machine can be busy; any of those
 * freezes the picture, and a frozen picture of a card game is indistinguishable
 * from a live one. */
GameView.prototype.tick = function (now) {
  if (!this.on) return;
  if (this.pov && this.povAt) {
    var boxAge = now - this.povAt;
    if (boxAge > BOX_FRESH_MS && this._lastBoxAge !== Math.round(boxAge / 500)) {
      this._lastBoxAge = Math.round(boxAge / 500);
      this._align(this.pov, boxAge);
      this.draw();
    }
  }
  if (this.state !== 'live' || !this.frameAt) return;
  var age = now - this.frameAt;
  var stale = age > Math.max(600, this.interval * 4);
  if (stale) {
    el('game-stale').hidden = false;
    el('game-stale').textContent = 'stale · no frame for ' + (age / 1000).toFixed(1) + ' s';
    this.staleShown = true;
  } else if (this.staleShown) {
    el('game-stale').hidden = true;
    this.staleShown = false;
  }
};

function Client() {
  this.brain = new BrainView(el('brain'));
  this.panel = new PanelView();
  this.game = new GameView();
  this.hello = null;
  this.status = null;
  this.pendingSubsteps = [];
  this.queue = [];
  this.waveMs = 100;
  this.toastTimer = null;
  var self = this;
  this._controls();
  this._loadCloud().then(function () { self._connect(); });
  (function loop(now) {
    self._drain(now);
    self.brain.tick(now);
    self.game.tick(now);
    el('st-fps').textContent = self.brain.fps;
    requestAnimationFrame(loop);
  })(performance.now());
  // Safety net: if rAF is throttled or paused, the queue still advances.
  setInterval(function () { self._drain(performance.now()); }, 40);
}

Client.prototype._loadCloud = function () {
  var self = this;
  return fetch('/static/neurons.bin').then(function (r) {
    if (!r.ok) throw new Error('neurons.bin HTTP ' + r.status);
    return r.arrayBuffer();
  }).then(function (buf) {
    var n = self.brain.load(buf);
    el('bt-points').textContent = fmt(n);
    self.brain.resize();
  }).catch(function (err) {
    el('ck-note').textContent = 'point cloud failed to load: ' + err.message;
    console.error(err);
  });
};

Client.prototype._controls = function () {
  var self = this;
  el('btn-play').addEventListener('click', function () { self.send({ cmd: 'toggle' }); });
  el('btn-step').addEventListener('click', function () { self.send({ cmd: 'step' }); });
  el('btn-reset').addEventListener('click', function () { self.send({ cmd: 'reset' }); });
  el('sel-speed').addEventListener('change', function (e) {
    self.send({ cmd: 'speed', value: parseFloat(e.target.value) });
  });
  el('sel-cond').addEventListener('change', function (e) {
    self.send({ cmd: 'condition', value: e.target.value });
  });
  el('sel-learn').addEventListener('change', function (e) {
    self.send({ cmd: 'learning', value: e.target.value === 'on' });
  });
  window.addEventListener('keydown', function (e) {
    if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    if (e.code === 'Space') { e.preventDefault(); self.send({ cmd: 'toggle' }); }
    if (e.code === 'ArrowRight') { e.preventDefault(); self.send({ cmd: 'step' }); }
    if (e.key === 'r') self.send({ cmd: 'reset' });
  });
};

Client.prototype._connect = function () {
  var self = this;
  var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  this.ws = new WebSocket(proto + '//' + location.host + '/ws');
  this.ws.binaryType = 'arraybuffer';
  this.ws.onmessage = function (ev) {
    if (typeof ev.data === 'string') self._onJson(JSON.parse(ev.data));
    else self._onBinary(ev.data);
  };
  this.ws.onclose = function () {
    el('head-mode').innerHTML = '<span class="warn">disconnected</span> — restart the server and reload';
    setTimeout(function () { self._connect(); }, 3000);
  };
  this.ws.onerror = function (e) { console.error('websocket error', e); };
};

Client.prototype.send = function (msg) {
  if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(msg));
};

/* Playback is driven from requestAnimationFrame, not setTimeout: a hidden tab
 * clamps setTimeout to 1 Hz, which stretched the 570 ms wave over six seconds
 * and let the next action cancel the readout reveal before it ever ran. rAF
 * simply stops while hidden, and _flush() runs whatever is still queued the
 * moment the next action arrives, so the readout is never left blank. */
Client.prototype._flush = function () {
  var queue = this.queue;
  this.queue = [];
  for (var i = 0; i < queue.length; i++) queue[i].fn();
};

Client.prototype._drain = function (now) {
  var queue = this.queue;
  while (queue.length && queue[0].at <= now) queue.shift().fn();
};

Client.prototype._at = function (delayMs, fn) {
  this.queue.push({ at: performance.now() + delayMs, fn: fn });
};

Client.prototype._onJson = function (msg) {
  switch (msg.t) {
    case 'hello': return this._onHello(msg);
    case 'status': return this._onStatus(msg);
    case 'state': return this._onState(msg);
    case 'decision': return this._onDecision(msg);
    case 'outcome': return this._onOutcome(msg);
    case 'episode': return this._onEpisode(msg);
    case 'game': return this.game.enable(msg);
    case 'notice': return this._toast(msg.text, '', 1400);
    default: return undefined;
  }
};

Client.prototype._onHello = function (msg) {
  this.hello = msg;
  this.game.enable(msg.game);
  el('bt-edges').textContent = fmt(msg.n_edges);
  el('bt-syn').textContent = fmt(msg.n_synapses);
  el('bt-points').textContent = fmt(msg.n_points);
  this.panel.setPopulations(msg.populations);
  this.panel.setBlocks(msg.bit_blocks, msg.n_features);
  this.panel.setPlastic(msg.plastic);
  this.highlight = msg.highlight || null;
  this._applyHighlight();
  var toBrain = msg.n_bits_to_brain || msg.n_features;
  this.panel.setGlomeruli(msg.glomeruli, msg.n_features, toBrain);
  el('bits-title').textContent = 'sensory input \u00b7 ' + fmt(msg.n_features) + ' bits'
    + (msg.encoding === 'glomerular32'
        ? ' \u00b7 ' + toBrain + ' sent to the brain as whole glomeruli'
          + (msg.n_features > toBrain ? ', ' + (msg.n_features - toBrain) + ' not sent' : '')
        : (msg.n_relay_bits
            ? ' (first ' + msg.n_relay_bits + ' are the relayed hand analysis)' : ''));

  // The legend leads with the four populations the panel is about -- they are
  // the reason the cloud is coloured at all -- and the broad categories follow,
  // dimmed, as context for the rest of the brain.
  var legend = el('legend');
  legend.textContent = '';
  var hl = this.highlight;
  if (hl) {
    ['approach', 'avoid', 'pam', 'ppl1'].forEach(function (key) {
      var pts = hl[key] || [];
      if (!pts.length) return;
      var spec = HILITE[key];
      var row = document.createElement('div');
      row.className = 'row hl';
      row.innerHTML = '<span class="dot" style="background:' + spec.css + '"></span>'
        + '<span class="name">' + spec.label + '</span><span class="n">'
        + fmt(pts.length) + '</span>';
      legend.appendChild(row);
    });
  }
  for (var i = 0; i < CATEGORIES.length; i++) {
    var cat = CATEGORIES[i], n = (msg.category_counts || {})[cat.key] || 0;
    if (!n) continue;
    var row2 = document.createElement('div');
    row2.className = 'row';
    row2.innerHTML = '<span class="dot" style="background:' + cat.css + '"></span>'
      + '<span class="name">' + cat.label + '</span><span class="n">' + fmt(n) + '</span>';
    legend.appendChild(row2);
  }

  var s = msg.cloud_stats || {};
  el('ck-note').textContent =
    fmt(s.n_no_soma) + ' neurons have no soma in the imaged volume (all ' + fmt(s.n_orn)
    + ' ORNs among them — their cell bodies sit in the antenna) and ' + fmt(s.n_outside_brain_box)
    + ' fall outside the brain\u2019s bounding box, mostly ascending neurons whose somata are '
    + 'in the ventral nerve cord. None are drawn; spikes in them still propagate.';

  var ticks = el('ck-ticks');
  ticks.textContent = '';
  for (var k = 0; k < msg.substeps; k++) {
    var t = document.createElement('div');
    t.className = 'tick';
    ticks.appendChild(t);
  }
  this._onStatus(msg);
};

/* Paint the four plastic-mode populations into the point cloud's colour
 * attribute. Their somata are a few hundred points among 123,930, so without
 * this the neurons the panel is about are invisible on screen. */
Client.prototype._applyHighlight = function () {
  if (!this.highlight || !this.brain.ready) return;
  var b = this.brain, h = this.highlight;
  ['approach', 'avoid', 'pam', 'ppl1'].forEach(function (key) {
    if (h[key] && h[key].length) b.recolor(h[key], HILITE[key].rgb, HILITE[key].size);
  });
};

Client.prototype._onStatus = function (msg) {
  this.status = msg;
  this.game.enable(msg.game);
  this.waveMs = clamp(msg.interval_ms * 0.085, 45, 125);
  el('btn-play').textContent = msg.paused ? 'PLAY' : 'PAUSE';
  el('btn-play').className = msg.paused ? '' : 'primary';
  if (el('sel-cond').value !== msg.condition) el('sel-cond').value = msg.condition;
  el('st-ep').textContent = msg.episodes;
  el('st-wl').textContent = msg.wins + '/' + msg.losses;
  el('st-act').textContent = msg.actions;
  el('st-ms').textContent = Math.round(msg.brain_ms);
  el('st-rss').textContent = Math.round(msg.rss_mb);
  el('foot-text').textContent = msg.footer;
  // The plastic and real-game footers are several claims each, not a label;
  // they have to be readable, not ellipsised on one line.
  el('footline').className =
    (msg.policy_mode === 'plastic' || msg.source === 'realgame') ? 'wrap' : '';

  var inLoop = msg.brain_in_loop;
  var plastic = msg.policy_mode === 'plastic';
  var realgame = msg.source === 'realgame';
  // Wiring is not switchable in the plastic mode (a degree-preserving shuffle
  // deletes the KC -> MBON pathway rather than rewiring it), it is not
  // switchable at all while following a real game (that game ran on whatever
  // wiring play.py had), and learning is not switchable in any other.
  el('sel-cond').hidden = plastic || realgame;
  el('lbl-cond').hidden = plastic || realgame;
  el('sel-learn').hidden = !plastic;
  el('lbl-learn').hidden = !plastic;
  // Episodes / wins / losses are the headless loop's bookkeeping. A real run
  // has none of it in the per-decision log, so the counters say so.
  if (realgame) {
    var rg = msg.realgame || {};
    el('st-ep').textContent = '\u2014';
    el('st-wl').textContent = '\u2014';
    el('st-act').textContent = rg.decisions || 0;
    // pools_match is null until the first record carrying a synapse count
    // arrives, false if this process built a different fly than the one the
    // recording was made on -- in which case the somata flashing in the cloud
    // are the wrong neurons and the header has to say so.
    var pools = rg.pools_match === false
      ? ' \u00b7 <span class="warn">brain does not match the recording</span>'
      : '';
    el('head-mode').innerHTML =
      'source <b>real Balatro</b> \u00b7 ' + (rg.live ? 'live feed' : 'replay')
      + (rg.replay_decisions ? ' (' + rg.replay_decisions + ' hands)' : '')
      + ' \u00b7 spikes <b>' + (rg.spike_mode || '\u2014') + '</b>'
      + ' \u00b7 ' + (rg.recorded || 0) + ' recorded / '
      + (rg.resimulated || 0) + ' re-simulated'
      + (rg.emit_ms ? ' \u00b7 frame ' + rg.emit_ms + ' ms' : '')
      + pools;
    el('ro-title').textContent = 'the action the fly took in the real game';
    el('ro-sub').textContent = msg.policy_input;
  } else if (plastic) {
    var on = !!msg.learning;
    if (el('sel-learn').value !== (on ? 'on' : 'off')) el('sel-learn').value = on ? 'on' : 'off';
    el('head-mode').innerHTML =
      'live learning <b>' + msg.plastic_version + '</b>'
      + (msg.plastic_homeostasis ? ' (KC homeostasis)' : ' (no homeostasis)')
      + ' · learning <b class="' + (on ? 'lrn-on' : 'lrn-off') + '">'
      + (on ? 'ON' : 'off') + '</b> · no readout, the decision is its own MBONs';
  } else {
    el('head-mode').innerHTML =
      'policy <b>' + msg.policy_mode + '</b> · wiring <b>' + msg.condition + '</b>'
      + (inLoop
          ? ' · readout reads ' + msg.policy_input
          : ' · <span class="warn">brain is not in the decision path</span>');
    el('ro-title').textContent = inLoop
      ? 'trained readout · top 5 legal'
      : 'action scores · top 5 legal';
    el('ro-sub').textContent = msg.policy_label;
  }

  var caveat = el('caveat');
  if (msg.caveat) {
    caveat.className = 'on';
    caveat.innerHTML = '<b>What you are watching:</b> ' + msg.caveat;
  } else {
    caveat.className = '';
    caveat.textContent = '';
  }
};

Client.prototype._onState = function (msg) {
  this._flush();
  this.pendingSubsteps = [];
  this.brain.clearFlashes();
  this._setTicks(-1);
  el('ck-spikes').textContent = '—';
  el('ck-label').textContent = 'driving ' + msg.n_bits_on + ' feature channels into the antennal lobe';
  this.panel.state(msg);
  this.panel.dopamineClear();
  // The boxes and the headline are the harness's hand analysis, which exists
  // before the fly runs -- they can go up now. The PLAY/DIG verdict is the
  // fly's own output and is held back to _onDecision, so the game panel cannot
  // announce the decision before the cloud has finished firing it.
  this.game.setPov(msg.pov || null);
  this.game.setVerdict('');
  var running = 'running 50 ms\u2026';
  if (this.panel.plastic) el('mb-sub').textContent = running;
  else el('ro-sub').textContent = running;
};

/* Two kinds of binary frame share this socket. A spike sub-step's first word
 * is its index, 0-4; a captured game frame's is GAME_TAG, which is not in that
 * range. Dispatching on it first is what keeps a JPEG from being pushed into
 * pendingSubsteps as a sixth sub-step and corrupting the 50 ms wave. */
Client.prototype._onBinary = function (buffer) {
  var head = new Uint32Array(buffer, 0, 2);
  if (head[0] === GAME_TAG) return this.game.onFrame(buffer);
  var count = head[1];
  this.pendingSubsteps.push({
    index: head[0],
    points: count ? new Uint32Array(buffer, 8, count) : new Uint32Array(0)
  });
};

Client.prototype._onDecision = function (msg) {
  var self = this, subs = this.pendingSubsteps.slice();
  this.pendingSubsteps = [];
  var iv = this.waveMs;

  var stepMs = self.hello ? self.hello.substep_ms : 10;
  subs.forEach(function (sub, k) {
    self._at(k * iv, function () {
      self.brain.flash(sub.points);
      self._setTicks(sub.index);
      el('ck-spikes').textContent = fmt(sub.points.length);
      el('ck-label').textContent = 'somata firing between t = '
        + Math.round(sub.index * stepMs) + ' and ' + Math.round((sub.index + 1) * stepMs)
        + ' ms of the 50 ms window';
    });
  });

  this._at(subs.length * iv + 60, function () {
    self.panel.decision(msg, msg.has_logits);
    self.game.setVerdict(action_word(msg.action_name));
    if (msg.mb) {
      self.panel.mb(msg.mb);
      // The headless loop resolves the hand inside the same record, so its
      // pulse arrives here. The real game logs the pulse on its own record and
      // it arrives at _onOutcome instead; either way it is the last thing to
      // light up, because it happens after the window ran.
      self._pulse(msg.mb.dopamine);
    }
    if (self.status && !msg.mb) el('ro-sub').textContent = self.status.policy_label;
    el('ck-spikes').textContent = fmt(msg.n_spiking);
    // Following a real game, say where the wave came from on every decision:
    // "recorded" is what the fly's own window did, "re-simulated" is the same
    // 32 relay bits driven through the same frozen brain again, and "none"
    // means the record has neither.
    if (msg.spike_mode === 'none') {
      el('ck-label').textContent = 'no spikes for this decision — the record has '
        + 'no sub-step sidecar and the relay block was silent';
    } else {
      el('ck-label').textContent = 'distinct neurons fired · ' + fmt(msg.total_spikes)
        + ' spikes total · ' + msg.brain_ms + ' ms to simulate 50 ms'
        + (msg.spike_mode ? ' · spikes ' + msg.spike_mode : '');
    }
    self._setTicks(-1);
  });
};

/* What the real game paid for the hand the fly just decided, and the dopamine
 * pulse that bought. No board and no spikes: nothing new fired, so the cloud
 * keeps the window it is already showing and only the pulse is added to it. */
Client.prototype._onOutcome = function (msg) {
  if (!msg.mb) return;
  var self = this, o = msg.outcome || {};
  // Queued, not applied now. The decision's own panel update is deferred until
  // its wave has played, and at 4x the server can send the outcome before that
  // lands -- applying the pulse first would then be overwritten by the
  // pre-pulse weights and the strip chart would step backwards for a beat.
  // _at is FIFO behind a time gate, so this runs after whatever is already
  // queued for the decision it belongs to.
  this._at(0, function () {
    self.game.clearBoxes();
    self.panel.mb(msg.mb);
    if (msg.dopamine) {
      self._toast(
        msg.dopamine === 'reward' ? 'DOPAMINE · REWARD' : 'DOPAMINE · PUNISHMENT',
        fmt(msg.mb.dopamine_changed) + ' of ' + fmt(msg.mb.weights.n_edges)
          + ' KC \u2192 MBON synapses depressed'
          + (o.chips_gained ? ' \u00b7 ' + fmt(o.chips_gained) + ' chips' : ''),
        1600,
        msg.dopamine === 'reward' ? 'win' : 'lose'
      );
    }
  });
  this._pulse(msg.mb.dopamine);
};

/* Flash the dopamine cluster that actually released. Once is invisible: the
 * glow e-folds in 260 ms, so a single flash is one frame of a clip. Three
 * spaced re-flashes hold it for about a second without turning it into an
 * animation the data does not have -- it is the same pulse, drawn long enough
 * to be seen. */
Client.prototype._pulse = function (kind) {
  var self = this, h = this.highlight;
  if (!h || (kind !== 'reward' && kind !== 'punish')) return;
  var points = kind === 'reward' ? h.pam : h.ppl1;
  if (!points || !points.length) return;
  [0, 220, 440].forEach(function (ms) {
    self._at(ms, function () { self.brain.flash(points); });
  });
};

Client.prototype._setTicks = function (active) {
  var ticks = el('ck-ticks').children;
  for (var i = 0; i < ticks.length; i++) ticks[i].className = 'tick' + (i <= active ? ' on' : '');
};

Client.prototype._onEpisode = function (msg) {
  var why = msg.truncated ? 'hit the step cap' : (msg.won ? 'cleared the ante' : 'ran out of hands');
  this._toast(
    msg.won ? 'RUN WON' : 'RUN LOST',
    why + ' · ' + fmt(msg.chips_total) + ' chips over ' + msg.steps + ' actions · seed ' + msg.seed,
    1900,
    msg.won ? 'win' : 'lose'
  );
};

Client.prototype._toast = function (title, sub, ms, cls) {
  var node = el('toast');
  node.innerHTML = title + (sub ? '<span class="s">' + sub + '</span>' : '');
  node.className = 'on ' + (cls || '');
  if (this.toastTimer) clearTimeout(this.toastTimer);
  this.toastTimer = setTimeout(function () { node.className = cls || ''; }, ms || 1500);
};

window.addEventListener('load', function () {
  if (typeof THREE === 'undefined') {
    el('ck-note').textContent = 'three.js failed to load from cdnjs; the brain panel cannot render.';
    console.error('three.js missing');
    return;
  }
  window.flyClient = new Client();
});
