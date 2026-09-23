// Interactive point-cloud viewer for a Teeth3DS+ scan.
// Data is generated offline by export_viewer_data.py (kept outside this repo; it documents the .bin layout).
// Usage: <div class="tv" data-src="/assets/data/teeth3ds_<id>.bin"></div>
//        <script type="module" src="/assets/js/teeth_viewer.js"></script>
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/+esm';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js/+esm';

// Colour = tooth type (second FDI digit). Categorical order is fixed so that
// neighbouring teeth along the arch get neighbouring, CVD-separated slots.
// Steps are the dark-surface variants (the site has a black page).
const TYPE_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
const TYPE_NAMES = ['Central incisor', 'Lateral incisor', 'Canine', 'First premolar',
                    'Second premolar', 'First molar', 'Second molar', 'Third molar'];
const QUADRANTS = { 1: 'upper right', 2: 'upper left', 3: 'lower left', 4: 'lower right' };
const GINGIVA = '#9a968d';
const PLAIN = '#d9d4c7';
const BOUNDARY = '#f2f1ed';
const OTHER = '#5e5c57';
const SURFACE = 0x151514;
const SHIFT_AMOUNT = 0.8;

const MODES = [
  { id: 'scan', label: 'Scan' },
  { id: 'labels', label: 'Tooth labels' },
  { id: 'shift', label: 'Shift to centres' },
  { id: 'boundary', label: 'Boundary-aware sample' },
];

const CSS = `
.tv { --tv-surface: #151514; --tv-ink: #ffffff; --tv-ink-2: #c3c2b7; --tv-muted: #898781;
  --tv-border: rgba(255,255,255,0.14); --tv-btn-border: #4a4945; --tv-btn-hover: #2c2c2a;
  margin: 1.5rem 0; font-size: 15px; }
.tv-toolbar { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.tv-btn { font: inherit; font-size: 14px; padding: 6px 12px; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--tv-btn-border); background: transparent; color: var(--tv-ink); line-height: 1.3; }
.tv-btn:hover { background: var(--tv-btn-hover); }
.tv-btn[aria-pressed="true"] { background: var(--tv-ink); border-color: var(--tv-ink); color: #0b0b0b; }
.tv-btn:focus-visible { outline: 2px solid #3987e5; outline-offset: 2px; }
.tv-stage { position: relative; width: 100%; height: clamp(300px, 62vw, 520px);
  background: var(--tv-surface); border: 1px solid var(--tv-border); border-radius: 8px; overflow: hidden;
  touch-action: none; }
.tv-stage canvas { display: block; width: 100%; height: 100%; cursor: grab; }
.tv-stage canvas:active { cursor: grabbing; }
.tv-status { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  color: var(--tv-ink-2); font-size: 14px; padding: 16px; text-align: center; }
.tv-rotate { position: absolute; right: 10px; bottom: 10px; font-size: 13px; padding: 4px 10px;
  background: var(--tv-surface); }
.tv-hint { position: absolute; left: 12px; bottom: 12px; font-size: 12px; color: var(--tv-muted); pointer-events: none; }
.tv-tip { position: absolute; pointer-events: none; background: #ffffff; color: #0b0b0b; font-size: 13px;
  padding: 4px 8px; border-radius: 4px; white-space: nowrap; transform: translate(12px, -50%); display: none; }
.tv-caption { margin: 10px 0 6px; color: var(--tv-ink-2); font-size: 15px; line-height: 1.5; min-height: 3em; }
.tv-legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--tv-ink-2); }
.tv-legend span { display: inline-flex; align-items: center; gap: 6px; }
.tv-legend i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
`;

const VERT = `
attribute vec3 aColor;
attribute float aVisible;
uniform float uSize;
varying vec3 vColor;
varying float vShade;
varying float vVisible;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vec3 n = normalize(normalMatrix * normal);
  float diffuse = max(dot(n, normalize(vec3(0.3, 0.7, 0.65))), 0.0);
  vShade = 0.45 + 0.6 * diffuse;
  vColor = aColor;
  vVisible = aVisible;
  gl_PointSize = uSize / -mv.z;
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = `
varying vec3 vColor;
varying float vShade;
varying float vVisible;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  if (vVisible < 0.5 || dot(c, c) > 0.25) discard;
  gl_FragColor = vec4(min(vColor * vShade, 1.0), 1.0);
}`;

function hexToRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16 & 255) / 255, (v >> 8 & 255) / 255, (v & 255) / 255];
}

function toothName(fdi) {
  if (!fdi) return 'Gingiva';
  const q = Math.floor(fdi / 10), t = fdi % 10;
  return `FDI ${fdi} · ${QUADRANTS[q] || ''} ${TYPE_NAMES[t - 1].toLowerCase()}`;
}

function parse(buf) {
  const head = new DataView(buf, 0, 16);
  const nA = head.getUint32(0, true), nB = head.getUint32(4, true);
  const nMesh = head.getUint32(12, true);
  let o = 16;
  const take = (Type, n) => { const a = new Type(buf, o, n); o += n * Type.BYTES_PER_ELEMENT; return a; };
  const posA = take(Int16Array, nA * 3), posB = take(Int16Array, nB * 3);
  const nrmA = take(Int8Array, nA * 3), nrmB = take(Int8Array, nB * 3);
  const labA = take(Uint8Array, nA), labB = take(Uint8Array, nB), flagB = take(Uint8Array, nB);
  // Data axes: x = left/right, y = towards the back, z = occlusal (up). three.js: y up, z towards viewer.
  const toThree = (src, s) => {
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i += 3) {
      out[i] = src[i] * s; out[i + 1] = src[i + 2] * s; out[i + 2] = -src[i + 1] * s;
    }
    return out;
  };
  return {
    nMesh,
    A: { pos: toThree(posA, 1 / 32767), nrm: toThree(nrmA, 1 / 127), lab: labA },
    B: { pos: toThree(posB, 1 / 32767), nrm: toThree(nrmB, 1 / 127), lab: labB, flag: flagB },
  };
}

function toothCentres(pos, lab) {
  const sum = new Map();
  for (let i = 0; i < lab.length; i++) {
    if (!lab[i]) continue;
    const s = sum.get(lab[i]) || [0, 0, 0, 0];
    s[0] += pos[3 * i]; s[1] += pos[3 * i + 1]; s[2] += pos[3 * i + 2]; s[3]++;
    sum.set(lab[i], s);
  }
  const centres = new Map();
  for (const [k, s] of sum) centres.set(k, [s[0] / s[3], s[1] / s[3], s[2] / s[3]]);
  return centres;
}

function makeGeometry(set) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(set.pos.slice(), 3));
  g.setAttribute('normal', new THREE.BufferAttribute(set.nrm, 3));
  g.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(set.lab.length * 3), 3));
  g.setAttribute('aVisible', new THREE.BufferAttribute(new Float32Array(set.lab.length).fill(1), 1));
  g.computeBoundingSphere();
  return g;
}

function init(root) {
  if (!document.getElementById('tv-style')) {
    const style = document.createElement('style');
    style.id = 'tv-style';
    style.textContent = CSS;
    document.head.appendChild(style);
  }
  root.innerHTML = `
    <div class="tv-toolbar" role="group" aria-label="View mode"></div>
    <div class="tv-stage">
      <div class="tv-status">Loading scan…</div>
      <div class="tv-tip" role="status"></div>
      <span class="tv-hint">Drag to rotate · scroll or pinch to zoom</span>
      <button type="button" class="tv-btn tv-rotate" aria-pressed="false">Pause rotation</button>
    </div>
    <p class="tv-caption" aria-live="polite"></p>
    <div class="tv-legend"></div>`;
  const toolbar = root.querySelector('.tv-toolbar');
  const stage = root.querySelector('.tv-stage');
  const status = root.querySelector('.tv-status');
  const tip = root.querySelector('.tv-tip');
  const caption = root.querySelector('.tv-caption');
  const legend = root.querySelector('.tv-legend');
  const rotateBtn = root.querySelector('.tv-rotate');

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true });
  } catch (e) {
    status.textContent = 'This viewer needs WebGL, which is not available in this browser.';
    rotateBtn.remove();
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(SURFACE, 1);
  stage.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 0.8;
  controls.maxDistance = 8;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  controls.autoRotate = !reduceMotion;
  controls.autoRotateSpeed = 1.2;
  const setRotateLabel = () => {
    rotateBtn.textContent = controls.autoRotate ? 'Pause rotation' : 'Rotate';
  };
  setRotateLabel();
  rotateBtn.addEventListener('click', () => { controls.autoRotate = !controls.autoRotate; setRotateLabel(); });

  const material = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms: { uSize: { value: 10 } } });

  fetch(root.dataset.src)
    .then(r => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
    .then(buf => build(parse(buf)))
    .catch(() => { status.textContent = 'Could not load the scan data.'; });

  function build(data) {
    status.remove();
    const geoA = makeGeometry(data.A), geoB = makeGeometry(data.B);
    const pointsA = new THREE.Points(geoA, material), pointsB = new THREE.Points(geoB, material);
    scene.add(pointsA, pointsB);
    const centres = toothCentres(data.A.pos, data.A.lab);
    const nBoundary = data.B.flag.reduce((a, b) => a + b, 0);
    const presentTypes = [...new Set([...data.A.lab].filter(l => l).map(l => l % 10))].sort();
    const nTeeth = centres.size;
    const fmt = n => n.toLocaleString('en-GB');

    // Frame the teeth (the scan's flat base plate is much wider): a front-top view,
    // incisors towards the camera.
    const toothPts = [];
    for (let i = 0; i < data.A.lab.length; i++) {
      if (data.A.lab[i]) toothPts.push(new THREE.Vector3().fromArray(data.A.pos, 3 * i));
    }
    const teethSphere = new THREE.Sphere().setFromPoints(toothPts);
    const dist = teethSphere.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.05;
    controls.target.copy(teethSphere.center);
    camera.position.copy(teethSphere.center).add(new THREE.Vector3(0, 0.78, 0.62).multiplyScalar(dist));
    controls.update();

    const paint = (geo, fn) => {
      const col = geo.attributes.aColor.array, vis = geo.attributes.aVisible.array;
      for (let i = 0; i < vis.length; i++) {
        const [hex, v] = fn(i);
        const rgb = hexToRgb(hex);
        col[3 * i] = rgb[0]; col[3 * i + 1] = rgb[1]; col[3 * i + 2] = rgb[2];
        vis[i] = v;
      }
      geo.attributes.aColor.needsUpdate = true;
      geo.attributes.aVisible.needsUpdate = true;
    };
    const typeColor = fdi => (fdi ? TYPE_COLORS[(fdi % 10) - 1] : GINGIVA);

    const typeLegend = presentTypes.map(t =>
      `<span><i style="background:${TYPE_COLORS[t - 1]}"></i>${TYPE_NAMES[t - 1]}</span>`).join('');
    const gingivaLegend = `<span><i style="background:${GINGIVA}"></i>Gingiva</span>`;

    const modes = {
      scan: {
        caption: `The ${fmt(data.A.lab.length)} points the network sees, picked from the scan's ${fmt(data.nMesh)} vertices by farthest point sampling. Each point carries 6 numbers: its position (x, y, z) and its surface normal.`,
        legend: '',
        apply() { paint(geoA, () => [PLAIN, 1]); },
        points: pointsA, hover: false,
      },
      labels: {
        caption: `Ground-truth labels from Teeth3DS+: ${nTeeth} teeth plus gingiva. Colour shows the tooth type, so left and right share a colour. Hover over a point to see its FDI number.`,
        legend: typeLegend + gingivaLegend,
        apply() { paint(geoA, i => [typeColor(data.A.lab[i]), 1]); },
        points: pointsA, hover: true,
      },
      shift: {
        caption: `What the offset head learns: every tooth point is pushed towards the centre of its tooth, so each tooth becomes a tight cluster that DBSCAN can pick out. Gingiva is filtered out. (These are ground-truth offsets at ${Math.round(SHIFT_AMOUNT * 100)}% length, so each cluster keeps a little shape.)`,
        legend: typeLegend,
        apply() { paint(geoA, i => [typeColor(data.A.lab[i]), data.A.lab[i] ? 1 : 0]); },
        points: pointsA, hover: true, shift: SHIFT_AMOUNT,
      },
      boundary: {
        caption: `Stage 2 input: another ${fmt(data.B.lab.length)} points, but ${fmt(nBoundary)} of them are on boundaries, where nearby labels disagree. Only ${fmt(data.B.lab.length - nBoundary)} cover everything else. Here the boundaries come from ground-truth labels. ToothGroupNet finds them from its stage-1 predictions.`,
        legend: `<span><i style="background:${BOUNDARY}"></i>Boundary points (${fmt(nBoundary)})</span><span><i style="background:${OTHER}"></i>Other points (${fmt(data.B.lab.length - nBoundary)})</span>`,
        apply() { paint(geoB, i => [data.B.flag[i] ? BOUNDARY : OTHER, 1]); },
        points: pointsB, hover: true,
      },
    };

    // Shift animation: lerp FPS positions towards their tooth centre on the CPU,
    // so hover picking keeps working on the displayed positions.
    const basePos = data.A.pos, livePos = geoA.attributes.position.array;
    const targetPos = new Float32Array(basePos.length);
    for (let i = 0; i < data.A.lab.length; i++) {
      const c = centres.get(data.A.lab[i]);
      for (let k = 0; k < 3; k++) targetPos[3 * i + k] = c ? c[k] : basePos[3 * i + k];
    }
    let shiftNow = 0, shiftGoal = 0;
    const stepShift = () => {
      if (Math.abs(shiftGoal - shiftNow) < 1e-3) return;
      shiftNow += (shiftGoal - shiftNow) * 0.08;
      if (Math.abs(shiftGoal - shiftNow) < 1e-3) shiftNow = shiftGoal;
      for (let i = 0; i < livePos.length; i++) livePos[i] = basePos[i] + (targetPos[i] - basePos[i]) * shiftNow;
      geoA.attributes.position.needsUpdate = true;
      geoA.computeBoundingSphere();
    };

    let current;
    const buttons = MODES.map(m => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'tv-btn';
      b.textContent = m.label;
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', () => setMode(m.id));
      toolbar.appendChild(b);
      return b;
    });
    function setMode(id) {
      current = modes[id];
      buttons.forEach((b, i) => b.setAttribute('aria-pressed', String(MODES[i].id === id)));
      current.apply();
      pointsA.visible = current.points === pointsA;
      pointsB.visible = current.points === pointsB;
      shiftGoal = current.shift || 0;
      caption.textContent = current.caption;
      legend.innerHTML = current.legend;
      tip.style.display = 'none';
    }
    setMode('labels');

    // Hover: nearest displayed point under the cursor.
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let hoverEvent = null;
    renderer.domElement.addEventListener('pointermove', e => { hoverEvent = e; });
    renderer.domElement.addEventListener('pointerleave', () => { hoverEvent = null; tip.style.display = 'none'; });
    const updateHover = () => {
      if (!hoverEvent) return;
      const e = hoverEvent;
      hoverEvent = null;
      if (!current.hover || e.buttons) { tip.style.display = 'none'; return; }
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.params.Points.threshold = camera.position.length() * 0.006;
      raycaster.setFromCamera(mouse, camera);
      const vis = current.points.geometry.attributes.aVisible.array;
      const hit = raycaster.intersectObject(current.points).find(h => vis[h.index] > 0.5);
      if (!hit) { tip.style.display = 'none'; return; }
      const lab = current.points === pointsA ? data.A.lab : data.B.lab;
      let text = toothName(lab[hit.index]);
      if (current.points === pointsB) text = (data.B.flag[hit.index] ? 'Boundary point' : 'Other point') + ' · ' + text;
      tip.textContent = text;
      tip.style.left = `${e.clientX - rect.left}px`;
      tip.style.top = `${e.clientY - rect.top}px`;
      tip.style.display = 'block';
      // Keep the tooltip inside the stage.
      if (tip.offsetLeft + tip.offsetWidth + 16 > stage.clientWidth) {
        tip.style.left = `${stage.clientWidth - tip.offsetWidth - 16}px`;
      }
    };

    const resize = () => {
      const w = stage.clientWidth, h = stage.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      // Point size scales with canvas height so density looks the same on phones and desktops.
      material.uniforms.uSize.value = h * renderer.getPixelRatio() * 0.015;
    };
    new ResizeObserver(resize).observe(stage);
    resize();

    const loop = () => {
      controls.update();
      stepShift();
      updateHover();
      renderer.render(scene, camera);
    };
    // Only animate while the viewer is on screen.
    new IntersectionObserver(([entry]) => {
      renderer.setAnimationLoop(entry.isIntersecting ? loop : null);
    }).observe(stage);
  }
}

document.querySelectorAll('.tv[data-src]').forEach(init);
