// Interactive 3D viewer for one ToothFairy3 CBCT case (ground-truth labels as surface meshes).
// Data is generated offline by export_cbct_meshes.py (kept outside this repo; it documents the .bin layout).
// Usage: <div class="cv" data-src="/assets/data/toothfairy3_F001.bin"></div>
//        <script type="module" src="/assets/js/cbct_viewer.js"></script>
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/+esm';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js/+esm';

// Tooth colours = tooth type (second FDI digit), same palette as the Teeth3DS+ page.
const TYPE_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
const TYPE_NAMES = ['Central incisor', 'Lateral incisor', 'Canine', 'First premolar',
                    'Second premolar', 'First molar', 'Second molar', 'Third molar'];
const QUADRANTS = { 1: 'upper right', 2: 'upper left', 3: 'lower left', 4: 'lower right' };
const BONE = '#e6dfcc', AIR = '#5aa9e6', RESTO = '#aeb6bf', CANAL = '#ffd23f', PULP = '#ff5d73';
const GHOST = '#d9d4c7';
const SURFACE = 0x151514;

const NAMES = {
  1: 'Lower jawbone', 2: 'Upper jawbone',
  3: 'Left inferior alveolar canal', 4: 'Right inferior alveolar canal',
  5: 'Left maxillary sinus', 6: 'Right maxillary sinus', 7: 'Pharynx',
  8: 'Bridge', 9: 'Crown', 10: 'Implant',
  103: 'Left mandibular incisive canal', 104: 'Right mandibular incisive canal', 105: 'Lingual canal',
};

function kind(l) {
  if (l === 1 || l === 2) return 'jaw';
  if (l >= 5 && l <= 7) return 'air';
  if (l >= 8 && l <= 10) return 'resto';
  if (l === 3 || l === 4 || (l >= 103 && l <= 105)) return 'canal';
  if (l > 110) return 'pulp';
  return 'tooth';
}

function labelName(l) {
  if (NAMES[l]) return NAMES[l];
  const fdi = l > 100 ? l - 100 : l;
  const q = Math.floor(fdi / 10), t = fdi % 10;
  const tooth = `FDI ${fdi} · ${QUADRANTS[q]} ${TYPE_NAMES[t - 1].toLowerCase()}`;
  return l > 100 ? `Pulp of ${tooth}` : tooth;
}

const CSS = `
.cv { --cv-surface: #151514; --cv-ink: #ffffff; --cv-ink-2: #c3c2b7; --cv-muted: #898781;
  --cv-border: rgba(255,255,255,0.14); --cv-btn-border: #4a4945; --cv-btn-hover: #2c2c2a;
  margin: 1.5rem 0; font-size: 15px; }
.cv-toolbar { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.cv-btn { font: inherit; font-size: 14px; padding: 6px 12px; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--cv-btn-border); background: transparent; color: var(--cv-ink); line-height: 1.3; }
.cv-btn:hover { background: var(--cv-btn-hover); }
.cv-btn[aria-pressed="true"] { background: var(--cv-ink); border-color: var(--cv-ink); color: #0b0b0b; }
.cv-btn:focus-visible { outline: 2px solid #3987e5; outline-offset: 2px; }
.cv-stage { position: relative; width: 100%; height: clamp(300px, 62vw, 540px);
  background: var(--cv-surface); border: 1px solid var(--cv-border); border-radius: 8px; overflow: hidden;
  touch-action: none; }
.cv-stage canvas { display: block; width: 100%; height: 100%; cursor: grab; }
.cv-stage canvas:active { cursor: grabbing; }
.cv-status { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  color: var(--cv-ink-2); font-size: 14px; padding: 16px; text-align: center; }
.cv-corner { position: absolute; right: 10px; bottom: 10px; display: flex; gap: 6px; }
.cv-corner .cv-btn { font-size: 13px; padding: 4px 10px; background: var(--cv-surface); }
.cv-corner .cv-btn[aria-pressed="true"] { background: var(--cv-ink); }
.cv-hint { position: absolute; left: 12px; bottom: 12px; font-size: 12px; color: var(--cv-muted); pointer-events: none; }
.cv-tip { position: absolute; pointer-events: none; background: #ffffff; color: #0b0b0b; font-size: 13px;
  padding: 4px 8px; border-radius: 4px; white-space: nowrap; transform: translate(12px, -50%); display: none; }
.cv-caption { margin: 10px 0 6px; color: var(--cv-ink-2); font-size: 15px; line-height: 1.5; min-height: 3em; }
.cv-legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--cv-ink-2); }
.cv-legend span { display: inline-flex; align-items: center; gap: 6px; }
.cv-legend i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
@media (max-width: 575px) { .cv-hint { display: none; } }
`;

function parse(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'TF3M') throw new Error('bad file');
  const n = dv.getUint32(8, true);
  let o = 16;
  const heads = [];
  for (let i = 0; i < n; i++) {
    heads.push({ label: dv.getUint16(o, true), nv: dv.getUint32(o + 4, true), nt: dv.getUint32(o + 8, true) });
    o += 12;
  }
  return heads.map(h => {
    const pos = new Int16Array(buf, o, h.nv * 3); o += h.nv * 6;
    const nrm = new Int8Array(buf, o, h.nv * 3); o += h.nv * 3;
    o += (4 - (o % 4)) % 4;
    const big = h.nv >= 65536;
    const idx = big ? new Uint32Array(buf, o, h.nt * 3) : new Uint16Array(buf, o, h.nt * 3);
    o += h.nt * 3 * (big ? 4 : 2);
    o += (4 - (o % 4)) % 4;
    const p = new Float32Array(pos.length);
    for (let i = 0; i < pos.length; i++) p[i] = pos[i] / 32767;
    const nn = new Float32Array(nrm.length);
    for (let i = 0; i < nrm.length; i++) nn[i] = nrm[i] / 127;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nn, 3));
    g.setIndex(new THREE.BufferAttribute(big ? new Uint32Array(idx) : new Uint16Array(idx), 1));
    g.computeBoundingSphere();
    return { label: h.label, geometry: g };
  });
}

function init(root) {
  if (!document.getElementById('cv-style')) {
    const style = document.createElement('style');
    style.id = 'cv-style';
    style.textContent = CSS;
    document.head.appendChild(style);
  }
  root.innerHTML = `
    <div class="cv-toolbar" role="group" aria-label="View mode"></div>
    <div class="cv-stage">
      <div class="cv-status">Loading scan…</div>
      <div class="cv-tip" role="status"></div>
      <span class="cv-hint">Drag to rotate · scroll or pinch to zoom</span>
      <div class="cv-corner">
        <button type="button" class="cv-btn cv-rotate" aria-pressed="false">Pause rotation</button>
      </div>
    </div>
    <p class="cv-caption" aria-live="polite"></p>
    <div class="cv-legend"></div>`;
  const $ = s => root.querySelector(s);
  const toolbar = $('.cv-toolbar'), stage = $('.cv-stage'), status = $('.cv-status'), tip = $('.cv-tip');
  const caption = $('.cv-caption'), legend = $('.cv-legend'), rotateBtn = $('.cv-rotate');

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true });
  } catch (e) {
    status.textContent = 'This viewer needs WebGL, which is not available in this browser.';
    $('.cv-corner').remove();
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(SURFACE, 1);
  stage.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x303030, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(0.5, 1, 1.2);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xffffff, 0.6);
  rim.position.set(-1, 0.3, -1);
  scene.add(rim);

  const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 50);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 0.9;
  controls.maxDistance = 8;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  controls.autoRotate = !reduceMotion;
  controls.autoRotateSpeed = 1.0;
  const setRotateLabel = () => { rotateBtn.textContent = controls.autoRotate ? 'Pause rotation' : 'Rotate'; };
  setRotateLabel();
  rotateBtn.addEventListener('click', () => { controls.autoRotate = !controls.autoRotate; setRotateLabel(); });

  fetch(root.dataset.src)
    .then(r => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
    .then(buf => build(parse(buf)))
    .catch(() => { status.textContent = 'Could not load the scan data.'; });

  function build(items) {
    status.remove();
    const group = new THREE.Group();
    scene.add(group);
    const meshes = items.map(it => {
      const mat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.0, side: THREE.DoubleSide });
      const m = new THREE.Mesh(it.geometry, mat);
      m.userData = { label: it.label, kind: kind(it.label), opacity: 1, target: 1, color: new THREE.Color(), targetColor: new THREE.Color() };
      group.add(m);
      return m;
    });
    const present = new Set(items.map(it => it.label));
    const count = k => meshes.filter(m => m.userData.kind === k).length;

    // Frame the teeth and jaws: a front view, slightly from above and to one side.
    const box = new THREE.Box3();
    meshes.filter(m => ['tooth', 'resto', 'canal'].includes(m.userData.kind)).forEach(m => box.expandByObject(m));
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const dist = sphere.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 0.95;
    controls.target.copy(sphere.center);
    camera.position.copy(sphere.center).add(new THREE.Vector3(0.45, 0.35, 1).normalize().multiplyScalar(dist));
    controls.update();

    const typeColor = l => TYPE_COLORS[(l % 10) - 1];
    const fmt = n => n.toLocaleString('en-GB');
    const dot = (c, t) => `<span><i style="background:${c}"></i>${t}</span>`;
    const presentTypes = [...new Set(items.filter(it => kind(it.label) === 'tooth').map(it => it.label % 10))].sort();
    const typeLegend = presentTypes.map(t => dot(TYPE_COLORS[t - 1], TYPE_NAMES[t - 1])).join('');

    // Each mode returns [colour, opacity] for a mesh label.
    const modes = {
      all: {
        label: 'All structures',
        caption: `All ${present.size} labelled structures in this scan, from the dataset's ground truth. Bone is see-through so the teeth and the nerve canals inside it stay visible; the blue volumes are air spaces (sinuses and pharynx). Hover over a structure to see its name.`,
        legend: dot(BONE, 'Jawbone') + dot(AIR, 'Sinuses and pharynx') + dot(CANAL, 'Nerve canals') + dot(RESTO, 'Bridge, crown, implant') + typeLegend,
        style: l => ({ jaw: [BONE, 0.22], air: [AIR, 0.2], resto: [RESTO, 1], canal: [CANAL, 1], pulp: [PULP, 0], tooth: [typeColor(l), 1] })[kind(l)],
      },
      teeth: {
        label: 'Teeth',
        caption: `${count('tooth')} teeth, each its own class (FDI number), plus a bridge, a crown and an implant, which are labelled as separate classes. Colour shows tooth type, so left and right share a colour.`,
        legend: typeLegend + dot(RESTO, 'Bridge, crown, implant'),
        style: l => ({ jaw: [GHOST, 0.07], air: [AIR, 0], resto: [RESTO, 1], canal: [CANAL, 0], pulp: [PULP, 0], tooth: [typeColor(l), 1] })[kind(l)],
      },
      inside: {
        label: 'Pulp and nerve canals',
        caption: `What ToothFairy3 added: the pulp inside every tooth (${count('pulp')} in this scan) and three thin canals at the front of the lower jaw (left and right incisive canals, lingual canal), on top of the inferior alveolar canals that were already labelled. Teeth are shown see-through.`,
        legend: dot(PULP, 'Pulp') + dot(CANAL, 'Nerve canals') + dot(GHOST, 'Teeth (see-through)'),
        style: l => ({ jaw: [GHOST, 0.05], air: [AIR, 0], resto: [RESTO, 0.12], canal: [CANAL, 1], pulp: [PULP, 1], tooth: [GHOST, 0.16] })[kind(l)],
      },
    };

    let current = 'all';
    const buttons = Object.entries(modes).map(([id, m]) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'cv-btn';
      b.textContent = m.label;
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', () => setMode(id));
      toolbar.appendChild(b);
      return [id, b];
    });

    function apply() {
      const m = modes[current];
      meshes.forEach(mesh => {
        const [c, o] = m.style(mesh.userData.label);
        mesh.userData.targetColor.set(c);
        mesh.userData.target = o;
      });
      caption.textContent = m.caption;
      legend.innerHTML = m.legend;
    }
    function setMode(id) {
      current = id;
      buttons.forEach(([bid, b]) => b.setAttribute('aria-pressed', String(bid === id)));
      apply();
      tip.style.display = 'none';
    }
    setMode('all');
    meshes.forEach(mesh => {  // start at the target state, no fade-in
      mesh.userData.color.copy(mesh.userData.targetColor);
      mesh.userData.opacity = mesh.userData.target;
    });

    const stepStyle = () => {
      meshes.forEach(mesh => {
        const u = mesh.userData;
        u.opacity += (u.target - u.opacity) * 0.12;
        if (Math.abs(u.target - u.opacity) < 0.003) u.opacity = u.target;
        u.color.lerp(u.targetColor, 0.12);
        const mat = mesh.material;
        mat.color.copy(u.color);
        mat.opacity = u.opacity;
        mat.transparent = u.opacity < 0.999;
        mat.depthWrite = u.opacity > 0.6;
        mesh.visible = u.opacity > 0.01;
        mesh.renderOrder = mat.transparent ? 1 : 0;
      });
    };

    // Hover: first opaque-enough structure under the cursor.
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let hoverEvent = null;
    renderer.domElement.addEventListener('pointermove', e => { hoverEvent = e; });
    renderer.domElement.addEventListener('pointerleave', () => { hoverEvent = null; tip.style.display = 'none'; });
    const updateHover = () => {
      if (!hoverEvent) return;
      const e = hoverEvent;
      hoverEvent = null;
      if (e.buttons) { tip.style.display = 'none'; return; }
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(mouse, camera);
      const pickable = meshes.filter(m => m.visible && m.userData.target >= 0.15);
      const hit = raycaster.intersectObjects(pickable, false)[0];
      if (!hit) { tip.style.display = 'none'; return; }
      const l = hit.object.userData.label;
      tip.textContent = labelName(l);
      tip.style.left = `${e.clientX - rect.left}px`;
      tip.style.top = `${e.clientY - rect.top}px`;
      tip.style.display = 'block';
      if (tip.offsetLeft + tip.offsetWidth + 16 > stage.clientWidth) {
        tip.style.left = `${stage.clientWidth - tip.offsetWidth - 16}px`;
      }
    };

    const resize = () => {
      const w = stage.clientWidth, h = stage.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    new ResizeObserver(resize).observe(stage);
    resize();

    const loop = () => {
      controls.update();
      stepStyle();
      updateHover();
      renderer.render(scene, camera);
    };
    new IntersectionObserver(([entry]) => {
      renderer.setAnimationLoop(entry.isIntersecting ? loop : null);
    }).observe(stage);
  }
}

document.querySelectorAll('.cv[data-src]').forEach(init);
