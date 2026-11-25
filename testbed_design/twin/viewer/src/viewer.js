// three.js digital-twin viewer: builds the rack, devices, ports and cables from
// the scene JSON and applies view / filter / step / highlight state.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  toThree, dirToThree, cleanRoute, separateLanes, filletRoute, tubeGeometry, portShapes, quatForNormal,
  pointAtFraction,
} from './geom.js';
import {
  drawFace, drawPlain, drawRailFace, stripeTexture, bandTexture, setMaxAnisotropy, lighten, darken, luminance, mix,
} from './textures.js';
import { linkOfBundle, BANDED } from './model.js';

const BG = '#f6f7f9';
const ACCENT = '#ff7a00';
const OFF_RED = '#e0262b';
const DARK_EDGE = '#9a3412';
export const DEFAULT_FOV = 30;
const LINK_PALETTE = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#9a6324',
  '#469990', '#800000', '#808000', '#000075', '#e6a800', '#7b68ee', '#2e8b57', '#c71585',
];

const EAR = 16;
export const CONF_TINT = { estimate: '#f0a830', likely: '#4f9ad8', confirmed: '#3fb56a', other: '#b0b7c0' };
const _tmpV = new THREE.Vector3();
const BODY_INSET = (482.6 - 440) / 2;

export class Viewer {
  constructor(stage, model, opts = {}) {
    this.stage = stage;
    this.model = model;
    this.scene3 = model.scene;
    this.opts = opts;
    this.listeners = {};
    this.state = {
      classes: new Set(model.scene.legend.map((l) => l.cls)),
      showTable: true,
      showBlanks: true,
      showLabels: true,
      showDims: false,
      rackCallAlt: false,
      rackCallAll: false,
      partLabels: true,
      cableLabelMax: 16,
      confTint: false,
      xray: false,
      scenario: 'before',
      step: null,
      highlight: null,
      colorMode: 'class',
    };
    this.dev = new Map(); // id -> entry
    this.portObj = new Map();
    this.cab = new Map();
    this.pickables = [];
    this.labels = new Map();
    this.needsRender = true;
    this._init();
  }

  on(ev, fn) {
    (this.listeners[ev] ||= []).push(fn);
  }
  emit(ev, ...a) {
    for (const fn of this.listeners[ev] || []) fn(...a);
  }

  _init() {
    const r = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: false });
    r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    r.setClearColor(BG, 1);
    r.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer = r;
    setMaxAnisotropy(Math.min(8, r.capabilities.getMaxAnisotropy()));
    this.stage.appendChild(r.domElement);
    r.domElement.className = 'twin-canvas';

    this.labelLayer = document.createElement('div');
    this.labelLayer.className = 'label-layer';
    this.stage.appendChild(this.labelLayer);

    const s = new THREE.Scene();
    s.background = new THREE.Color(BG);
    s.fog = new THREE.Fog(BG, 14000, 30000);
    this.three = s;

    this.camera = new THREE.PerspectiveCamera(DEFAULT_FOV, 1, 5, 60000);
    this.camera.position.set(4000, 3000, 6000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.maxDistance = 40000;
    this.controls.minDistance = 40;
    this.controls.addEventListener('change', () => {
      this._updateNear();
      this.requestRender();
    });

    this._lights();
    this._floor();
    this.shapes = portShapes();
    this.stripe = stripeTexture();
    this.bands = bandTexture();
    this._rack();
    this._table();
    this._devices();
    this._ports();
    this._cables();
    this._dims();
    this._floorText();
    this._pickSetup();

    this.resize();
    window.addEventListener('resize', () => this.resize());
    const loop = () => {
      if (this.needsRender) this.renderNow();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  requestRender() {
    this.needsRender = true;
  }

  resize() {
    const w = this.stage.clientWidth || 800;
    const h = this.stage.clientHeight || 600;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = w + 'px';
    this.renderer.domElement.style.height = h + 'px';
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  renderNow() {
    this._checkWallFade();
    this.needsRender = false;
    this.renderer.render(this.three, this.camera);
    this._updateLabels();
  }

  // The lab wall stands behind the rack: from behind it hides the lower units,
  // and seen straight from the front it shows through the empty frame like a
  // rack part. Fade it to 12 % in those two cases.
  _checkWallFade() {
    const cam = this.camera.position;
    const camZ = -cam.z;
    const d = new THREE.Vector3().subVectors(cam, this.controls.target).normalize();
    const dz = -d.z;
    const hz = Math.hypot(d.x, d.z) || 1;
    let fade = false;
    for (const e of this.dev.values()) {
      if (e.d.kind !== 'wall') continue;
      if (camZ > e.d.pos[2] - 20 || (dz < 0 && -dz / hz > 0.94)) fade = true;
    }
    if (fade !== !!this._wallFaded) {
      this._wallFaded = fade;
      this.applyState();
    }
  }

  // ----------------------------------------------------------------- scene
  _lights() {
    const hemi = new THREE.HemisphereLight('#ffffff', '#cfd4db', 1.9);
    this.three.add(hemi);
    const key = new THREE.DirectionalLight('#ffffff', 1.9);
    key.position.set(3000, 5000, 4000);
    this.three.add(key);
    const fill = new THREE.DirectionalLight('#ffffff', 0.9);
    fill.position.set(-4000, 2500, -3000);
    this.three.add(fill);
    const front = new THREE.DirectionalLight('#ffffff', 0.7);
    front.position.set(200, 1200, 5000);
    this.three.add(front);
  }

  _floor() {
    const cx = 2100;
    const cz = -300;
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(16000, 12000),
      new THREE.MeshStandardMaterial({ color: '#eceef1', roughness: 1, metalness: 0 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(cx, -0.5, cz);
    this.three.add(floor);
    const minor = new THREE.GridHelper(16000, 160, '#dde1e6', '#dde1e6');
    minor.position.set(cx, 0.3, cz);
    minor.material.transparent = true;
    minor.material.opacity = 0.9;
    this.three.add(minor);
    const major = new THREE.GridHelper(16000, 16, '#bfc6cf', '#bfc6cf');
    major.position.set(cx, 0.6, cz);
    this.three.add(major);
  }

  _box(x0, y0, z0, x1, y1, z1, mats) {
    // scene-coordinate box (min/max), returns a mesh
    const g = new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0);
    const m = new THREE.Mesh(g, mats);
    m.position.set((x0 + x1) / 2, (y0 + y1) / 2, -(z0 + z1) / 2);
    return m;
  }

  _edges(mesh, color, opacity = 1) {
    const e = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry, 25),
      new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }),
    );
    e.position.copy(mesh.position);
    e.quaternion.copy(mesh.quaternion);
    e.scale.copy(mesh.scale);
    return e;
  }

  _rack() {
    const R = this.scene3.rack;
    const u1 = R.u1_y;
    const topY = u1 + R.units * 44.45;
    this.rackTopY = topY;
    const grp = new THREE.Group();
    grp.name = 'rack-frame';
    const steel = new THREE.MeshStandardMaterial({ color: '#3a4048', roughness: 0.55, metalness: 0.35 });
    const edgeCol = '#1a1d21';
    const fw = R.frame_w || 600;
    const fx0 = (482.6 - fw) / 2;
    const fx1 = fx0 + fw;
    const rd = R.rail_depth || 750;
    const add = (m) => {
      grp.add(m);
      grp.add(this._edges(m, edgeCol, 0.7));
      return m;
    };
    // posts: front rails carry the U numbers
    const railTexL = drawRailFace({ W: 44, H: topY + 30 - 40, S: 2.2, u1: u1 - 40, units: R.units, side: 'left' });
    const railTexR = drawRailFace({ W: 44, H: topY + 30 - 40, S: 2.2, u1: u1 - 40, units: R.units, side: 'right' });
    const railMatL = new THREE.MeshStandardMaterial({ map: railTexL, roughness: 0.6, metalness: 0.2 });
    const railMatR = new THREE.MeshStandardMaterial({ map: railTexR, roughness: 0.6, metalness: 0.2 });
    const postMats = (front) => [steel, steel, steel, steel, front || steel, steel];
    add(this._box(-28, 40, 3, 16, topY + 30, 40, postMats(railMatL)));
    add(this._box(466.6, 40, 3, 510.6, topY + 30, 40, postMats(railMatR)));
    add(this._box(-28, 40, rd - 37, 16, topY + 30, rd, postMats()));
    add(this._box(466.6, 40, rd - 37, 510.6, topY + 30, rd, postMats()));
    // base and top frames
    for (const [y0, y1] of [[28, 72], [topY + 30, topY + 70]]) {
      add(this._box(fx0, y0, -20, fx1, y1, 20, steel));
      add(this._box(fx0, y0, rd - 20, fx1, y1, rd + 20, steel));
      add(this._box(fx0, y0, 20, fx0 + 40, y1, rd - 20, steel));
      add(this._box(fx1 - 40, y0, 20, fx1, y1, rd - 20, steel));
    }
    // levelling feet
    const footMat = new THREE.MeshStandardMaterial({ color: '#202327', roughness: 0.8 });
    for (const x of [fx0 + 20, fx1 - 20]) {
      for (const z of [0, rd]) {
        const f = new THREE.Mesh(new THREE.CylinderGeometry(18, 22, 28, 16), footMat);
        f.position.set(x, 14, -z);
        grp.add(f);
      }
    }
    this.three.add(grp);
    this.rackFrame = grp;
    this.rackBox = { min: [fx0 - 15, 0, -95], max: [Math.max(fx1, 560) + 10, topY + 75, rd + 25] };
    // solid zone used to hide labels of things behind the rack
    this.rackSolid = new THREE.Box3(new THREE.Vector3(0, u1, -rd), new THREE.Vector3(482.6, topY, 0));
  }

  _table() {
    const T = this.scene3.table;
    const grp = new THREE.Group();
    grp.name = 'bench-table';
    const top = new THREE.MeshStandardMaterial({ color: '#e4d8c2', roughness: 0.85, metalness: 0 });
    const leg = new THREE.MeshStandardMaterial({ color: '#9aa3ad', roughness: 0.5, metalness: 0.5 });
    const tm = this._box(T.x0, T.top - 30, T.z0, T.x1, T.top, T.z1, top);
    grp.add(tm, this._edges(tm, '#a89474'));
    for (const x of [T.x0 + 40, T.x1 - 90]) {
      for (const z of [T.z0 + 40, T.z1 - 90]) {
        const l = this._box(x, 0, z, x + 50, T.top - 30, z + 50, leg);
        grp.add(l, this._edges(l, '#6b737c', 0.6));
      }
    }
    // stretcher along the rear
    const st = this._box(T.x0 + 40, 120, T.z1 - 80, T.x1 - 40, 150, T.z1 - 60, leg);
    grp.add(st);
    tm.userData.pick = { type: 'table', id: 'TABLE' };
    this.pickables.push(tm);
    this.three.add(grp);
    this.tableGroup = grp;
    this.tableBox = { min: [T.x0 - 20, T.top - 40, T.z0 - 30], max: [T.x1 + 20, T.top + 250, T.z1 + 20] };
  }

  // ------------------------------------------------------------- devices
  _devices() {
    const portsOf = (d) => (d.ports || []).map((pid) => this.model.ports.get(pid)).filter(Boolean);
    const blankTex = new Map();
    for (const d of this.scene3.devices) {
      const [x, y, z] = d.pos;
      const [w, h, dp] = d.size;
      const color = d.color || '#888888';
      const grp = new THREE.Group();
      grp.name = d.id;
      const entry = { d, grp, meshes: [], mats: [], edges: [], color };
      const mk = (map, col) => {
        const m = new THREE.MeshStandardMaterial({
          color: map ? '#ffffff' : col || color, map: map || null, roughness: 0.62, metalness: 0.08,
        });
        m.userData.baseColor = m.color.clone();
        entry.mats.push(m);
        return m;
      };
      const edgeCol = luminance(color) < 0.12 ? lighten(color, 0.5) : darken(color, 0.55);
      const addMesh = (mesh, withEdges = true) => {
        mesh.userData.pick = { type: 'device', id: d.id };
        grp.add(mesh);
        entry.meshes.push(mesh);
        this.pickables.push(mesh);
        if (withEdges) {
          const e = this._edges(mesh, edgeCol);
          e.userData.baseColor = new THREE.Color(edgeCol);
          grp.add(e);
          entry.edges.push(e);
        }
        return mesh;
      };
      const ps = portsOf(d);
      const facePorts = (face, x0, x1) => ps
        .filter((p) => {
          const n = p.normal;
          if (face === 'front') return n[2] < -0.5 && Math.abs(p.pos[2] - z) < 6;
          if (face === 'rear') return n[2] > 0.5 && Math.abs(p.pos[2] - (z + dp)) < 6;
          return false;
        })
        .map((p) => {
          const sz = (this.shapes[p.type] || this.shapes.default).size;
          return { p, u: face === 'front' ? p.pos[0] - x0 : x1 - p.pos[0], v: p.pos[1] - y, w: sz[0], h: sz[1] };
        });
      const isRackFull = d.group === 'rack' && !d.parent && w > 470;
      const S = 3;
      if (d.kind === 'blank') {
        const key = `${h}`;
        if (!blankTex.has(key)) blankTex.set(key, drawPlain({ W: w, H: h, S: 2, bg: color, ears: true, uN: d.u_n || 1 }));
        const plain = mk(null, color);
        const face = mk(blankTex.get(key));
        addMesh(this._box(x, y, z, x + w, y + h, z + 3, [plain, plain, plain, plain, face, plain]));
        // flange behind
        addMesh(this._box(x + BODY_INSET, y + 2, z + 3, x + w - BODY_INSET, y + h - 2, z + dp, plain), false);
      } else if (isRackFull && (['shelf', 'tray', 'attshelf'].includes(d.kind) || (this.model.children.get(d.id) || []).some((cid) => (this.model.devices.get(cid) || {}).kind !== 'padstack'))) {
        // a unit that carries parts on it is drawn as a shelf; pad stacks screwed onto a Spider box's tiles do not make it one
        const plain = mk(null, color);
        const lipH = Math.min(14, h);
        const lipTex = drawFace({ dev: d, face: 'front', W: w - 2 * EAR, H: lipH, S: 4, bg: color, ports: [], ears: false });
        const lip = mk(lipTex);
        addMesh(this._box(x + EAR, y, z, x + w - EAR, y + lipH, z + 3, [plain, plain, plain, plain, lip, plain]));
        const earTex = drawPlain({ W: EAR, H: h, S: 3, bg: color, ears: false });
        const earM = mk(earTex);
        for (const ex of [x, x + w - EAR]) addMesh(this._box(ex, y, z, ex + EAR, y + h, z + 3, [plain, plain, plain, plain, earM, plain]));
        addMesh(this._box(x + BODY_INSET, y, z + 3, x + w - BODY_INSET, y + 2.5, z + dp, plain));
        for (const sx of [x + BODY_INSET, x + w - BODY_INSET - 2]) addMesh(this._box(sx, y, z + 3, sx + 2, y + lipH, z + dp, plain), false);
        addMesh(this._box(x + BODY_INSET, y, z + dp - 2, x + w - BODY_INSET, y + lipH * 0.8, z + dp, plain), false);
      } else if (isRackFull) {
        const plain = mk(null, color);
        const side = mk(null, darken(color, 0.08));
        const topM = mk(null, lighten(color, 0.06));
        const frontTex = drawFace({ dev: d, face: 'front', W: w, H: h, S, bg: color, ports: facePorts('front', x, x + w), ears: true, uN: d.u_n });
        const front = mk(frontTex);
        addMesh(this._box(x, y, z, x + w, y + h, z + 3, [side, side, topM, plain, front, plain]));
        const bx0 = x + BODY_INSET;
        const bx1 = x + w - BODY_INSET;
        const rearPorts = facePorts('rear', bx0, bx1);
        const rearTex = drawFace({ dev: d, face: 'rear', W: bx1 - bx0, H: h, S: 2.5, bg: darken(color, 0.05), ports: rearPorts, ears: false, uN: d.u_n });
        const rear = mk(rearTex);
        addMesh(this._box(bx0, y + 0.3, z + 3, bx1, y + h - 0.3, z + dp, [side, side, topM, plain, plain, rear]));
      } else if (d.kind === 'padstack') {
        // fixed SMA pad stack: one cylinder per pad along IN -> OUT (port end first),
        // a steel hex at every joint; alternate pads a shade darker so the count reads
        const pin = this.model.ports.get(`${d.id}.in`);
        const pout = this.model.ports.get(`${d.id}.out`);
        const c0 = [x + w / 2, y + h / 2, z + dp / 2];
        const A = toThree(pin ? pin.pos : c0);
        const B = toThree(pout ? pout.pos : c0);
        const axis = B.clone().sub(A);
        const L = axis.length() || Math.max(w, h, dp);
        if (axis.lengthSq() < 1e-6) axis.set(1, 0, 0);
        axis.normalize();
        const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
        const pads = (d.pads && d.pads.length) ? d.pads : [{ len_mm: L, dia_mm: Math.min(w, h, dp) }];
        const tot = pads.reduce((s, pd) => s + (pd.len_mm || 0), 0) || L;
        const steel = mk(null, '#aeb5bd');
        const at = (t, len, r, seg, mat) => {
          const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, seg), mat);
          m.position.copy(A).addScaledVector(axis, t);
          m.quaternion.copy(q);
          return addMesh(m);
        };
        let t0 = 0;
        pads.forEach((pd, i) => {
          const len = ((pd.len_mm || L / pads.length) * L) / tot;
          const r = Math.max(2, (pd.dia_mm || Math.min(w, h, dp)) / 2);
          const nut = Math.min(6, len * 0.2);
          const body = mk(null, i % 2 ? darken(color, 0.22) : color);
          at(t0 + nut / 2, nut, r * 1.02, 6, steel);
          at(t0 + nut + (len - 2 * nut) / 2, len - 2 * nut, r * 0.9, 20, body);
          at(t0 + len - nut / 2, nut, r * 0.8, 6, steel);
          t0 += len;
        });
      } else if (d.kind === 'antenna') {
        const m = mk(null, color);
        const c = new THREE.Mesh(new THREE.CylinderGeometry(w / 2, w / 2, h, 24), m);
        c.position.set(x + w / 2, y + h / 2, -(z + dp / 2));
        addMesh(c);
        // lead-in conduit up to the ceiling
        const mast = new THREE.Mesh(new THREE.CylinderGeometry(6, 6, 200, 12), mk(null, '#8a939d'));
        mast.position.set(x + w / 2, y + h + 100, -(z + dp / 2));
        grp.add(mast);
      } else {
        // generic box (table equipment, shelf items, tray parts)
        const plain = mk(null, color);
        const side = mk(null, darken(color, 0.06));
        const mats = [side, side, plain, plain, plain, plain];
        const fp = facePorts('front', x, x + w);
        const rp = facePorts('rear', x, x + w);
        if (w >= 24 && h >= 12) mats[4] = mk(drawFace({ dev: d, face: 'front', W: w, H: h, S: w > 300 ? 2.5 : 4, bg: color, ports: fp, ears: false }));
        if (w >= 24 && h >= 12 && (rp.length || d.group !== 'rack')) mats[5] = mk(drawFace({ dev: d, face: 'rear', W: w, H: h, S: w > 300 ? 2.5 : 4, bg: darken(color, 0.04), ports: rp, ears: false }));
        if (d.group !== 'rack' && w >= 60 && dp >= 60) {
          const S2 = Math.min(3, 1400 / Math.max(w, dp));
          mats[2] = mk(drawFace({ dev: d, face: 'top', W: w, H: dp, S: S2, bg: lighten(color, 0.04), ports: [], ears: false }));
        }
        addMesh(this._box(x, y, z, x + w, y + h, z + dp, mats));
      }
      this.three.add(grp);
      this.dev.set(d.id, entry);
      // label anchor (three coords)
      entry.anchor = d.group === 'rack'
        ? new THREE.Vector3(x + w / 2, y + h + 4, -z + 2)
        : new THREE.Vector3(x + w / 2, y + h + 8, -(z + dp / 2));
      entry.box = new THREE.Box3(new THREE.Vector3(x, y, -(z + dp)), new THREE.Vector3(x + w, y + h, -z));
      entry.frontNormal = d.group === 'rack' ? this._partFacing(d, ps) : null;
      const rd = this.scene3.rack.rail_depth || 750;
      entry.callFront = new THREE.Vector3(-34, y + h / 2, 4);
      entry.callRear = new THREE.Vector3(-34, y + h / 2, -rd - 4);
      entry.callFrontR = new THREE.Vector3(482.6 + 34, y + h / 2, 4);
      entry.callRearR = new THREE.Vector3(482.6 + 34, y + h / 2, -rd - 4);
      entry.callCenter = new THREE.Vector3(x + w / 2, y + h / 2, -rd / 2);
      // where the faces are actually drawn (rear faces sit at very different depths)
      entry.faceFront = new THREE.Vector3(x + w / 2, y + h / 2, -z);
      entry.faceRear = new THREE.Vector3(x + w / 2, y + h / 2, -(z + dp));
    }
  }

  _partFacing(d, ps) {
    // which way a small rack part is best read from (for label culling)
    const fr = ps.filter((p) => p.normal[2] < -0.5).length;
    const re = ps.filter((p) => p.normal[2] > 0.5).length;
    if (!d.parent) return null;
    if (re > fr) return new THREE.Vector3(0, 0, -1);
    return new THREE.Vector3(0, 0, 1);
  }

  // --------------------------------------------------------------- ports
  _ports() {
    for (const p of this.scene3.ports) {
      const sh = this.shapes[p.type] || this.shapes.default;
      const mats = sh.mats.map((m) => {
        const mm = new THREE.MeshStandardMaterial({ color: m.color, metalness: m.metal, roughness: m.rough });
        mm.userData.baseColor = mm.color.clone();
        return mm;
      });
      const mesh = new THREE.Mesh(sh.geom, mats);
      mesh.position.copy(toThree(p.pos));
      mesh.quaternion.copy(quatForNormal(dirToThree(p.normal)));
      let s = 1;
      if (p.type === 'IEC60309' && p.device !== 'WALL') s = 0.55;
      mesh.scale.setScalar(s);
      mesh.userData.baseScale = s;
      mesh.userData.pick = { type: 'port', id: p.id };
      this.three.add(mesh);
      this.pickables.push(mesh);
      this.portObj.set(p.id, { p, mesh, mats });
    }
  }

  // -------------------------------------------------------------- cables
  _cables() {
    const routes = new Map();
    const meta = new Map();
    for (const c of this.scene3.cables) {
      routes.set(c.id, cleanRoute(c.route));
      meta.set(c.id, { r: Math.max((c.dia || 3) / 2, 1.5), key: `${c.bundle || c.id}|${c.id}` });
    }
    const laned = separateLanes(routes, meta);
    // link colours (for 'colour by link')
    const links = [...new Set(this.scene3.cables.filter((c) => c.cls === 'RF' || c.cls === 'WG').map((c) => linkOfBundle(c.bundle)).filter(Boolean))];
    this.linkColor = new Map();
    let li = 0;
    for (const l of links) {
      if (l === 'spl') this.linkColor.set(l, '#5d6d7e');
      else this.linkColor.set(l, LINK_PALETTE[li++ % LINK_PALETTE.length]);
    }
    for (const c of this.scene3.cables) {
      const r = meta.get(c.id).r;
      const pa = this.model.ports.get(c.a);
      const pb = this.model.ports.get(c.b);
      const coincident = pa && pb && Math.hypot(pa.pos[0] - pb.pos[0], pa.pos[1] - pb.pos[1], pa.pos[2] - pb.pos[2]) < 1;
      const mat = new THREE.MeshStandardMaterial({ color: c.color, roughness: 0.42, metalness: 0.05 });
      if (BANDED.has(c.cls)) mat.map = this.bands;
      let mesh;
      let pts;
      if (coincident) {
        // a bolted flange / direct mate: draw a collar instead of a tube
        const g = new THREE.TorusGeometry(17, 2.2, 8, 28);
        mesh = new THREE.Mesh(g, mat);
        mesh.position.copy(toThree(pa.pos));
        mesh.quaternion.copy(quatForNormal(dirToThree(pa.normal)));
        pts = [pa.pos, pa.pos];
      } else {
        const bend = Math.max(10, Math.min(36, r * 6));
        pts = filletRoute(laned.get(c.id), bend);
        const v3 = pts.map(toThree);
        mesh = new THREE.Mesh(tubeGeometry(v3, r, r > 2.6 ? 10 : 8), mat);
      }
      mesh.userData.pick = { type: 'cable', id: c.id };
      this.three.add(mesh);
      this.pickables.push(mesh);
      // thin centre line used for forgiving picking
      const lg = new THREE.BufferGeometry().setFromPoints(pts.map(toThree));
      const line = new THREE.Line(lg, new THREE.LineBasicMaterial({ visible: false }));
      line.userData.pick = { type: 'cable', id: c.id, line: true };
      this.three.add(line);
      this.cab.set(c.id, {
        c, mesh, mat, line, pts, coincident,
        mid: toThree(pointAtFraction(pts, 0.5)),
        endA: toThree(pts[Math.min(1, pts.length - 1)]),
      });
    }
  }

  cableColor(c) {
    if (this.state.colorMode === 'link' && (c.cls === 'RF' || c.cls === 'WG')) {
      const l = linkOfBundle(c.bundle);
      if (l && this.linkColor.has(l)) return this.linkColor.get(l);
    }
    return c.color;
  }

  // ------------------------------------------------------- dimension lines
  _dims() {
    const grp = new THREE.Group();
    grp.visible = false;
    const mat = new THREE.LineBasicMaterial({ color: '#4a5560' });
    const R = this.scene3.rack;
    const T = this.scene3.table;
    const topY = this.rackTopY;
    this.dimLabels = [];
    const dim = (a, b, text, off) => {
      const A = toThree(a);
      const B = toThree(b);
      const O = new THREE.Vector3(...off);
      const pts = [A, A.clone().add(O), B.clone().add(O), B];
      const g = new THREE.BufferGeometry().setFromPoints(pts);
      grp.add(new THREE.Line(g, mat));
      this.dimLabels.push({ text, anchor: A.clone().add(B).multiplyScalar(0.5).add(O) });
    };
    dim([-70, R.u1_y, 0], [-70, topY, 0], `42U = ${Math.round(topY - R.u1_y)} mm`, [-60, 0, 0]);
    dim([0, topY + 80, 0], [482.6, topY + 80, 0], '19 in panel 482.6 mm', [0, 60, 0]);
    dim([560, topY + 80, 0], [560, topY + 80, R.rail_depth], `rail depth ${R.rail_depth} mm`, [60, 0, 0]);
    dim([T.x0, T.top + 10, T.z0 - 40], [T.x1, T.top + 10, T.z0 - 40], `table ${Math.round(T.x1 - T.x0)} mm (assumed)`, [0, 0, 140]);
    dim([T.x1 + 40, 0, T.z0], [T.x1 + 40, T.top, T.z0], `top ${T.top} mm`, [60, 0, 0]);
    dim([T.x1 + 40, T.top, T.z0], [T.x1 + 40, T.top, T.z1], `${Math.round(T.z1 - T.z0)} mm`, [80, 0, 0]);
    this.three.add(grp);
    this.dimGroup = grp;
  }

  // big orientation words painted on the floor
  _floorText() {
    const R = this.scene3.rack;
    const T = this.scene3.table;
    const mk = (text, w, h, x, z, rot = 0, size = 0.55) => {
      const c = document.createElement('canvas');
      const S = 0.5;
      c.width = Math.round(w * S);
      c.height = Math.round(h * S);
      const ctx = c.getContext('2d');
      ctx.fillStyle = 'rgba(60,70,85,0.55)';
      ctx.font = `800 ${Math.round(h * S * size)}px Inter, "Segoe UI", Arial, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, c.width / 2, c.height / 2);
      const t = new THREE.CanvasTexture(c);
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 8;
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: t, transparent: true, depthWrite: false }));
      m.rotation.x = -Math.PI / 2;
      m.rotation.z = rot;
      m.position.set(x, 1.5, -z);
      this.three.add(m);
      return m;
    };
    mk('▲ FRONT · operator side', 1800, 170, 241, -430);
    mk('REAR ▼', 900, 170, 241, (R.rail_depth || 750) + 380);
    const tl = `BENCH TABLE (assumed ${((T.x1 - T.x0) / 1000).toFixed(1)} m × ${((T.z1 - T.z0) / 1000).toFixed(1)} m, top ${Math.round(T.top)} mm) →`;
    mk(tl, Math.min(3400, T.x1 - T.x0), 150, (T.x0 + T.x1) / 2, T.z0 - 300, 0, 0.42);
  }

  // ------------------------------------------------------------- picking
  _pickSetup() {
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    const el = this.renderer.domElement;
    let down = null;
    el.addEventListener('pointerdown', (e) => {
      down = [e.clientX, e.clientY];
    });
    el.addEventListener('pointerup', (e) => {
      if (!down) return;
      const moved = Math.hypot(e.clientX - down[0], e.clientY - down[1]);
      down = null;
      if (moved > 4 || e.button !== 0) return;
      const hit = this.pickAt(e.clientX, e.clientY);
      this.emit('pick', hit, e);
    });
    el.addEventListener('dblclick', (e) => {
      const hit = this.pickAt(e.clientX, e.clientY);
      if (hit) this.emit('dblpick', hit, e);
    });
    let pending = null;
    el.addEventListener('pointermove', (e) => {
      pending = e;
      if (this._hoverQueued) return;
      this._hoverQueued = true;
      requestAnimationFrame(() => {
        this._hoverQueued = false;
        const ev = pending;
        if (!ev) return;
        const hit = ev.buttons ? null : this.pickAt(ev.clientX, ev.clientY);
        this.emit('hover', hit, ev);
      });
    });
    el.addEventListener('pointerleave', () => {
      pending = null;
      this.emit('hover', null, null);
    });
  }

  _objVisible(o) {
    let p = o;
    while (p) {
      if (!p.visible && !(p.userData.pick && p.userData.pick.line)) return false;
      p = p.parent;
    }
    return true;
  }

  pickAt(clientX, clientY) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const dist = this.camera.position.distanceTo(this.controls.target);
    const worldPerPx = (2 * dist * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) / rect.height;
    this.raycaster.params.Line.threshold = Math.max(1.5, worldPerPx * 5);
    const objs = [...this.pickables];
    for (const e of this.cab.values()) if (e.mesh.visible) objs.push(e.line);
    const hits = this.raycaster.intersectObjects(objs, false).filter((h) => {
      const pk = h.object.userData.pick;
      if (!pk) return false;
      if (pk.type === 'cable') return this.cab.get(pk.id).mesh.visible;
      return this._objVisible(h.object);
    });
    if (!hits.length) return null;
    const first = hits[0];
    const firstDev = hits.find((h) => h.object.userData.pick.type === 'device' || h.object.userData.pick.type === 'table');
    const pref = hits.find((h) => ['cable', 'port'].includes(h.object.userData.pick.type));
    const transparentDev = firstDev && this._isSeeThrough(firstDev.object.userData.pick.id);
    if (pref && (!firstDev || pref.distance <= firstDev.distance + 8 || transparentDev)) {
      // prefer ports over cables when both are at the same spot
      const port = hits.find((h) => h.object.userData.pick.type === 'port' && Math.abs(h.distance - pref.distance) < 6);
      return { ...(port || pref).object.userData.pick, point: (port || pref).point };
    }
    return { ...first.object.userData.pick, point: first.point };
  }

  _isSeeThrough(devId) {
    const e = this.dev.get(devId);
    return !!(e && ((e.mode && (e.mode === 'ghost' || e.mode === 'xray')) || e.faded));
  }

  // --------------------------------------------------------------- state
  set(partial) {
    Object.assign(this.state, partial);
    this.applyState();
  }

  applyState() {
    const st = this.state;
    const m = this.model;
    const hl = st.highlight;
    const stepObj = st.step != null ? m.steps.find((s) => s.n === st.step) : null;
    const stepDevs = stepObj ? new Set(stepObj.devices) : null;
    const stepCabs = stepObj ? new Set(stepObj.cables) : null;
    const ctxDevs = new Set();
    if (stepCabs) {
      for (const cid of stepCabs) {
        const c = m.cables.get(cid);
        for (const pid of [c.a, c.b]) {
          const p = m.ports.get(pid);
          if (p) ctxDevs.add(p.device);
        }
      }
    }
    const ghostDev = hl ? 0.16 : 0.32;
    const ghostCab = hl ? 0.1 : 0.22;
    const devVis = new Map();
    for (const [id, e] of this.dev) {
      const d = e.d;
      let vis = true;
      if (d.kind === 'blank' && !st.showBlanks) vis = false;
      if (d.group === 'table' && !st.showTable) vis = false;
      if (st.step != null) {
        const s = m.devStep.get(id);
        if (s != null && s > st.step) vis = false;
      }
      if (hl && hl.devices.has(id) && !(d.kind === 'blank' && !st.showBlanks)) vis = vis || (d.group !== 'table' || st.showTable);
      devVis.set(id, vis);
    }
    for (const [id, e] of this.dev) {
      const vis = devVis.get(id);
      let mode = 'normal';
      if (hl) mode = hl.devices.has(id) ? 'hi' : 'ghost';
      else if (stepObj) mode = stepDevs.has(id) ? 'hi' : ctxDevs.has(id) ? 'normal' : 'ghost';
      if (st.xray && (mode === 'normal' || mode === 'hi')) mode = mode === 'hi' ? 'hi-x' : 'xray';
      if (e.d.group === 'room' && mode === 'normal') mode = 'room';
      const off = st.scenario === 'after' && m.switched.has(id);
      const dark = st.scenario === 'after' && !off && m.dark.devices.has(id);
      this._applyDevice(e, vis, mode, off, ghostDev, dark);
    }
    const tableVis = st.showTable && (st.step == null || m.tableStep == null || st.step >= m.tableStep);
    this.tableGroup.visible = tableVis;
    for (const [id, e] of this.portObj) {
      const de = this.dev.get(e.p.device);
      const vis = de ? de.grp.visible : true;
      let mode = de ? de.mode : 'normal';
      if (mode === 'hi' || mode === 'hi-x' || mode === 'xray') mode = 'normal';
      if (hl && hl.ports.has(id)) mode = 'hi';
      else if (hl && de && hl.devices.has(e.p.device)) mode = 'normal';
      const darkP = st.scenario === 'after' && (m.dark.ports.has(id) || (de && (de.off || de.dark)));
      this._applyPort(e, vis, mode, ghostDev, darkP, de && de.d.kind === 'wall' && this._wallFaded);
    }
    for (const [id, e] of this.cab) {
      const c = e.c;
      const pa = m.ports.get(c.a);
      const pb = m.ports.get(c.b);
      let vis = st.classes.has(c.cls);
      for (const p of [pa, pb]) if (p && devVis.has(p.device) && !devVis.get(p.device)) vis = false;
      if (st.step != null) {
        const s = m.cabStep.get(id);
        if (s != null && s > st.step && !(hl && hl.cables.has(id))) vis = false;
      }
      let mode = 'normal';
      if (hl) mode = hl.cables.has(id) ? 'hi' : 'ghost';
      else if (stepObj) mode = stepCabs.has(id) ? 'hi' : 'ghost';
      const dashed = st.scenario === 'after' && ((pa && m.switched.has(pa.device)) || (pb && m.switched.has(pb.device)) || m.dark.cables.has(id));
      this._applyCable(e, vis, mode, dashed, ghostCab);
    }
    this.dimGroup.visible = !!st.showDims;
    this.requestRender();
    this.emit('state', st);
  }

  _applyDevice(e, vis, mode, off, ghost, dark = false) {
    e.grp.visible = vis;
    e.mode = mode;
    e.off = off;
    e.dark = dark;
    const faded = e.d.kind === 'wall' && this._wallFaded;
    e.faded = faded;
    const transparent = faded || mode === 'ghost' || mode === 'xray' || mode === 'hi-x' || mode === 'room';
    let op = mode === 'ghost' ? ghost : mode === 'xray' ? 0.2 : mode === 'hi-x' ? 0.55 : mode === 'room' ? 0.5 : 1;
    if (faded) op = Math.min(op, 0.12);
    const tint = this.state.confTint ? CONF_TINT[e.d.conf] || CONF_TINT.other : null;
    for (const mat of e.mats) {
      mat.color.copy(mat.userData.baseColor);
      if (tint && !off) {
        if (mat.map) mat.color.set(tint).lerp(new THREE.Color('#ffffff'), 0.25);
        else mat.color.lerp(new THREE.Color(tint), 0.7);
      }
      if (off) mat.color.multiplyScalar(mat.map ? 0.32 : 0.45);
      else if (dark) mat.color.multiplyScalar(mat.map ? 0.5 : 0.6);
      if (mat.transparent !== transparent) mat.needsUpdate = true;
      mat.transparent = transparent;
      mat.opacity = op;
      mat.depthWrite = !transparent;
      mat.emissive.set(mode === 'hi' || mode === 'hi-x' ? ACCENT : '#000000');
      mat.emissiveIntensity = 0;
    }
    for (const ed of e.edges) {
      const lm = ed.material;
      if (off) lm.color.set(OFF_RED);
      else if (dark) lm.color.set(DARK_EDGE);
      else if (mode === 'hi' || mode === 'hi-x') lm.color.set(ACCENT);
      else lm.color.copy(ed.userData.baseColor);
      let eo = mode === 'ghost' ? Math.min(0.45, ghost * 1.6) : mode === 'xray' ? 0.55 : 1;
      if (faded) eo = 0.15;
      lm.transparent = eo < 1;
      lm.opacity = eo;
    }
  }

  _applyPort(e, vis, mode, ghost, dark = false, faded = false) {
    e.mesh.visible = vis;
    const hi = mode === 'hi';
    e.mesh.scale.setScalar(e.mesh.userData.baseScale * (hi ? 1.9 : 1));
    const transparent = mode === 'ghost' || faded;
    e.mats.forEach((mat, i) => {
      if (mat.transparent !== transparent) mat.needsUpdate = true;
      mat.transparent = transparent;
      mat.opacity = faded ? 0.12 : transparent ? Math.min(0.5, ghost * 1.5) : 1;
      mat.color.copy(mat.userData.baseColor);
      if (dark) mat.color.multiplyScalar(0.35);
      mat.depthWrite = !transparent;
      mat.emissive.set(hi && i === 0 ? ACCENT : '#000000');
      mat.emissiveIntensity = hi ? 0.8 : 0;
    });
  }

  _applyCable(e, vis, mode, dashed, ghost) {
    e.mesh.visible = vis;
    e.mode = mode;
    e.dashed = dashed;
    const mat = e.mat;
    const col = dashed ? '#8f969e' : this.cableColor(e.c);
    mat.color.set(col);
    const transparent = mode === 'ghost';
    const wantAlpha = dashed ? this.stripe : null;
    if (mat.alphaMap !== wantAlpha || mat.transparent !== transparent) mat.needsUpdate = true;
    mat.alphaMap = wantAlpha;
    mat.alphaTest = dashed ? 0.5 : 0;
    mat.transparent = transparent;
    mat.opacity = transparent ? ghost : 1;
    mat.depthWrite = !transparent;
    mat.emissive.set(mode === 'hi' ? col : '#000000');
    mat.emissiveIntensity = mode === 'hi' ? 0.55 : 0;
    e.mesh.renderOrder = mode === 'hi' ? 2 : 0;
  }

  // ------------------------------------------------------------- cameras
  subjectBox(name) {
    const m = this.model;
    const union = (boxes) => {
      const mn = [Infinity, Infinity, Infinity];
      const mx = [-Infinity, -Infinity, -Infinity];
      for (const b of boxes) {
        for (let k = 0; k < 3; k++) {
          mn[k] = Math.min(mn[k], b.min[k]);
          mx[k] = Math.max(mx[k], b.max[k]);
        }
      }
      return { min: mn, max: mx };
    };
    const devBox = (d) => ({ min: d.pos.slice(), max: [d.pos[0] + d.size[0], d.pos[1] + d.size[1], d.pos[2] + d.size[2]] });
    if (name === 'rack') return this.rackBox;
    if (name === 'table') return this.tableBox;
    if (name === 'rackIF') {
      const ds = this.scene3.devices.filter((d) => d.group === 'rack' && ['spider', 'ztdat', 'tray', 'attshelf', 'splitter', 'coupler', 'padstack'].includes(d.kind));
      if (!ds.length) return this.rackBox;
      const b = union(ds.map(devBox));
      // pad stacks on the Spider tiles stand proud of the front rail
      return { min: [-70, b.min[1] - 10, Math.min(-70, b.min[2] - 10)], max: [560, b.max[1] + 10, 60] };
    }
    const room = this.scene3.devices.filter((d) => d.group === 'room' && d.kind !== 'antenna').map(devBox);
    return union([this.rackBox, this.tableBox, { min: [0, 0, 0], max: [0, 0, 0] }, ...room]);
  }

  static VIEWS = {
    front: { dir: [0, 0, -1], subject: 'rack' },
    rear: { dir: [0, 0, 1], subject: 'rack' },
    left: { dir: [-1, 0.14, 0], subject: 'all' },
    right: { dir: [1, 0.14, 0], subject: 'all' },
    top: { dir: [0, 1, -0.0015], subject: 'all' },
    iso: { dir: [0.62, 0.55, -1], subject: 'all' },
    table: { dir: [0, 1.05, -1], subject: 'table' },
    rack: { dir: [0.34, 0.12, -1], subject: 'rackIF' },
  };

  setView(name) {
    const v = Viewer.VIEWS[name];
    if (!v) return false;
    const b = this.subjectBox(v.subject);
    this.frameBox(b.min, b.max, v.dir, { margin: v.subject === 'all' ? 1.04 : 1.06, fov: DEFAULT_FOV });
    this.currentView = name;
    return true;
  }

  currentDirScene() {
    const d = new THREE.Vector3().subVectors(this.camera.position, this.controls.target).normalize();
    return [d.x, d.y, -d.z];
  }

  // min/max in SCENE coordinates; dir = direction from subject to camera (scene coords)
  frameBox(min, max, dir, { margin = 1.08, insets = null, fov = null } = {}) {
    dir = dir || this.currentDirScene();
    if (fov && fov !== this.camera.fov) {
      // a narrow field of view keeps straight-on close-ups close to an elevation drawing
      this.camera.fov = fov;
      this.camera.updateProjectionMatrix();
    }
    const ins = insets || (this.getInsets ? this.getInsets() : null) || { top: 0, right: 0, bottom: 0, left: 0 };
    const cw = this.renderer.domElement.clientWidth || 800;
    const ch = this.renderer.domElement.clientHeight || 600;
    const fw = Math.max(0.3, (cw - ins.left - ins.right) / cw);
    const fh = Math.max(0.3, (ch - ins.top - ins.bottom) / ch);
    const corners = [];
    for (const x of [min[0], max[0]]) for (const y of [min[1], max[1]]) for (const z of [min[2], max[2]]) corners.push(toThree([x, y, z]));
    const center = toThree([(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2]);
    const back = dirToThree(dir).normalize();
    const cam = this.camera;
    const tanV0 = Math.tan(THREE.MathUtils.degToRad(cam.fov / 2));
    const tanV = tanV0 * fh;
    const tanH = tanV0 * cam.aspect * fw;
    let target = center.clone();
    let D = 1000;
    for (let it = 0; it < 3; it++) {
      cam.position.copy(target).addScaledVector(back, 1000);
      cam.up.set(0, 1, 0);
      cam.lookAt(target);
      cam.updateMatrixWorld();
      const r = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0);
      const u = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 1);
      const bk = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 2);
      D = 0;
      for (const c of corners) {
        const p = c.clone().sub(target);
        const x = p.dot(r);
        const y = p.dot(u);
        const zb = p.dot(bk);
        D = Math.max(D, zb + Math.abs(x) / tanH, zb + Math.abs(y) / tanV);
      }
      D *= margin;
      // recentre on the projected bounding rectangle
      let x0 = Infinity;
      let x1 = -Infinity;
      let y0 = Infinity;
      let y1 = -Infinity;
      for (const c of corners) {
        const p = c.clone().sub(target);
        const depth = D - p.dot(bk);
        const sx = p.dot(r) / (depth * tanH);
        const sy = p.dot(u) / (depth * tanV);
        x0 = Math.min(x0, sx);
        x1 = Math.max(x1, sx);
        y0 = Math.min(y0, sy);
        y1 = Math.max(y1, sy);
      }
      const ox = ((x0 + x1) / 2) * tanH * D;
      const oy = ((y0 + y1) / 2) * tanV * D;
      target = target.clone().addScaledVector(r, ox).addScaledVector(u, oy);
    }
    {
      // shift so the subject sits in the middle of the area not covered by panels
      cam.position.copy(target).addScaledVector(back, D);
      cam.lookAt(target);
      cam.updateMatrixWorld();
      const r = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0);
      const u = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 1);
      const wpp = (2 * D * tanV0) / ch;
      target = target.clone().addScaledVector(r, -((ins.left - ins.right) / 2) * wpp).addScaledVector(u, ((ins.top - ins.bottom) / 2) * wpp);
    }
    cam.position.copy(target).addScaledVector(back, D);
    cam.up.set(0, 1, 0);
    this.controls.target.copy(target);
    cam.lookAt(target);
    this._updateNear();
    this.controls.update();
    this.requestRender();
  }

  _updateNear() {
    const d = this.camera.position.distanceTo(this.controls.target);
    this.camera.near = Math.max(0.5, Math.min(40, d / 300));
    this.camera.far = d * 8 + 30000;
    this.camera.updateProjectionMatrix();
  }

  boxOf(kind, id) {
    // scene-coord box for an item
    if (kind === 'device') {
      const d = this.model.devices.get(id);
      return { min: d.pos.slice(), max: [d.pos[0] + d.size[0], d.pos[1] + d.size[1], d.pos[2] + d.size[2]] };
    }
    if (kind === 'port') {
      const p = this.model.ports.get(id);
      const r = 60;
      return { min: [p.pos[0] - r, p.pos[1] - r, p.pos[2] - r], max: [p.pos[0] + r, p.pos[1] + r, p.pos[2] + r] };
    }
    if (kind === 'cable') {
      const e = this.cab.get(id);
      const mn = [Infinity, Infinity, Infinity];
      const mx = [-Infinity, -Infinity, -Infinity];
      for (const p of e.pts) for (let k = 0; k < 3; k++) {
        mn[k] = Math.min(mn[k], p[k]);
        mx[k] = Math.max(mx[k], p[k]);
      }
      for (let k = 0; k < 3; k++) {
        mn[k] -= 40;
        mx[k] += 40;
      }
      return { min: mn, max: mx };
    }
    return null;
  }

  // ------------------------------------------------------------- labels
  setLabelProvider(fn) {
    this.labelProvider = fn;
  }

  _labelEl(key, cls, html) {
    let l = this.labels.get(key);
    if (!l) {
      const el = document.createElement('div');
      el.className = 'tlabel ' + cls;
      el.innerHTML = html;
      this.labelLayer.appendChild(el);
      l = { el, html, cls, size: null };
      this.labels.set(key, l);
    } else if (l.html !== html || l.cls !== cls) {
      l.el.innerHTML = html;
      l.el.className = 'tlabel ' + cls;
      l.html = html;
      l.cls = cls;
      l.size = null;
    }
    return l;
  }

  _updateLabels() {
    const items = this.labelProvider ? this.labelProvider() : [];
    const w = this.renderer.domElement.clientWidth;
    const h = this.renderer.domElement.clientHeight;
    const placed = [];
    const used = new Set();
    items.sort((a, b) => b.pri - a.pri);
    const camPos = this.camera.position;
    const v = new THREE.Vector3();
    const occluders = [];
    if (items.some((it) => it.occlude)) {
      for (const e of this.dev.values()) {
        if (!e.grp.visible || (e.mode && e.mode !== 'normal' && e.mode !== 'hi')) continue;
        for (const mm of e.meshes) occluders.push(mm);
      }
      for (const o of this.rackFrame.children) if (o.isMesh) occluders.push(o);
    }
    const ray = new THREE.Raycaster();
    for (const it of items) {
      if (it.occlude && occluders.length) {
        const dirv = it.anchor.clone().sub(camPos);
        const dist = dirv.length();
        ray.set(camPos, dirv.normalize());
        ray.far = dist - 6;
        const hits = ray.intersectObjects(occluders, false);
        if (hits.some((hh) => hh.object.userData.pick && hh.object.userData.pick.id !== it.devId)) continue;
        // things outside the rack seen through the open shelves still count as hidden
        if (it.outsideRack && this.rackSolid && ray.ray.intersectBox(this.rackSolid, _tmpV) && _tmpV.distanceTo(camPos) < dist - 6 && !this.state.xray) continue;
      }
      if (it.normal) {
        const to = camPos.clone().sub(it.anchor);
        if (to.dot(it.normal) < 0) continue;
      }
      v.copy(it.anchor).project(this.camera);
      if (v.z > 1 || v.z < -1) continue;
      const sx = (v.x * 0.5 + 0.5) * w;
      let sy = (-v.y * 0.5 + 0.5) * h;
      if (it.yFrom) {
        // beside the rack, but at the height the device face itself is drawn
        _tmpV.copy(it.yFrom).project(this.camera);
        sy = (-_tmpV.y * 0.5 + 0.5) * h;
      }
      if (sx < -50 || sx > w + 50 || sy < -20 || sy > h + 20) continue;
      const l = this._labelEl(it.key, it.cls, it.html);
      if (!l.size) {
        l.el.style.display = 'block';
        l.size = [l.el.offsetWidth, l.el.offsetHeight];
      }
      const [lw, lh] = l.size;
      let align = it.align || 'above';
      if (align === 'auto' && it.away) {
        v.copy(it.away).project(this.camera);
        align = (v.x * 0.5 + 0.5) * w < sx ? 'right' : 'left';
      }
      let x = sx - lw / 2;
      let y = sy - lh - 6;
      if (align === 'left') {
        x = sx - lw - 4;
        y = sy - lh / 2;
      } else if (align === 'right') {
        x = sx + 4;
        y = sy - lh / 2;
      }
      let rect = [x, y, x + lw, y + lh];
      const pad = it.pad != null ? it.pad : 2;
      const clash = (rc) => placed.some((p) => rc[0] < p[2] + pad && rc[2] > p[0] - pad && rc[1] < p[3] + pad && rc[3] > p[1] - pad);
      if (clash(rect)) {
        // try a few nudges
        let ok = false;
        // last resort: well clear to the left (a room item beside the rack whose U tags took its place)
        const nudges = align === 'above' ? [[0, -lh - 3], [lw / 2 + 4, 0], [-lw / 2 - 4, 0], [0, lh + 12], [-lw - 12, lh / 2 + 4]] : [];
        for (const [dx, dy] of nudges) {
          const rc = [x + dx, y + dy, x + dx + lw, y + dy + lh];
          if (!clash(rc)) {
            x += dx;
            y += dy;
            rect = rc;
            ok = true;
            break;
          }
        }
        if (!ok && !it.force) continue;
      }
      placed.push(rect);
      used.add(it.key);
      l.el.style.display = 'block';
      l.el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    }
    for (const [k, l] of this.labels) if (!used.has(k)) l.el.style.display = 'none';
  }
}

export { mix };
