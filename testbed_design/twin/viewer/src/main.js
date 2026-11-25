// Entry: reads the inlined scene JSON, builds the viewer and the UI, exposes
// window.twin for scripted renders and applies URL query parameters.
//
// window.twin (every call renders synchronously, so a screenshot can follow):
//   ready                      Promise, resolves once the first frame is drawn
//   setView(name)              front | rear | left | right | top | iso | table | rack
//   setStep(n | null, {frame}) installation state after step n (null = finished bench)
//   frameStep(n)               camera on the items of step n, from the step's view
//   highlight({devices, cables, ports}) / clearHighlight()   everything else ghosted
//   select(id) / focus(id)     open the info panel (focus also moves the camera)
//   trace(cableId, frame=true) follow the signal through passive parts; returns the paths
//   setClasses([cls] | null)   setScenario('before'|'after')   setLabels(b)  setXray(b)
//   showTable(b)  showBlanks(b)  setDims(b)  setConfTint(b)  setColorMode('class'|'link')
//   setUI('full'|'min'|'none') frameBox([x0,y0,z0],[x1,y1,z1], dir?)  render()  getState()
// URL: ?view=front&step=7&highlight=R001,R002&trace=R009&focus=SB2&select=SB2
//      &classes=RF,WG&scenario=after&labels=0&xray=1&table=0&blanks=0&dims=1&conf=1
//      &color=link&ui=none
import css from './style.css';
import * as THREE from 'three';
import { Viewer } from './viewer.js';
import { buildModel, traceCable, tokenLabel, escapeHtml as esc, CLASS_ORDER, linkOfBundle, isPassive, mateOf, cableAt as cableAtPort } from './model.js';

const VIEW_NAMES = [
  ['front', 'Front'], ['rear', 'Rear'], ['left', 'Left'], ['right', 'Right'],
  ['top', 'Top'], ['iso', 'Iso'], ['table', 'Table'], ['rack', 'Rack close-up'],
];
const KIND_NAMES = {
  spider: 'Siklu Spider box', ztdat: 'programmable attenuator', tray: 'tray', shelf: 'shelf', dcdist: 'DC distribution',
  fuse: 'fuse panel', rect: 'rectifier', pdu: 'switched PDU', sw_data: 'data switch', sw_oob: 'management switch',
  gnss: 'GNSS splitter', cm: 'cable manager', blank: 'blanking panel', minipc: 'mini PC', adaptor: 'DC/PoE adaptor',
  injector: 'PoE injector', splitter: 'RF splitter', coupler: 'directional coupler', transition: 'waveguide transition',
  mwatt: 'multi-channel attenuator', ip50: 'IP-50 radio', server: 'server', pc: 'PC', specan: 'spectrum analyser',
  strip: 'mains strip', psu: 'power supply', wall: 'wall sockets', antenna: 'antenna',
  rackserver: 'rack server', attshelf: 'attenuator shelf', monitor: 'monitor', keyboard: 'keyboard',
  padstack: 'fixed pad stack (SMA attenuators)', armstack: 'DC block / pad stack', arrestor: 'GNSS surge arrestor',
  bridge: 'cable bridge', earthbar: 'earth bar', spacer: 'spacer bar',
};

function h(tag, attrs = {}, html = '') {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  if (html) el.innerHTML = html;
  return el;
}

class App {
  constructor(scene) {
    this.scene = scene;
    this.model = buildModel(scene);
    this.sel = null; // {type,id}
    this.trace = null;
    this.uiMode = 'full';
    // rack items in elevation order (for callouts that alternate left / right)
    this.rackOrder = new Map(scene.devices
      .filter((d) => d.group === 'rack' && !d.parent && d.u && d.kind !== 'blank')
      .sort((a, b) => (b.u_top || 0) - (a.u_top || 0))
      .map((d, i) => [d.id, i]));
    this._build();
  }

  _build() {
    const s = this.scene;
    const meta = s.meta;
    document.title = `${meta.poc} digital twin (tentative)`;
    const st = document.createElement('style');
    st.textContent = css;
    document.head.appendChild(st);
    const app = h('div', { id: 'app' });
    document.body.appendChild(app);

    // --- top bar
    const top = h('header', { class: 'topbar' });
    top.appendChild(h('div', { class: 'brand' }, `<span class="poc">${esc(meta.poc)}</span><span class="badge-tent" title="Geometry partly estimated; see Checks and assumptions">TENTATIVE</span>`));
    top.appendChild(h('div', { class: 'ttl' }, `<span class="name" title="${esc(meta.name)}">${esc(meta.name)}</span><span class="dwg">${esc(meta.drawing)} · 3D digital twin · ${s.devices.length} devices · ${s.cables.length} cables · ${s.ports.length} ports · geometry partly estimated</span>`));
    const views = h('div', { class: 'views' });
    this.viewBtns = {};
    for (const [k, label] of VIEW_NAMES) {
      const b = h('button', { onclick: () => this.setView(k), title: `${label} view` }, label);
      this.viewBtns[k] = b;
      views.appendChild(b);
    }
    top.appendChild(views);
    const search = h('div', { class: 'search' });
    search.innerHTML = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="7" cy="7" r="5"/><path d="M11 11l4 4"/></svg>';
    // search example: a cable, a passive part of this bench (attenuator or pad stack) and a switch port
    const exPart = ['ATT-03', ...this.scene.devices.filter((d) => d.kind === 'padstack' || d.kind === 'ztdat').map((d) => d.id)].find((id) => this.model.devices.has(id)) || 'SB2';
    // example cable from the scene: an RF cable that starts on a pad stack (PoC 1), else R057 if it exists, else the first RF cable
    const rfc = this.scene.cables.filter((c) => c.cls === 'RF');
    const exCable = (rfc.find((c) => (c.pad_ends || []).length) || rfc.find((c) => c.id === 'R057') || rfc[0] || { id: 'R001' }).id;
    const inp = h('input', { type: 'search', placeholder: `Find ${exCable}, ${exPart}, SW-DATA.P7…`, list: 'twin-ids', spellcheck: 'false' });
    const dl = h('datalist', { id: 'twin-ids' });
    dl.innerHTML = this.model.allIds.map((id) => `<option value="${esc(id)}">`).join('');
    const msg = h('div', { class: 'msg' });
    inp.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const ok = this.focus(inp.value.trim(), true);
        msg.textContent = ok ? '' : `not found: ${inp.value.trim()}`;
      }
    });
    inp.addEventListener('change', () => {
      if (this.model.allIds.includes(inp.value.trim())) {
        this.focus(inp.value.trim(), true);
        msg.textContent = '';
      }
    });
    search.append(inp, dl, msg);
    this.searchInput = inp;
    top.appendChild(search);
    app.appendChild(top);

    const body = h('div', { class: 'body' });
    app.appendChild(body);
    const side = h('aside', { class: 'side' });
    body.appendChild(side);
    const stage = h('main', { class: 'stage' });
    body.appendChild(stage);
    this.stage = stage;

    this.viewer = new Viewer(stage, this.model);
    const V = this.viewer;
    V.getInsets = () => {
      const r = { top: 0, right: 0, bottom: 0, left: 0 };
      if (this.info.classList.contains('show') && this.info.offsetWidth) r.right = this.info.offsetWidth + 24;
      // the step card is docked on the left, under the view tag
      if (this.stepcap.style.display !== 'none' && this.stepcap.offsetWidth) r.left = this.stepcap.offsetWidth + 24;
      return r;
    };

    // overlays
    this.tooltip = h('div', { class: 'tooltip' });
    this.info = h('div', { class: 'info' });
    this.stepcap = h('div', { class: 'stepcap' });
    this.stepcap.style.display = 'none';
    this.viewtag = h('div', { class: 'viewtag' });
    stage.append(this.tooltip, this.info, this.stepcap, this.viewtag,
      h('div', { class: 'watermark' }, 'TENTATIVE · geometry partly estimated · not for construction'),
      h('div', { class: 'hint' }, 'drag: orbit · right-drag: pan · wheel: zoom · click: inspect · double-click: focus'));

    // --- side: cable classes
    const secC = h('section');
    secC.appendChild(h('h3', {}, 'Cables by class <span class="mini" data-act="all">all</span>'));
    this.classRows = {};
    const legend = [...s.legend].sort((a, b) => CLASS_ORDER.indexOf(a.cls) - CLASS_ORDER.indexOf(b.cls));
    for (const l of legend) {
      const row = h('label', { class: 'legend-row', title: `${l.n} ${l.cls} cables` });
      const cb = h('input', { type: 'checkbox' });
      cb.checked = true;
      cb.addEventListener('change', () => this._classesFromUI());
      row.append(cb, h('span', { class: 'sw', style: `background:${l.color}` }), h('span', { class: 'nm' }, `<b>${esc(l.cls)}</b>${esc(l.name)}`), h('span', { class: 'ct' }, String(l.n)));
      this.classRows[l.cls] = { row, cb };
      secC.appendChild(row);
    }
    secC.querySelector('[data-act=all]').addEventListener('click', () => {
      const allOn = Object.values(this.classRows).every((r) => r.cb.checked);
      for (const r of Object.values(this.classRows)) r.cb.checked = !allOn;
      this._classesFromUI();
    });
    side.appendChild(secC);

    // --- side: display toggles
    const secD = h('section');
    secD.appendChild(h('h3', {}, 'Display'));
    this.toggles = {};
    const tog = (key, label, init, fn, title = '') => {
      const row = h('label', { class: 'toggle', title });
      const cb = h('input', { type: 'checkbox' });
      cb.checked = init;
      cb.addEventListener('change', () => fn(cb.checked));
      row.append(cb, h('span', {}, label));
      this.toggles[key] = cb;
      secD.appendChild(row);
    };
    tog('table', 'Bench table and its equipment', true, (v) => this.showTable(v));
    tog('blanks', 'Blanking panels', true, (v) => this.showBlanks(v));
    tog('labels', 'Labels', true, (v) => this.setLabels(v));
    tog('xray', 'X-ray (see cables inside the rack)', false, (v) => this.setXray(v));
    tog('dims', 'Main dimensions', false, (v) => this.setDims(v));
    tog('conf', 'Tint devices by confidence (amber = estimate, blue = likely)', false, (v) => this.setConfTint(v), 'how sure the geometry is');
    tog('link', 'Colour IF / IP-50 cables by link', false, (v) => this.setColorMode(v ? 'link' : 'class'), 'one colour per IF link / IP-50 hop (like blueprint sheet 2)');
    this.linkLegend = h('div', { class: 'linklegend' });
    this.linkLegend.innerHTML = [...V.linkColor.entries()].map(([l, c]) => `<span><i style="background:${esc(c)}"></i>${esc(l === 'spl' ? 'tile→splitter' : l)}</span>`).join('');
    this.linkLegend.style.display = 'none';
    secD.appendChild(this.linkLegend);
    side.appendChild(secD);

    // --- side: scenario
    if (s.meta.switched && s.meta.switched.length) {
      const secS = h('section');
      secS.appendChild(h('h3', {}, 'Event t0'));
      const seg = h('div', { class: 'seg' });
      this.scenBtns = {
        before: h('button', { onclick: () => this.setScenario('before') }, 'Before t0'),
        after: h('button', { class: 'after', onclick: () => this.setScenario('after') }, 'After t0'),
      };
      seg.append(this.scenBtns.before, this.scenBtns.after);
      secS.appendChild(seg);
      const dk = this.model.dark;
      const darkTxt = [...dk.nodes.map((n) => `<a class="lnk" data-id="${esc(n.port)}">${esc(n.box)} ${esc(n.node)}</a>`),
        ...[...dk.devices].filter((id) => !dk.nodes.some((n) => n.box === id)).map((id) => `<a class="lnk" data-id="${esc(id)}">${esc(id)}</a>`)];
      secS.appendChild(h('div', { class: 'event' }, `<span style="color:var(--bad);font-weight:600">Switched off at t0:</span> ${s.meta.switched.map((id) => `<a class="lnk" data-id="${esc(id)}">${esc(id)}</a>`).join(', ')}`
        + (darkTxt.length ? `<div style="margin-top:4px"><span style="color:#9a3412;font-weight:600">Dark downstream (no power):</span> ${darkTxt.join(', ')}; ${dk.cables.size} cables dead, links ${esc([...dk.links].join(', '))}</div>` : '')
        + `<details class="evt"><summary>what happens at t0</summary>${esc(s.meta.event || '')}</details>`));
      secS.addEventListener('click', (e) => {
        const a = e.target.closest('[data-id]');
        if (a) this.select(a.dataset.id, true);
      });
      side.appendChild(secS);
    }

    // --- side: steps
    const secT = h('section');
    secT.appendChild(h('h3', {}, `Installation steps <span class="mini" data-act="all">show finished bench</span>`));
    const ctl = h('div', { class: 'steps-ctl' });
    this.stepAll = h('button', { class: 'all on', title: 'finished bench', onclick: () => this.setStep(null, { frame: false, fromUI: true }) }, 'All');
    const prev = h('button', { title: 'previous step', onclick: () => this.stepBy(-1) }, '‹');
    const next = h('button', { title: 'next step', onclick: () => this.stepBy(1) }, '›');
    this.stepRange = h('input', { type: 'range', min: '0', max: String(s.steps.length), value: '0' });
    this.stepRange.addEventListener('input', () => {
      const v = +this.stepRange.value;
      this.setStep(v === 0 ? null : v, { frame: true, fromUI: true });
    });
    ctl.append(this.stepAll, prev, this.stepRange, next);
    secT.appendChild(ctl);
    const ul = h('ul', { class: 'steplist' });
    this.stepItems = [];
    for (const stp of s.steps) {
      const li = h('li', { title: stp.text }, `<span class="n">${stp.n}</span><span>${esc(stp.title)}</span>`);
      li.addEventListener('click', () => this.setStep(stp.n, { frame: true, fromUI: true }));
      ul.appendChild(li);
      this.stepItems.push(li);
    }
    secT.appendChild(ul);
    secT.querySelector('[data-act=all]').addEventListener('click', () => this.setStep(null, { frame: false, fromUI: true }));
    side.appendChild(secT);

    // --- side: checks
    const secK = h('section');
    const det = h('details', { class: 'coll', open: '' });
    const nFail = s.checks.filter((c) => !c.ok).length;
    det.appendChild(h('summary', {}, `<h3>Checks (${s.checks.length - nFail}/${s.checks.length} pass) &amp; assumptions</h3>`));
    const cl = h('ul', { class: 'checks' });
    for (const c of s.checks) {
      cl.appendChild(h('li', {}, `<span class="st ${c.ok ? 'ok' : 'bad'}">${c.ok ? 'PASS' : 'FAIL'}</span><span>${esc(c.name)}<span class="dt">${esc(c.detail || '')}</span></span>`));
    }
    // viewer-side geometry sanity
    const ov = this.model.overlaps;
    const ovTxt = ov.length ? ov.map(([a, b]) => `${a} / ${b}`).join(', ') : 'none';
    cl.appendChild(h('li', {}, `<span class="st ${ov.length ? 'warn' : 'ok'}">${ov.length ? 'WARN' : 'PASS'}</span><span>Viewer: no two device boxes intersect<span class="dt">${ov.length} intersecting pair(s): ${esc(ovTxt)}</span></span>`));
    const nEst = s.devices.filter((d) => d.conf === 'estimate').length;
    cl.appendChild(h('li', {}, `<span class="st warn">INFO</span><span>Confidence of device geometry<span class="dt">${nEst} of ${s.devices.length} devices are estimates, ${s.devices.length - nEst} likely</span></span>`));
    det.appendChild(cl);
    const adet = h('details', { class: 'evt' });
    adet.appendChild(h('summary', {}, `Assumptions (${s.assumptions.length + 1})`));
    det.appendChild(adet);
    const al = h('ul', { class: 'assume' });
    for (const a of s.assumptions) al.appendChild(h('li', {}, esc(a)));
    al.appendChild(h('li', {}, 'Viewer: rack devices drawn as a 482.6 mm faceplate plus a 440 mm wide body; overlapping cable runs are spread a few mm apart so each cable stays visible (ports and lengths unchanged).'));
    al.appendChild(h('li', {}, 'Viewer: cable colours are a display palette (DC yellow, GNSS lime, WG pink, each banded) so that no two classes look alike on paper; the cable schedules keep the colours of the scene file. After t0, what the switched devices feed (node, radio) is shown dark too, worked out from the POE / DC cables.'));
    al.appendChild(h('li', {}, 'Viewer: the lab wall is faded when the camera looks at the rack straight from the front or from behind the wall.'));
    adet.appendChild(al);
    secK.appendChild(det);
    side.appendChild(secK);

    // --- events
    V.on('hover', (hit, ev) => this._hover(hit, ev));
    V.on('pick', (hit) => {
      if (!hit || hit.type === 'table') {
        this.clearSelection();
        return;
      }
      this.select(hit.id, false, hit.type);
    });
    V.on('dblpick', (hit) => {
      if (hit && hit.type !== 'table') this.focus(hit.id, false);
    });
    V.setLabelProvider(() => this._labels());
    window.addEventListener('keydown', (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (e.key === 'ArrowRight') this.stepBy(1);
      else if (e.key === 'ArrowLeft') this.stepBy(-1);
      else if (e.key === 'Escape') this.clearSelection();
    });
    this.info.addEventListener('click', (e) => this._infoClick(e));

    this._syncScenario();
    V.applyState();
    V.resize();
    this.setView('iso');
  }

  // ------------------------------------------------------------ actions
  _classesFromUI() {
    const on = Object.entries(this.classRows).filter(([, r]) => r.cb.checked).map(([k]) => k);
    this.setClasses(on, false);
  }

  setClasses(list, syncUI = true) {
    const set = new Set(list == null ? Object.keys(this.classRows) : list.map((x) => String(x).toUpperCase()));
    for (const [k, r] of Object.entries(this.classRows)) {
      if (syncUI) r.cb.checked = set.has(k);
      r.row.classList.toggle('off', !set.has(k));
    }
    this.viewer.set({ classes: set });
  }

  setView(name) {
    const ok = this.viewer.setView(name);
    for (const [k, b] of Object.entries(this.viewBtns)) b.classList.toggle('on', k === name);
    const lbl = (VIEW_NAMES.find(([k]) => k === name) || [])[1];
    this.viewtag.textContent = lbl ? `${lbl} view${name === 'front' ? ' · rack front faces you' : name === 'rear' ? ' · from behind the rack' : ''}` : '';
    this.viewtag.style.display = lbl ? 'block' : 'none';
    return ok;
  }

  _clearViewTag() {
    for (const b of Object.values(this.viewBtns)) b.classList.remove('on');
    this.viewtag.style.display = 'none';
    this.viewer.currentView = null;
  }

  showTable(v) {
    this.toggles.table.checked = !!v;
    this.viewer.set({ showTable: !!v });
  }
  showBlanks(v) {
    this.toggles.blanks.checked = !!v;
    this.viewer.set({ showBlanks: !!v });
  }
  setLabels(v) {
    this.toggles.labels.checked = !!v;
    this.viewer.set({ showLabels: !!v });
  }
  setXray(v) {
    this.toggles.xray.checked = !!v;
    this.viewer.set({ xray: !!v });
  }
  setDims(v) {
    this.toggles.dims.checked = !!v;
    this.viewer.set({ showDims: !!v });
  }
  setConfTint(v) {
    this.toggles.conf.checked = !!v;
    this.viewer.set({ confTint: !!v });
  }
  setColorMode(m) {
    this.toggles.link.checked = m === 'link';
    this.linkLegend.style.display = m === 'link' ? 'flex' : 'none';
    this.viewer.set({ colorMode: m });
  }

  setScenario(s) {
    if (s !== 'before' && s !== 'after') return false;
    this.viewer.set({ scenario: s });
    this._syncScenario();
    if (this.sel) this._renderInfo();
    return true;
  }
  _syncScenario() {
    if (!this.scenBtns) return;
    const s = this.viewer.state.scenario;
    this.scenBtns.before.classList.toggle('on', s === 'before');
    this.scenBtns.after.classList.toggle('on', s === 'after');
  }

  stepBy(d) {
    const n = this.scene.steps.length;
    const cur = this.viewer.state.step || 0;
    let nx = cur + d;
    if (nx < 0) nx = 0;
    if (nx > n) nx = n;
    this.setStep(nx === 0 ? null : nx, { frame: true, fromUI: true });
  }

  setStep(n, { frame = false, fromUI = false } = {}) {
    if (n != null) {
      n = +n;
      if (!this.model.steps.find((s) => s.n === n)) return false;
    }
    this.viewer.state.highlight = null;
    this.sel = null;
    this.trace = null;
    this.viewer.set({ step: n });
    this.stepRange.value = String(n || 0);
    this.stepAll.classList.toggle('on', n == null);
    this.stepItems.forEach((li, i) => {
      const k = i + 1;
      li.classList.toggle('cur', k === n);
      li.classList.toggle('done', n != null && k < n);
    });
    if (n != null) {
      const st = this.model.steps.find((s) => s.n === n);
      this.stepcap.innerHTML = `<b>Step ${st.n} / ${this.model.steps.length} · ${esc(st.title)}</b><div class="tx">${esc(st.text)}</div>`;
      this.stepcap.style.display = 'block';
      if (frame) this.frameStep(n);
      if (fromUI) this.stepItems[n - 1].scrollIntoView({ block: 'nearest' });
    } else {
      this.stepcap.style.display = 'none';
      if (fromUI && frame) this.setView('iso');
    }
    this._renderInfo();
    return true;
  }

  frameStep(n) {
    const st = this.model.steps.find((s) => s.n === n);
    if (!st) return;
    const V = this.viewer;
    const view = Viewer.VIEWS[st.view] ? st.view : 'iso';
    if (view === 'iso' || view === 'table' || (!st.devices.length && !st.cables.length)) {
      this.setView(view === 'iso' || view === 'table' ? view : st.view);
      return;
    }
    const mn = [Infinity, Infinity, Infinity];
    const mx = [-Infinity, -Infinity, -Infinity];
    const grow = (b) => {
      for (let k = 0; k < 3; k++) {
        mn[k] = Math.min(mn[k], b.min[k]);
        mx[k] = Math.max(mx[k], b.max[k]);
      }
    };
    for (const id of st.devices) grow(V.boxOf('device', id));
    for (const id of st.cables) grow(V.boxOf('cable', id));
    // always keep the rack width in frame so the step reads in context
    mn[0] = Math.min(mn[0], -60);
    mx[0] = Math.max(mx[0], 545);
    V.frameBox(mn, mx, Viewer.VIEWS[view].dir, { margin: 1.12 });
    for (const [k, b] of Object.entries(this.viewBtns)) b.classList.toggle('on', k === view);
    this.viewtag.textContent = `${(VIEW_NAMES.find(([k]) => k === view) || [])[1] || view} view · step ${n}`;
    this.viewtag.style.display = 'block';
  }

  // resolve an id (case-insensitive, exact first then prefix)
  resolve(id) {
    if (!id) return null;
    const m = this.model;
    const tryExact = (x) => {
      if (m.cables.has(x)) return { type: 'cable', id: x };
      if (m.devices.has(x)) return { type: 'device', id: x };
      if (m.ports.has(x)) return { type: 'port', id: x };
      return null;
    };
    let r = tryExact(id);
    if (r) return r;
    const up = id.toUpperCase();
    const all = m.allIds;
    const ci = all.find((x) => x.toUpperCase() === up);
    if (ci) return tryExact(ci);
    const pre = all.find((x) => x.toUpperCase().startsWith(up));
    if (pre) return tryExact(pre);
    return null;
  }

  select(id, focus = false, typeHint = null) {
    const r = typeHint && this._exists(typeHint, id) ? { type: typeHint, id } : this.resolve(id);
    if (!r) return false;
    this.sel = r;
    this.trace = null;
    const m = this.model;
    let hl;
    if (r.type === 'device') {
      const cabs = m.devCables.get(r.id) || [];
      const kids = m.children.get(r.id) || [];
      hl = { devices: [r.id, ...kids], cables: cabs, ports: [] };
    } else if (r.type === 'cable') {
      const c = m.cables.get(r.id);
      hl = { devices: [m.ports.get(c.a).device, m.ports.get(c.b).device], cables: [r.id], ports: [c.a, c.b] };
    } else {
      const p = m.ports.get(r.id);
      const cid = m.portCable.get(r.id);
      const ports = [r.id];
      const devs = [p.device];
      if (cid) {
        const c = m.cables.get(cid);
        ports.push(c.a === r.id ? c.b : c.a);
        devs.push(m.ports.get(ports[1]).device);
      }
      hl = { devices: devs, cables: cid ? [cid] : [], ports };
    }
    this._setHighlight(hl);
    this._renderInfo();
    if (focus) this._focusItem(r);
    return true;
  }

  _exists(type, id) {
    const m = this.model;
    return (type === 'device' && m.devices.has(id)) || (type === 'cable' && m.cables.has(id)) || (type === 'port' && m.ports.has(id));
  }

  clearSelection() {
    this.sel = null;
    this.trace = null;
    this.viewer.set({ highlight: null });
    this._renderInfo();
  }

  _setHighlight(hl) {
    if (!hl) {
      this.viewer.set({ highlight: null });
      return;
    }
    // make sure highlighted things are actually shown
    const st = this.viewer.state;
    const m = this.model;
    const cls = new Set(st.classes);
    let changed = false;
    for (const cid of hl.cables || []) {
      const c = m.cables.get(cid);
      if (c && !cls.has(c.cls)) {
        cls.add(c.cls);
        changed = true;
      }
    }
    if (changed) this.setClasses([...cls]);
    const touchesTable = [...(hl.devices || [])].some((id) => m.devices.get(id) && m.devices.get(id).group === 'table');
    if (touchesTable && !st.showTable) this.showTable(true);
    this.viewer.set({
      highlight: {
        devices: new Set(hl.devices || []),
        cables: new Set(hl.cables || []),
        ports: new Set(hl.ports || []),
      },
    });
  }

  highlight(spec) {
    this.sel = null;
    this.trace = null;
    const expand = (ids) => (ids || []).map((x) => (this.resolve(x) || {}).id || x);
    this._setHighlight({ devices: expand(spec.devices), cables: expand(spec.cables), ports: expand(spec.ports) });
    this._renderInfo();
  }

  doTrace(cableId, frame = true) {
    const r = this.resolve(cableId);
    if (!r) return null;
    let cid = r.id;
    if (r.type === 'port') cid = cableAtPort(this.model, r.id);
    if (r.type === 'device') {
      const l = this.model.devCables.get(r.id) || [];
      cid = l[0];
    }
    if (!cid) return null;
    const t = traceCable(this.model, cid);
    this.sel = { type: 'cable', id: cid };
    this.trace = t;
    this._setHighlight({ devices: [...t.devices], cables: [...t.cables], ports: [...t.ports] });
    this._renderInfo();
    if (frame) {
      const V = this.viewer;
      const mn = [Infinity, Infinity, Infinity];
      const mx = [-Infinity, -Infinity, -Infinity];
      for (const id of t.cables) {
        const b = V.boxOf('cable', id);
        for (let k = 0; k < 3; k++) {
          mn[k] = Math.min(mn[k], b.min[k]);
          mx[k] = Math.max(mx[k], b.max[k]);
        }
      }
      const c = this.model.cables.get(cid);
      const dir = c.cls === 'WG' ? Viewer.VIEWS.table.dir : Viewer.VIEWS.rack.dir;
      V.frameBox(mn, mx, this._frontish(dir), { margin: 1.15 });
      this._clearViewTag();
    }
    this._renderInfo();
    return t.paths.map((p) => p.map((tok) => tokenLabel(this.model, tok)));
  }

  _frontish(dir) {
    return dir;
  }

  focus(id, highlightIt = true) {
    const r = this.resolve(id);
    if (!r) return false;
    if (highlightIt) this.select(r.id, false, r.type);
    this._focusItem(r);
    return true;
  }

  _focusItem(r) {
    const V = this.viewer;
    const b = V.boxOf(r.type, r.id);
    if (!b) return;
    let dir = V.currentDirScene();
    if (r.type === 'cable') {
      // look from the side both ends face: a rear cable from behind the rack
      const c = this.model.cables.get(r.id);
      const nz = [c.a, c.b].map((pid) => (this.model.ports.get(pid) || { normal: [0, 0, 0] }).normal[2]);
      const onTable = [c.a, c.b].some((pid) => {
        const p = this.model.ports.get(pid);
        const d = p && this.model.devices.get(p.device);
        return d && d.group === 'table';
      });
      if (nz.every((z) => z > 0.5)) dir = [0.3, 0.25, 1];
      else if (nz.every((z) => z < -0.5)) dir = [0.3, 0.25, -1];
      else if (nz.some((z) => z > 0.5)) dir = onTable ? [0.25, 0.55, 1] : [0.3, 0.25, 1];
      else if (onTable) dir = Viewer.VIEWS.table.dir;
    } else if (r.type === 'port') {
      const p = this.model.ports.get(r.id);
      const n = p.normal;
      dir = [n[0] + 0.25, n[1] + 0.3, n[2] - (Math.abs(n[2]) < 0.5 ? 0.6 : 0)];
    } else if (r.type === 'device') {
      const d = this.model.devices.get(r.id);
      const pad = Math.max(80, Math.max(...d.size) * 0.35);
      for (let k = 0; k < 3; k++) {
        b.min[k] -= pad;
        b.max[k] += pad;
      }
      if (d.group === 'rack') {
        const front = (d.ports || []).some((pid) => this.model.ports.get(pid).normal[2] < -0.5) || !(d.ports || []).length;
        dir = front ? [0.3, 0.25, -1] : [0.3, 0.25, 1];
      }
    }
    V.frameBox(b.min, b.max, dir, { margin: 1.2 });
    this._clearViewTag();
  }

  // ------------------------------------------------------------ hover/tooltip
  _hover(hit, ev) {
    const tt = this.tooltip;
    if (!hit || !ev || hit.type === 'table') {
      tt.style.display = 'none';
      this.viewer.renderer.domElement.style.cursor = 'default';
      return;
    }
    const m = this.model;
    let html = '';
    if (hit.type === 'device') {
      const d = m.devices.get(hit.id);
      html = `<b>${esc(d.id)}</b> ${d.u ? `· ${esc(d.u)}` : ''}<div class="tt2">${esc(d.title || KIND_NAMES[d.kind] || d.kind)}</div>`;
    } else if (hit.type === 'cable') {
      const c = m.cables.get(hit.id);
      html = `<b>${esc(c.id)}</b> · ${esc(c.cls)} · ${esc(c.length)}<div class="tt2">${esc(c.a)} → ${esc(c.b)}</div>`;
    } else if (hit.type === 'port') {
      const p = m.ports.get(hit.id);
      const cid = m.portCable.get(hit.id);
      const mt = cid ? null : mateOf(m, hit.id);
      html = `<b>${esc(p.id)}</b> · ${esc(p.type)}<div class="tt2">${esc(p.label || '')}${cid ? ` · cable ${esc(cid)}` : mt ? ` · screwed onto ${esc(mt)}` : ' · not connected'}</div>`;
    }
    tt.innerHTML = html;
    const rect = this.stage.getBoundingClientRect();
    tt.style.display = 'block';
    let x = ev.clientX - rect.left + 14;
    let y = ev.clientY - rect.top + 14;
    if (x + tt.offsetWidth > rect.width - 6) x = ev.clientX - rect.left - tt.offsetWidth - 10;
    if (y + tt.offsetHeight > rect.height - 6) y = ev.clientY - rect.top - tt.offsetHeight - 10;
    tt.style.left = x + 'px';
    tt.style.top = y + 'px';
    this.viewer.renderer.domElement.style.cursor = 'pointer';
  }

  // ------------------------------------------------------------ info panel
  _infoClick(e) {
    const x = e.target.closest('.x');
    if (x) {
      this.clearSelection();
      return;
    }
    const a = e.target.closest('[data-id]');
    if (a) {
      e.preventDefault();
      this.select(a.dataset.id, true);
      return;
    }
    const b = e.target.closest('[data-act]');
    if (!b) return;
    const act = b.dataset.act;
    if (act === 'trace') this.doTrace(b.dataset.arg);
    else if (act === 'focus') this.focus(b.dataset.arg, false);
    else if (act === 'step') this.setStep(+b.dataset.arg, { frame: true, fromUI: true });
    else if (act === 'clear') this.clearSelection();
  }

  _link(id) {
    return `<a data-id="${esc(id)}">${esc(id)}</a>`;
  }

  _renderInfo() {
    const box = this.info;
    const m = this.model;
    const st = this.viewer.state;
    if (!this.sel) {
      if (st.step != null) {
        box.classList.remove('show');
      } else box.classList.remove('show');
      return;
    }
    const r = this.sel;
    let hd = '';
    let bd = '';
    const row = (k, v) => `<div class="row"><div class="k">${k}</div><div class="v">${v}</div></div>`;
    if (r.type === 'device') {
      const d = m.devices.get(r.id);
      const off = st.scenario === 'after' && m.switched.has(d.id);
      hd = `<div><div class="id">${esc(d.id)}</div><div class="kind">${esc(KIND_NAMES[d.kind] || d.kind)}</div></div>`;
      bd += row('Title', esc(d.title || d.label || ''));
      if (d.sub) bd += row('Part', esc(d.sub));
      let where = '';
      if (d.group === 'rack' && d.u) where = `rack <b>${esc(d.u)}</b> (${d.u_n || 1}U), front at the rail plane`;
      else if (d.parent) {
        const p = m.devices.get(d.parent);
        where = `on ${this._link(d.parent)}${p && p.u ? ` (${esc(p.u)})` : ''}, x ${Math.round(d.pos[0])}, ${Math.round(d.pos[2])} mm behind the front rail`;
      } else if (d.group === 'table') where = `bench table, x ${Math.round(d.pos[0])}–${Math.round(d.pos[0] + d.size[0])} mm, z ${Math.round(d.pos[2])}–${Math.round(d.pos[2] + d.size[2])} mm`;
      else where = `${esc(d.group)}, at x ${Math.round(d.pos[0])}, y ${Math.round(d.pos[1])}, z ${Math.round(d.pos[2])} mm`;
      bd += row('Position', where);
      bd += row('Size', `<span class="mono">${d.size.map((v) => Math.round(v * 10) / 10).join(' × ')} mm</span> <span style="color:var(--ink3)">(w × h × d)</span>`);
      bd += row('Confidence', `<span class="chip ${esc(d.conf)}">${esc(d.conf)}</span>`);
      if (d.note) bd += `<div class="note">${esc(d.note)}</div>`;
      if (m.switched.has(d.id)) bd += row('Event t0', `<span class="chip off">switched off at t0</span>${off ? '' : ' <span style="color:var(--ink3)">(see After t0)</span>'}`);
      const dkNodes = m.dark.nodes.filter((n) => n.box === d.id);
      if (m.dark.devices.has(d.id) || dkNodes.length) {
        const why = m.dark.why.get(d.id) || (dkNodes.length ? m.dark.why.get(`${d.id}.${dkNodes[0].node}`) : null);
        const what = dkNodes.length ? `node${dkNodes.length > 1 ? 's' : ''} ${dkNodes.map((n) => esc(n.node)).join(', ')}` : 'the whole unit';
        bd += row('Event t0', `<span class="chip dark">dark after t0</span> ${what}: no power once ${why ? this._link(why.from) : 'its feed'} is off${why && why.via.length ? ` (via ${why.via.map((x) => this._link(x)).join(', ')})` : ''}`);
      }
      if (d.modules) bd += row('Modules', d.modules.map((x) => `<span class="mono">${esc(x.m)}</span> ${esc(x.sector)}`).join(' · '));
      if (d.kind === 'padstack') {
        const pads = (d.pads || []).map((x) => `<span class="mono">${esc(x.pn)}</span> ${esc(x.db)} dB`).join(' + ');
        bd += row('Pads', `${pads || esc(d.text)} = <b>${esc(d.db)} dB</b>${d.target_db != null ? ` <span style="color:var(--ink3)">(target ${esc(d.target_db)} dB)</span>` : ''}`);
        if (d.host) bd += row('On', `${this._link(d.host)} <span style="color:var(--ink3)">(IN screwed on; port first, ${esc(d.stack_mm)} mm long)</span>`);
        if (d.support) bd += row('Support', esc(d.support));
      }
      if (d.spare) bd += row('Spare', 'yes (unwired)');
      const kids = m.children.get(d.id) || [];
      if (kids.length) bd += row('Holds', kids.map((k) => this._link(k)).join(' '));
      const stepN = m.devStep.get(d.id);
      if (stepN) bd += row('Installed', `<a class="lnk" data-act="step" data-arg="${stepN}">step ${stepN}</a> · ${esc(m.steps.find((s) => s.n === stepN).title)}`);
      const ps = (d.ports || []).map((pid) => m.ports.get(pid));
      if (ps.length) {
        bd += `<div class="sec">Ports (${ps.length})</div><table><tr><th>port</th><th>type</th><th>cable</th><th>other end</th></tr>`;
        for (const p of ps) {
          const cid = m.portCable.get(p.id);
          const c = cid ? m.cables.get(cid) : null;
          const other = c ? (c.a === p.id ? c.b : c.a) : '';
          const mt = c ? null : mateOf(m, p.id);
          const cabCell = c ? this._link(cid) : mt ? '<span style="color:var(--ink3)">screwed on</span>' : '<span style="color:#b3bac4">—</span>';
          bd += `<tr><td class="mono">${this._link(p.id).replace(`>${esc(p.id)}<`, `>${esc(p.name || p.id)}<`)}</td><td>${esc(p.type)}</td><td>${cabCell}</td><td class="mono">${other ? this._link(other) : mt ? this._link(mt) : ''}</td></tr>`;
        }
        bd += '</table>';
      }
      bd += `<div class="btns"><button class="pri" data-act="focus" data-arg="${esc(d.id)}">Focus</button>${(m.devCables.get(d.id) || []).length && isPassive(d.kind) ? `<button data-act="trace" data-arg="${esc(m.devCables.get(d.id)[0])}">Trace through</button>` : ''}<button data-act="clear">Clear</button></div>`;
    } else if (r.type === 'cable') {
      const c = m.cables.get(r.id);
      const clsLeg = this.scene.legend.find((l) => l.cls === c.cls);
      hd = `<div><div class="id">${esc(c.id)}</div><div class="kind"><span class="chip cls" style="background:${esc(c.color)}">${esc(c.cls)}</span> ${esc(clsLeg ? clsLeg.name : '')}</div></div>`;
      bd += row('From', `${this._link(c.a)}${c.a_raw && c.a_raw !== c.a ? `<div style="color:var(--ink3);font-size:11.5px">${esc(c.a_raw)}</div>` : ''}`);
      bd += row('To', `${this._link(c.b)}${c.b_raw && c.b_raw !== c.b ? `<div style="color:var(--ink3);font-size:11.5px">${esc(c.b_raw)}</div>` : ''}`);
      bd += row('Part', `${esc(c.part || '')}${c.pn ? ` <span class="mono" style="color:var(--ink3)">(${esc(c.pn)})</span>` : ''}`);
      let lenTxt = `<b>${esc(c.length || '?')}</b> chosen`;
      if (c.route_mm != null) lenTxt += ` · routed ${Math.round(c.route_mm)} mm`;
      if (c.spec_mm != null && c.route_mm != null) {
        const slack = c.spec_mm - c.route_mm;
        lenTxt += slack < 0 ? ` · <span class="warnline">short by ${Math.round(-slack)} mm</span>` : ` · slack ${Math.round(slack)} mm (${Math.round((c.route_mm / c.spec_mm) * 100)} % used)`;
      }
      bd += row('Length', lenTxt);
      if (c.vlan) bd += row('VLAN', `<span class="mono">${esc(c.vlan)}</span>`);
      if (c.note) bd += row('Note', esc(c.note));
      bd += row('Diameter', `${c.dia} mm`);
      const link = linkOfBundle(c.bundle);
      if (c.bundle && c.bundle !== c.id) {
        const mates = this.scene.cables.filter((x) => x.bundle === c.bundle && x.id !== c.id).map((x) => this._link(x.id));
        bd += row('Bundle', `<span class="mono" style="font-size:11px">${esc(link || c.bundle)}</span>${mates.length ? `<div>${mates.join(' ')}</div>` : ''}`);
      }
      const stepN = m.cabStep.get(c.id);
      if (stepN) bd += row('Installed', `<a class="lnk" data-act="step" data-arg="${stepN}">step ${stepN}</a> · ${esc(m.steps.find((s) => s.n === stepN).title)}`);
      const sw = [c.a, c.b].some((pid) => m.switched.has(m.ports.get(pid).device));
      if (sw) bd += row('Event t0', '<span class="chip off">dead after t0</span> one end is switched off');
      else if (m.dark.cables.has(c.id)) bd += row('Event t0', '<span class="chip dark">dead after t0</span> part of a path that loses power at t0');
      if (this.trace) {
        const t = this.trace;
        bd += `<div class="sec">Signal trace (${t.paths.length} path${t.paths.length === 1 ? '' : 's'}${t.truncated ? ', truncated' : ''})</div>`;
        for (const p of t.paths) {
          bd += '<div class="chain">' + p.map((tok) => {
            const lbl = esc(tokenLabel(m, tok));
            if (tok.t === 'cable') return `<a data-id="${esc(tok.id)}">${lbl}</a>`;
            if (tok.t === 'dev') return `<a class="pd" data-id="${esc(tok.id)}">${lbl}</a>`;
            return `<a class="ep" data-id="${esc(tok.id)}">${lbl}</a>`;
          }).join('<span class="ar"> → </span>') + '</div>';
        }
      }
      bd += `<div class="btns"><button class="pri" data-act="trace" data-arg="${esc(c.id)}">Trace signal</button><button data-act="focus" data-arg="${esc(c.id)}">Focus</button><button data-act="clear">Clear</button></div>`;
    } else {
      const p = m.ports.get(r.id);
      const cid = m.portCable.get(p.id);
      const c = cid ? m.cables.get(cid) : null;
      hd = `<div><div class="id">${esc(p.id)}</div><div class="kind">port · ${esc(p.type)}</div></div>`;
      bd += row('Device', this._link(p.device));
      bd += row('Label', esc(p.label || p.name));
      bd += row('Faces', `<span class="mono">${p.normal[2] < -0.5 ? 'front' : p.normal[2] > 0.5 ? 'rear' : p.normal[1] > 0.5 ? 'up' : p.normal[1] < -0.5 ? 'down' : p.normal[0] > 0 ? 'right' : 'left'}</span>`);
      bd += row('Position', `<span class="mono">x ${Math.round(p.pos[0])}, y ${Math.round(p.pos[1])}, z ${Math.round(p.pos[2])} mm</span>`);
      const mt = c ? null : mateOf(m, p.id);
      if (c) {
        bd += row('Cable', `${this._link(cid)} · ${esc(c.cls)} · ${esc(c.length)}`);
        bd += row('Other end', this._link(c.a === p.id ? c.b : c.a));
      } else if (mt) {
        bd += row('Screwed onto', `${this._link(mt)} <span style="color:var(--ink3)">(no cable: direct SMA mate)</span>`);
      } else bd += row('Cable', '<span style="color:var(--ink3)">not connected</span>');
      const tc = c ? cid : mt ? cableAtPort(m, p.id) : null;
      bd += `<div class="btns">${tc ? `<button class="pri" data-act="trace" data-arg="${esc(tc)}">Trace signal</button>` : ''}<button data-act="focus" data-arg="${esc(p.id)}">Focus</button><button data-act="clear">Clear</button></div>`;
    }
    box.innerHTML = `<div class="hd">${hd}<button class="x" title="close">×</button></div><div class="bd">${bd}</div>`;
    box.classList.add('show');
  }

  // ------------------------------------------------------------ labels
  _labels() {
    const V = this.viewer;
    const m = this.model;
    const st = V.state;
    const out = [];
    const hl = st.highlight;
    const cam = V.camera;
    const el = V.renderer.domElement;
    const tanV = Math.tan(THREE.MathUtils.degToRad(cam.fov / 2));
    const pxPerMm = (p) => el.clientHeight / (2 * tanV * Math.max(1, cam.position.distanceTo(p)));
    const padGroups = new Map(); // highlighted pad stacks: one label per link side
    for (const [id, e] of V.dev) {
      if (!e.grp.visible) continue;
      const d = e.d;
      const isHi = hl && hl.devices.has(id);
      if (e.off) {
        out.push({ key: 'off:' + id, cls: 'off', html: `${esc(id)} · OFF at t0`, anchor: e.anchor, pri: 100, force: true });
        continue;
      }
      if (e.dark) {
        out.push({ key: 'dark:' + id, cls: 'dark', html: `${esc(id)} · dark after t0`, anchor: e.anchor, pri: 95, force: true });
        continue;
      }
      if (d.kind === 'padstack' && (isHi || (!hl && e.mode === 'hi'))) {
        const k = `${d.chain}|${d.link}|${d.side}`;
        if (!padGroups.has(k)) padGroups.set(k, []);
        padGroups.get(k).push(e);
        continue;
      }
      const rackTop = d.group === 'rack' && !d.parent && d.u && d.kind !== 'blank';
      if (isHi && !(rackTop && st.rackCallAll)) {
        out.push({ key: 'd:' + id, cls: 'hi', html: esc(id), anchor: e.anchor, pri: 90 });
        continue;
      }
      if (!st.showLabels || d.kind === 'blank') continue;
      // rackCallAll (scripted elevation / band renders): every rack item keeps its callout
      if (e.mode === 'ghost' && !(rackTop && st.rackCallAll)) continue;
      if (d.group === 'rack' && !d.parent && d.u) {
        const uPx = pxPerMm(e.callFront) * 44.45;
        if (uPx < 15) continue;
        const html = `<b>${esc(d.u)}</b> ${esc(id)}`;
        const cur = isHi || e.mode === 'hi' || e.mode === 'hi-x';
        // alternate mode (elevation drawings): every other item on the other side
        const alt = !!st.rackCallAlt;
        const right = alt && (this.rackOrder.get(id) || 0) % 2 === 1;
        // elevation / band drawings (rackCallAll): the U tags are placed before the room items beside the rack
        // (earth bar, arrestor), which then move aside, so every U keeps its tag
        const pri = st.rackCallAll ? 50 : 30;
        out.push({ key: 'rf:' + id, cls: cur ? 'rackcall cur' : 'rackcall', html, anchor: right ? e.callFrontR : e.callFront, yFrom: e.faceFront, align: right ? 'right' : 'left', pad: 0, pri, normal: new THREE.Vector3(0, 0, 1), force: alt });
        out.push({ key: 'rr:' + id, cls: cur ? 'rackcall cur' : 'rackcall', html, anchor: right ? e.callRearR : e.callRear, yFrom: e.faceRear, align: 'auto', away: e.callCenter, pad: 0, pri, normal: new THREE.Vector3(0, 0, -1), force: alt });
        continue;
      }
      if (e.faded) continue;
      if (d.group === 'table' || d.group === 'room') {
        if (d.parent && pxPerMm(e.anchor) * Math.max(d.size[0], d.size[2]) < 26) continue;
        out.push({ key: 'd:' + id, cls: d.parent ? 'part' : '', html: esc(id), anchor: e.anchor, pri: d.parent ? 25 : 45, occlude: true, outsideRack: true, devId: id });
      } else if (d.parent) {
        if (st.partLabels === false) continue;
        // parts big enough to carry their id on their own face need no overlay
        if (d.size[0] >= 60 && d.size[1] >= 30) continue;
        // pad stacks: only once they are drawn large enough to tell apart (zoomed in)
        if (pxPerMm(e.anchor) * (d.kind === 'padstack' ? Math.max(...d.size) / 2.5 : d.size[0]) < 20) continue;
        const html = d.kind === 'padstack' ? `${esc(id)} · ${esc(d.text)} dB` : esc(id);
        out.push({ key: 'd:' + id, cls: 'part', html, anchor: e.anchor, pri: 20, normal: d.kind === 'padstack' ? null : e.frontNormal, occlude: d.kind !== 'padstack', devId: id });
      }
    }
    for (const [k, es] of padGroups) {
      // 'PAD-HUB-S.t1-t4 · 30 + 5 dB' above the group of stacks
      const d0 = es[0].d;
      const anchor = new THREE.Vector3();
      es.forEach((e) => anchor.add(e.anchor));
      anchor.multiplyScalar(1 / es.length);
      anchor.y = Math.max(...es.map((e) => e.anchor.y));
      const tiles = es.map((e) => e.d.tile).filter((t) => t != null).sort((a, b) => a - b);
      let name = d0.id;
      if (es.length > 1 && tiles.length === es.length) {
        const run = tiles.every((t, i) => !i || t === tiles[i - 1] + 1);
        name = `${d0.id.replace(/\.t\d+$/, '')}.${run ? `t${tiles[0]}-t${tiles[tiles.length - 1]}` : tiles.map((t) => `t${t}`).join(',')}`;
      }
      out.push({ key: 'pg:' + k, cls: 'hi', html: `${esc(name)} · ${esc(d0.text)} dB`, anchor, pri: 88 });
    }
    if (hl) {
      const many = hl.cables.size > 36;
      if (!many) {
        for (const cid of hl.cables) {
          const e = V.cab.get(cid);
          if (!e || !e.mesh.visible) continue;
          out.push({ key: 'c:' + cid, cls: 'cab', html: `<i style="background:${esc(V.cableColor(e.c))}"></i>${esc(cid)}`, anchor: e.mid, pri: 80 });
        }
      }
      for (const pid of hl.ports) {
        const e = V.portObj.get(pid);
        if (!e || !e.mesh.visible) continue;
        if (hl.ports.size > 6 && !(this.sel && this.sel.type === 'port' && this.sel.id === pid)) continue;
        out.push({ key: 'p:' + pid, cls: 'hi', html: esc(pid), anchor: e.mesh.position.clone().add(new THREE.Vector3(0, 10, 0)), pri: 85 });
      }
    } else if (st.step != null) {
      const stp = m.steps.find((s) => s.n === st.step);
      if (stp && stp.cables.length <= (st.cableLabelMax || 16)) {
        for (const cid of stp.cables) {
          const e = V.cab.get(cid);
          if (!e || !e.mesh.visible) continue;
          out.push({ key: 'c:' + cid, cls: 'cab', html: `<i style="background:${esc(V.cableColor(e.c))}"></i>${esc(cid)}`, anchor: e.mid, pri: 70 });
        }
      }
    }
    if (st.showDims && V.dimLabels) {
      V.dimLabels.forEach((dl, i) => out.push({ key: 'dim:' + i, cls: 'dim', html: esc(dl.text), anchor: dl.anchor, pri: 60 }));
    }
    return out;
  }

  setUI(mode) {
    this.uiMode = mode;
    document.body.classList.toggle('ui-min', mode === 'min');
    document.body.classList.toggle('ui-none', mode === 'none');
    this.viewer.resize();
    if (this.viewer.currentView) this.viewer.setView(this.viewer.currentView);
  }
}

// ------------------------------------------------------------------ boot
function boot() {
  const raw = document.getElementById('twin-scene');
  const scene = JSON.parse(raw.textContent);
  let resolveReady;
  const ready = new Promise((r) => {
    resolveReady = r;
  });
  const api = { ready };
  window.twin = api;
  const start = () => {
    const app = new App(scene);
    const V = app.viewer;
    const sync = (fn) => (...a) => {
      const r = fn(...a);
      V.renderNow();
      return r;
    };
    Object.assign(api, {
      app,
      scene,
      setView: sync((n) => app.setView(n)),
      setStep: sync((n, opts) => app.setStep(n == null || n === 'all' ? null : +n, opts || {})),
      frameStep: sync((n) => app.frameStep(+n)),
      highlight: sync((spec) => app.highlight(spec || {})),
      clearHighlight: sync(() => app.clearSelection()),
      select: sync((id) => app.select(id, false)),
      trace: sync((id, frame = true) => app.doTrace(id, frame)),
      setClasses: sync((l) => app.setClasses(l)),
      setScenario: sync((s) => app.setScenario(s)),
      setLabels: sync((b) => app.setLabels(b)),
      setXray: sync((b) => app.setXray(b)),
      showTable: sync((b) => app.showTable(b)),
      showBlanks: sync((b) => app.showBlanks(b)),
      setDims: sync((b) => app.setDims(b)),
      setColorMode: sync((m) => app.setColorMode(m)),
      setConfTint: sync((b) => app.setConfTint(b)),
      setUI: sync((m) => app.setUI(m)),
      focus: sync((id) => app.focus(id, true)),
      frameBox: sync((a, b, dir) => V.frameBox(a, b, dir)),
      render: () => V.renderNow(),
      getState: () => ({ ...V.state, classes: [...V.state.classes], highlight: V.state.highlight ? { devices: [...V.state.highlight.devices], cables: [...V.state.highlight.cables], ports: [...V.state.highlight.ports] } : null, view: V.currentView || null }),
      views: Object.keys(Viewer.VIEWS),
    });
    // URL parameters
    const q = new URLSearchParams(location.search);
    const bool = (v) => !['0', 'false', 'off', 'no'].includes(String(v).toLowerCase());
    if (q.has('ui')) app.setUI(q.get('ui'));
    if (q.has('classes')) app.setClasses(q.get('classes').split(',').filter(Boolean));
    if (q.has('table')) app.showTable(bool(q.get('table')));
    if (q.has('blanks')) app.showBlanks(bool(q.get('blanks')));
    if (q.has('labels')) app.setLabels(bool(q.get('labels')));
    if (q.has('xray')) app.setXray(bool(q.get('xray')));
    if (q.has('dims')) app.setDims(bool(q.get('dims')));
    if (q.has('color')) app.setColorMode(q.get('color'));
    if (q.has('conf')) app.setConfTint(bool(q.get('conf')));
    if (q.has('scenario')) app.setScenario(q.get('scenario'));
    if (q.has('step')) app.setStep(q.get('step') === 'all' ? null : +q.get('step'), { frame: !q.has('view') });
    if (q.has('view')) app.setView(q.get('view'));
    if (q.has('highlight')) {
      const ids = q.get('highlight').split(',').filter(Boolean);
      const spec = { devices: [], cables: [], ports: [] };
      for (const id of ids) {
        const r = app.resolve(id);
        if (r) spec[r.type + 's'].push(r.id);
      }
      app.highlight(spec);
    }
    if (q.has('trace')) app.doTrace(q.get('trace'), !q.has('view'));
    if (q.has('select')) app.select(q.get('select'), !q.has('view'));
    if (q.has('focus')) app.focus(q.get('focus'), true);
    V.renderNow();
    requestAnimationFrame(() => {
      V.renderNow();
      resolveReady(api);
    });
  };
  const fontsReady = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
  Promise.race([fontsReady, new Promise((r) => setTimeout(r, 1500))]).then(start);
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
