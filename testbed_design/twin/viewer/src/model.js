// Scene data indexing, signal tracing and small helpers. No three.js here.

export const CLASS_ORDER = ['AC', 'DC', 'POE', 'FIB', 'ETH', 'RF', 'GNSS', 'WG', 'AUX', 'PE'];

// Display palette per cable class. The scene's own colours put three pairs too
// close for print or a projector (DC/POE, FIB/GNSS, AC/WG); the viewer and the
// guide both use this palette, and DC, GNSS and WG are also drawn banded.
export const DISPLAY_COLORS = {
  AC: '#c0392b', DC: '#f2c200', POE: '#e67e22', FIB: '#16a085', ETH: '#2980b9',
  RF: '#8e44ad', GNSS: '#8bc34a', WG: '#ff4da6', AUX: '#7f8c8d', PE: '#2e7d32',
};
export const BANDED = new Set(['DC', 'GNSS', 'WG']);

export function buildModel(scene) {
  for (const c of scene.cables) {
    if (c.color_scene == null) c.color_scene = c.color;
    if (DISPLAY_COLORS[c.cls]) c.color = DISPLAY_COLORS[c.cls];
  }
  for (const l of scene.legend || []) {
    if (l.color_scene == null) l.color_scene = l.color;
    if (DISPLAY_COLORS[l.cls]) l.color = DISPLAY_COLORS[l.cls];
  }
  const devices = new Map(scene.devices.map((d) => [d.id, d]));
  const ports = new Map(scene.ports.map((p) => [p.id, p]));
  const cables = new Map(scene.cables.map((c) => [c.id, c]));

  const portCable = new Map();
  for (const c of scene.cables) {
    portCable.set(c.a, c.id);
    portCable.set(c.b, c.id);
  }
  const devCables = new Map(scene.devices.map((d) => [d.id, []]));
  for (const c of scene.cables) {
    for (const pid of [c.a, c.b]) {
      const p = ports.get(pid);
      if (p && devCables.has(p.device)) {
        const l = devCables.get(p.device);
        if (!l.includes(c.id)) l.push(c.id);
      }
    }
  }
  const children = new Map();
  for (const d of scene.devices) {
    if (d.parent) {
      if (!children.has(d.parent)) children.set(d.parent, []);
      children.get(d.parent).push(d.id);
    }
  }
  // port name -> port id, per device
  const portByName = new Map();
  for (const d of scene.devices) {
    const m = new Map();
    for (const pid of d.ports || []) {
      const p = ports.get(pid);
      if (p) m.set(p.name, pid);
    }
    portByName.set(d.id, m);
  }

  // installation steps
  const steps = scene.steps || [];
  const devStep = new Map();
  const cabStep = new Map();
  for (const st of steps) {
    for (const id of st.devices) if (!devStep.has(id)) devStep.set(id, st.n);
    for (const id of st.cables) if (!cabStep.has(id)) cabStep.set(id, st.n);
  }
  let tableStep = null;
  for (const st of steps) {
    if (st.devices.some((id) => devices.get(id) && devices.get(id).group === 'table')) {
      tableStep = st.n;
      break;
    }
  }

  const switched = new Set((scene.meta && scene.meta.switched) || []);

  const model = {
    scene, devices, ports, cables, portCable, devCables, children, portByName,
    steps, devStep, cabStep, tableStep, switched,
  };
  model.overlaps = findOverlaps(scene.devices);
  model.dark = computeDark(model);
  model.allIds = [...cables.keys(), ...devices.keys(), ...ports.keys()];
  return model;
}

export function cableEndsDevices(model, c) {
  const pa = model.ports.get(c.a);
  const pb = model.ports.get(c.b);
  return [pa ? pa.device : null, pb ? pb.device : null];
}

export function otherEnd(c, pid) {
  return c.a === pid ? c.b : c.a;
}

// --- signal trace through passive devices --------------------------------

function internalPeers(model, pid) {
  const p = model.ports.get(pid);
  if (!p) return [];
  const d = model.devices.get(p.device);
  if (!d) return [];
  const byName = model.portByName.get(d.id);
  const pick = (...names) => names.map((n) => byName.get(n)).filter(Boolean);
  const n = p.name;
  let m;
  switch (d.kind) {
    case 'ztdat':
    case 'mwatt':
      m = /^ch(\d+)\.(in|out)$/i.exec(n);
      if (m) return pick(`ch${m[1]}.${m[2].toLowerCase() === 'in' ? 'out' : 'in'}`);
      return [];
    case 'splitter':
      if (n === 'S') return [...byName.keys()].filter((k) => /^P\d+$/.test(k)).map((k) => byName.get(k));
      if (/^P\d+$/.test(n)) return pick('S');
      return [];
    case 'coupler':
      if (n === 'IN') return pick('THRU', 'CPL');
      if (n === 'THRU' || n === 'CPL') return pick('IN');
      return [];
    case 'transition':
      if (n === 'wg') return pick('coax');
      if (n === 'coax') return pick('wg');
      return [];
    case 'padstack':
      // fixed SMA pad stack: IN is screwed onto its host port (port.mate), OUT takes the cable
      if (n === 'in') return pick('out');
      if (n === 'out') return pick('in');
      return [];
    case 'gnss':
      if (n === 'in') return [...byName.keys()].filter((k) => /^out\d+$/.test(k)).map((k) => byName.get(k));
      if (/^out\d+$/.test(n)) return pick('in');
      return [];
    default:
      return [];
  }
}

export function isPassive(kind) {
  return ['ztdat', 'mwatt', 'splitter', 'coupler', 'transition', 'gnss', 'padstack'].includes(kind);
}

// Ports screwed straight onto each other (a pad stack's IN on a tile or a
// splitter port) carry `mate` instead of a cable.
export function mateOf(model, pid) {
  const p = model.ports.get(pid);
  return p && p.mate && model.ports.has(p.mate) ? p.mate : null;
}

// The cable on a port, or (for a port with a pad stack screwed onto it) the
// cable on the far side of that stack.
export function cableAt(model, pid) {
  const cid = model.portCable.get(pid);
  if (cid) return cid;
  const m = mateOf(model, pid);
  if (!m) return null;
  for (const q of internalPeers(model, m)) {
    const c = model.portCable.get(q);
    if (c) return c;
  }
  return null;
}

// Returns { paths: [[token]], cables:Set, devices:Set, ports:Set }.
// token = {t:'port'|'cable'|'dev', id, from?, to?, open?}
export function traceCable(model, cableId) {
  const start = model.cables.get(cableId);
  const res = { paths: [], cables: new Set(), devices: new Set(), ports: new Set(), truncated: false };
  if (!start) return res;
  const visited = new Set([cableId]);
  const mated = new Set();
  let budget = 400;

  function explore(pid, depth) {
    budget -= 1;
    const peers = internalPeers(model, pid);
    if (!peers.length || depth > 24 || budget < 0) return [[{ t: 'port', id: pid }]];
    const dev = model.ports.get(pid).device;
    const out = [];
    for (const q of peers) {
      const tok = { t: 'dev', id: dev, from: pid, to: q };
      const cq = model.portCable.get(q);
      const mq = cq ? null : mateOf(model, q);
      if (mq && !mated.has(q)) {
        // screwed straight onto the next part (pad stack on a tile or splitter port): no cable between
        mated.add(q);
        mated.add(mq);
        res.ports.add(q);
        res.ports.add(mq);
        for (const tail of explore(mq, depth + 1)) out.push([tok, ...tail]);
        continue;
      }
      if (!cq || visited.has(cq)) {
        out.push([tok, { t: 'port', id: q, open: !cq }]);
        continue;
      }
      visited.add(cq);
      const c = model.cables.get(cq);
      for (const tail of explore(otherEnd(c, q), depth + 1)) out.push([tok, { t: 'cable', id: cq }, ...tail]);
    }
    return out;
  }
  const sideA = explore(start.a, 0);
  const sideB = explore(start.b, 0);
  for (const a of sideA) {
    for (const b of sideB) {
      if (res.paths.length >= 16) {
        res.truncated = true;
        break;
      }
      // side A is read backwards: swap each passive device's in/out so the chain reads end to end
      const ra = a.slice().reverse().map((tok) => (tok.t === 'dev' ? { ...tok, from: tok.to, to: tok.from } : tok));
      res.paths.push([...ra, { t: 'cable', id: cableId }, ...b]);
    }
  }
  for (const path of res.paths) {
    for (const tok of path) {
      if (tok.t === 'cable') {
        res.cables.add(tok.id);
        const c = model.cables.get(tok.id);
        res.ports.add(c.a);
        res.ports.add(c.b);
      } else if (tok.t === 'dev') {
        res.devices.add(tok.id);
      } else if (tok.t === 'port') {
        res.ports.add(tok.id);
        const p = model.ports.get(tok.id);
        if (p) res.devices.add(p.device);
      }
    }
  }
  return res;
}

export function tokenLabel(model, tok) {
  if (tok.t === 'cable') return tok.id;
  if (tok.t === 'port') return tok.id + (tok.open ? ' (open)' : '');
  const d = model.devices.get(tok.id);
  const pf = model.ports.get(tok.from);
  const pt = model.ports.get(tok.to);
  if (!d || !pf) return tok.id;
  if (d.kind === 'ztdat' || d.kind === 'mwatt') {
    const m = /^ch(\d+)/i.exec(pf.name);
    return `${d.id} CH${m ? m[1] : '?'}`;
  }
  if (d.kind === 'padstack') return `${d.id} (${d.text || d.db} dB)`;
  if (d.kind === 'coupler' && pt) return `${d.id} (${pf.name}/${pt.name})`;
  return d.id;
}

// --- after t0: what goes dark downstream of the switched devices ------------
// From each switched device follow its POE / DC feed cables to what they power
// (through adaptors, injectors and PSUs). A fed Spider-box node goes dark with
// its tiles (modules whose sector starts with the node name); a fed radio goes
// dark as a whole. Every cable on a dark port, and every cable the signal trace
// reaches from those ports, is dead after t0.
const SOURCE_KINDS = new Set(['dcdist', 'fuse', 'rect', 'pdu', 'strip', 'wall']);
const FEEDER_KINDS = new Set(['adaptor', 'injector', 'psu']);

export function computeDark(model) {
  const res = {
    devices: new Set(), ports: new Set(), cables: new Set(), nodes: [], links: new Set(),
    why: new Map(), // dark device / node key -> {from: switched id, via: [cable ids]}
  };
  const sw = model.switched;
  if (!sw || !sw.size) return res;
  const darkPortsOfNode = (box, node) => {
    const d = model.devices.get(box);
    const out = [];
    const secs = new Set((d.modules || []).filter((m) => String(m.sector || '').split('.')[0] === node).map((m) => m.m));
    for (const pid of d.ports || []) {
      const p = model.ports.get(pid);
      if (!p) continue;
      const head = p.name.split('.')[0];
      if (head === node || secs.has(head)) out.push(pid);
    }
    return out;
  };
  const queue = [...sw].map((id) => ({ id, from: id, via: [] }));
  const seen = new Set(sw);
  const seedPorts = [];
  while (queue.length) {
    const { id, from, via } = queue.shift();
    for (const cid of model.devCables.get(id) || []) {
      const c = model.cables.get(cid);
      if (c.cls !== 'POE' && c.cls !== 'DC') continue;
      const mine = [c.a, c.b].find((pid) => (model.ports.get(pid) || {}).device === id);
      if (!mine) continue;
      const other = otherEnd(c, mine);
      const op = model.ports.get(other);
      if (!op) continue;
      const od = model.devices.get(op.device);
      if (!od || seen.has(od.id) || SOURCE_KINDS.has(od.kind)) continue;
      // a feeder only passes the outage on when the cable lands on its input
      if (FEEDER_KINDS.has(od.kind) && /^(rj45|poe|out)$/i.test(op.name)) continue;
      const path = [...via, cid];
      if (od.kind === 'spider') {
        const node = op.name.split('.')[0];
        const key = `${od.id}.${node}`;
        if (res.why.has(key)) continue;
        res.why.set(key, { from, via: path });
        res.nodes.push({ box: od.id, node, port: other });
        for (const pid of darkPortsOfNode(od.id, node)) {
          res.ports.add(pid);
          seedPorts.push(pid);
        }
        continue;
      }
      seen.add(od.id);
      res.devices.add(od.id);
      const hop = /^IP50-(.+)@/.exec(od.id);
      if (hop) res.links.add(`IP-50 ${hop[1]}`);
      res.why.set(od.id, { from, via: path });
      for (const pid of od.ports || []) seedPorts.push(pid);
      if (FEEDER_KINDS.has(od.kind)) queue.push({ id: od.id, from, via: path });
    }
  }
  // a Spider box whose every node is dark goes dark as a whole
  const boxes = new Map();
  for (const n of res.nodes) {
    if (!boxes.has(n.box)) boxes.set(n.box, new Set());
    boxes.get(n.box).add(n.node);
  }
  for (const [box, nodes] of boxes) {
    const d = model.devices.get(box);
    const all = new Set((d.nodes || []).length ? d.nodes : [...nodes]);
    if ([...all].every((n) => nodes.has(n))) res.devices.add(box);
  }
  for (const pid of seedPorts) {
    // a dark tile may carry a pad stack: its cable starts on the stack
    const cid = cableAt(model, pid);
    if (!cid || res.cables.has(cid)) continue;
    const c = model.cables.get(cid);
    res.cables.add(cid);
    if (['RF', 'WG', 'GNSS'].includes(c.cls)) {
      const t = traceCable(model, cid);
      for (const x of t.cables) res.cables.add(x);
    }
  }
  for (const cid of res.cables) {
    const l = linkOfBundle(model.cables.get(cid).bundle);
    if (l && l !== 'spl') res.links.add(l);
  }
  return res;
}

// --- geometry sanity (viewer-side) -----------------------------------------

function findOverlaps(devs) {
  const out = [];
  const byId = new Map(devs.map((d) => [d.id, d]));
  // a part on a part on a shelf (a pad stack on a splitter on the tray) sits inside its ancestors' boxes
  const ancestor = (a, d) => {
    for (let p = d.parent, k = 0; p && k < 8; p = (byId.get(p) || {}).parent, k++) if (p === a.id) return true;
    return false;
  };
  for (let i = 0; i < devs.length; i++) {
    for (let j = i + 1; j < devs.length; j++) {
      const a = devs[i];
      const b = devs[j];
      if (ancestor(a, b) || ancestor(b, a)) continue;
      let ok = true;
      for (let k = 0; k < 3; k++) {
        const lo = Math.max(a.pos[k], b.pos[k]);
        const hi = Math.min(a.pos[k] + a.size[k], b.pos[k] + b.size[k]);
        if (!(lo < hi - 0.5)) {
          ok = false;
          break;
        }
      }
      if (ok) out.push([a.id, b.id]);
    }
  }
  return out;
}

export function linkOfBundle(bundle) {
  if (!bundle) return null;
  const m = /^\('([^']+)'/.exec(bundle);
  return m ? m[1] : null;
}

export function shortPortName(p) {
  return p.name;
}

export function fmtMM(v) {
  if (v == null || Number.isNaN(v)) return '?';
  return `${Math.round(v)} mm`;
}

export function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}
