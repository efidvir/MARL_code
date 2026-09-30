const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

// ---------- palette ----------
const NAVY = "0B1D33";
const NAVY2 = "14304F";
const INK = "1A2B3C";
const MUTED = "5A6B7D";
const TINT = "F2F5F8";
const LINE = "D5DDE5";
const WHITE = "FFFFFF";
// domain colour code, used on every slide
const C_INTENT = "7C5CFF";
const C_SDN = "0FA3B1";
const C_SIM = "E8912D";
const C_HW = "22A06B";

const HEAD = "Arial";
const BODY = "Calibri";

async function icon(Comp, color) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(Comp, { color: "#" + color, size: "256" })
  );
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

// icon in a coloured circle: the deck's visual motif
async function iconCircle(slide, pres, Comp, x, y, d, bg, fg = WHITE) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: bg }, line: { color: bg } });
  const pad = d * 0.25;
  slide.addImage({ data: await icon(Comp, fg), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad });
}

function title(slide, kicker, text) {
  slide.addText(kicker.toUpperCase(), {
    x: 0.5, y: 0.3, w: 9, h: 0.25, fontFace: HEAD, fontSize: 10, bold: true,
    color: C_SDN, charSpacing: 2, margin: 0, isTextBox: true,
  });
  slide.addText(text, {
    x: 0.5, y: 0.55, w: 9, h: 0.6, fontFace: HEAD, fontSize: 24, bold: true,
    color: NAVY, margin: 0, valign: "middle", isTextBox: true,
  });
}

function txt(slide, text, o) {
  slide.addText(text, Object.assign({ fontFace: BODY, fontSize: 11, color: INK, margin: 0, isTextBox: true, valign: "top" }, o));
}

function arrow(slide, pres, x1, y1, x2, y2, color = MUTED, both = false) {
  const flipH = x2 < x1, flipV = y2 < y1;
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH, flipV,
    line: { color, width: 1.75, endArrowType: "triangle", beginArrowType: both ? "triangle" : "none" },
  });
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625
  pres.title = "Digital Twin Adapters & Standards";

  // =====================================================================
  // 1. Title
  // =====================================================================
  {
    const s = pres.addSlide();
    s.background = { color: NAVY };
    txt(s, "TELECOM DIGITAL TWIN ARCHITECTURE", { x: 0.6, y: 0.9, w: 5.6, h: 0.3, fontFace: HEAD, fontSize: 11, bold: true, color: C_SDN, charSpacing: 3 });
    txt(s, "Digital Twin Adapters & Standards", { x: 0.6, y: 1.3, w: 5.6, h: 1.5, fontFace: HEAD, fontSize: 38, bold: true, color: WHITE, valign: "middle" });
    txt(s, "How operator intent, ETSI TeraFlowSDN, ns-3 co-simulation and Ceragon microwave hardware work as one closed loop", { x: 0.6, y: 2.95, w: 5.4, h: 0.8, fontSize: 15, color: "C9D6E3" });
    txt(s, "Ceragon Networks  ·  ETSI TeraFlowSDN research  ·  Standalone & production hybrid architecture", { x: 0.6, y: 4.75, w: 8.8, h: 0.3, fontSize: 10, color: "8FA3B8" });

    // the four domains as a vertical chain
    const items = [
      [fa.FaUserTie, C_INTENT, "Intent", "Operator & SLA"],
      [fa.FaProjectDiagram, C_SDN, "SDN controller", "ETSI TeraFlowSDN"],
      [fa.FaFlask, C_SIM, "Simulator", "ns-3 discrete events"],
      [fa.FaBroadcastTower, C_HW, "Hardware", "Ceragon MH-T261"],
    ];
    const x0 = 6.9, y0 = 0.85, step = 1.05, d = 0.62;
    s.addShape(pres.shapes.LINE, { x: x0 + d / 2, y: y0 + d / 2, w: 0, h: step * 3, line: { color: "3A5575", width: 2, dashType: "dash" } });
    for (let i = 0; i < items.length; i++) {
      const [I, c, a, b] = items[i];
      await iconCircle(s, pres, I, x0, y0 + i * step, d, c);
      txt(s, a, { x: x0 + 0.8, y: y0 + i * step + 0.06, w: 2.2, h: 0.28, fontFace: HEAD, fontSize: 13, bold: true, color: WHITE });
      txt(s, b, { x: x0 + 0.8, y: y0 + i * step + 0.33, w: 2.2, h: 0.25, fontSize: 10.5, color: "8FA3B8" });
    }
    s.addNotes(
      "What are the adapters? Software translators that bridge the different timing, data formats and control protocols of the four parts of the twin.\n" +
      "Standards: TM Forum TMF921, 3GPP TS 28.561, ITU-T Y.3090, IRTF NMRG digital-twin drafts, IETF RFC 8040 RESTCONF / RFC 7950 YANG 1.1, with ETSI TeraFlowSDN as the SDN controller.\n" +
      "Closed-loop automation: every change is predicted in a discrete-event what-if run before it is actuated on hardware with a two-phase commit."
    );
  }

  // =====================================================================
  // 2. The problem: four timescales, four languages
  // =====================================================================
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    title(s, "The problem", "Four components, four languages, four clocks");

    const lo = -6.3, hi = 5.3, X0 = 3.75, X1 = 9.5;
    const X = (v) => X0 + ((v - lo) / (hi - lo)) * (X1 - X0);
    const rowY = 1.45, rowH = 0.75, nRows = 4;
    const axisY = rowY + nRows * rowH + 0.05;

    // gridlines + tick labels
    const ticks = [[-6, "1 µs"], [-3, "1 ms"], [0, "1 s"], [Math.log10(60), "1 min"], [Math.log10(3600), "1 h"], [Math.log10(86400), "1 day"]];
    for (const [v, lab] of ticks) {
      s.addShape(pres.shapes.LINE, { x: X(v), y: rowY - 0.1, w: 0, h: axisY - rowY + 0.1, line: { color: LINE, width: 0.75 } });
      txt(s, lab, { x: X(v) - 0.4, y: axisY + 0.05, w: 0.8, h: 0.25, fontSize: 10, color: MUTED, align: "center" });
    }
    s.addShape(pres.shapes.LINE, { x: X0, y: axisY, w: X1 - X0, h: 0, line: { color: MUTED, width: 1 } });
    txt(s, "Operating timescale (log)", { x: X0, y: axisY + 0.32, w: X1 - X0, h: 0.25, fontSize: 9.5, italic: true, color: MUTED, align: "center" });

    const rows = [
      [fa.FaUserTie, C_INTENT, "Declarative intent", "Natural language / JSON", Math.log10(60), Math.log10(86400), "minutes – days"],
      [fa.FaProjectDiagram, C_SDN, "SDN controller", "Context DB · gRPC · CSPF", -0.3, Math.log10(60), "seconds"],
      [fa.FaFlask, C_SIM, "Discrete simulator", "C++ event queue & packets", -6, -3, "microseconds"],
      [fa.FaBroadcastTower, C_HW, "Physical hardware", "RESTCONF · PHY registers", -5, -3, "sub-ms"],
    ];
    for (let i = 0; i < rows.length; i++) {
      const [I, c, name, speaks, a, b, lab] = rows[i];
      const y = rowY + i * rowH;
      await iconCircle(s, pres, I, 0.5, y + 0.1, 0.5, c);
      txt(s, name, { x: 1.15, y: y + 0.08, w: 2.5, h: 0.28, fontFace: HEAD, fontSize: 12.5, bold: true, color: INK });
      txt(s, speaks, { x: 1.15, y: y + 0.36, w: 2.5, h: 0.25, fontSize: 10.5, color: MUTED });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: X(a), y: y + 0.17, w: X(b) - X(a), h: 0.36, fill: { color: c }, line: { color: c }, rectRadius: 0.08 });
      txt(s, lab, { x: X(a), y: y + 0.17, w: X(b) - X(a), h: 0.36, fontSize: 10.5, bold: true, color: WHITE, align: "center", valign: "middle" });
    }
    // take-away
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 4.85, w: 3.05, h: 0.5, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    txt(s, [
      { text: "Adapters ", options: { bold: true, color: NAVY } },
      { text: "translate both the protocol and the clock between neighbours.", options: { color: INK } },
    ], { x: 0.62, y: 4.85, w: 2.85, h: 0.5, fontSize: 10.5, valign: "middle" });

    s.addNotes(
      "In a telecom digital twin the components speak different languages and run on very different timescales. Adapters bridge that semantic and temporal gap.\n" +
      "1. Declarative intent (human operator / SLA): natural language or JSON, minutes to days. Example: 'Ensure URLLC latency < 1.5 ms and 99.999% availability during storms'.\n" +
      "2. SDN controller (ETSI TeraFlowSDN): Context DB, gRPC, CSPF; seconds (control plane). Data: device UUIDs, endpoints, topology graphs, TI-LFA backup paths.\n" +
      "3. Discrete simulator (ns-3 core): C++ event queue and packets; microseconds. Data: QueueDisc buffer drops, Wi-Fi EDCA AC_VO, link propagation delay.\n" +
      "4. Physical hardware (Ceragon MH-T261): RFC 8040 RESTCONF and PHY; sub-millisecond. Data: V-band 60.48 GHz, ACM MCS 0-9, RSSI/SNR, RF registers."
    );
  }

  // =====================================================================
  // 3. Reference architecture
  // =====================================================================
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    title(s, "Architecture at a glance", "Three adapters connect four domains in one loop");

    // layer frames
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.95, y: 1.3, w: 5.6, h: 1.45, fill: { color: "EAF7F8" }, line: { color: C_SDN, width: 1, dashType: "dash" }, rectRadius: 0.1 });
    txt(s, "DIGITAL TWIN LAYER", { x: 4.05, y: 1.34, w: 3, h: 0.2, fontSize: 8.5, bold: true, color: C_SDN, charSpacing: 1.5 });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.95, y: 3.8, w: 5.6, h: 1.3, fill: { color: "EAF6F0" }, line: { color: C_HW, width: 1, dashType: "dash" }, rectRadius: 0.1 });
    txt(s, "PHYSICAL NETWORK LAYER", { x: 6.05, y: 4.0, w: 3, h: 0.2, fontSize: 8.5, bold: true, color: C_HW, charSpacing: 1.5 });
    txt(s, "Layers follow ITU-T Y.3090; twin lifecycle follows 3GPP TS 28.561", { x: 6.05, y: 4.3, w: 3.35, h: 0.5, fontSize: 10, italic: true, color: MUTED });

    async function box(x, y, w, I, c, name, sub) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.0, fill: { color: WHITE }, line: { color: c, width: 1.5 }, rectRadius: 0.1, shadow: { type: "outer", blur: 4, offset: 1, angle: 90, color: "000000", opacity: 0.12 } });
      await iconCircle(s, pres, I, x + w / 2 - 0.18, y + 0.1, 0.36, c);
      txt(s, name, { x, y: y + 0.5, w, h: 0.24, fontFace: HEAD, fontSize: 11, bold: true, align: "center" });
      txt(s, sub, { x, y: y + 0.74, w, h: 0.2, fontSize: 9, color: MUTED, align: "center" });
    }
    function pill(x, y, w, label, std) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.4, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.2 });
      txt(s, label, { x, y, w, h: 0.4, fontSize: 10, bold: true, color: WHITE, align: "center", valign: "middle" });
      if (std) txt(s, std, { x: x - 0.1, y: y + 0.45, w: w + 0.2, h: 0.35, fontSize: 9, color: MUTED, align: "center" });
    }

    const rowA = 1.6;
    await box(0.5, rowA, 1.75, fa.FaUserTie, C_INTENT, "Operator intent", "SLA · natural language");
    await box(4.15, rowA, 1.75, fa.FaProjectDiagram, C_SDN, "TeraFlowSDN", "ETSI OSG SDN controller");
    await box(7.65, rowA, 1.75, fa.FaFlask, C_SIM, "ns-3 simulator", "discrete-event core");
    await box(4.15, 3.95, 1.75, fa.FaBroadcastTower, C_HW, "Ceragon MH-T261", "60 GHz V-band radio");

    // adapter 1
    arrow(s, pres, 2.25, rowA + 0.5, 4.15, rowA + 0.5, NAVY);
    pill(2.5, rowA + 0.3, 1.45, "① Intent", "TM Forum TMF921");
    // adapter 2
    arrow(s, pres, 5.9, rowA + 0.5, 7.65, rowA + 0.5, NAVY, true);
    pill(6.05, rowA + 0.3, 1.45, "② Simulation", "IRTF NMRG twin interface");
    // adapter 3
    arrow(s, pres, 5.025, rowA + 1.0, 5.025, 3.95, NAVY, true);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.25, y: 3.07, w: 1.55, h: 0.4, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.2 });
    txt(s, "③ Southbound", { x: 4.25, y: 3.07, w: 1.55, h: 0.4, fontSize: 10, bold: true, color: WHITE, align: "center", valign: "middle" });
    txt(s, "RFC 8040 RESTCONF · YANG 1.1", { x: 5.95, y: 3.12, w: 2.6, h: 0.3, fontSize: 9, color: MUTED });

    // key number
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 3.35, w: 3.0, h: 1.75, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.1 });
    txt(s, "≈ 2 s", { x: 0.7, y: 3.5, w: 2.6, h: 0.75, fontFace: HEAD, fontSize: 40, bold: true, color: C_SDN, valign: "middle" });
    txt(s, "from SLA intent to a verified configuration change on the live radio", { x: 0.7, y: 4.3, w: 2.6, h: 0.7, fontSize: 11, color: INK });

    s.addNotes(
      "Intent adapter: translates natural-language or SLA bounds into TeraFlowSDN service and CSPF requests (TM Forum TMF921).\n" +
      "Simulation adapter: syncs the live TFS graph into the ns-3 C++ discrete-event engine, scopes fidelity (tiers 1-5) and returns microsecond traces. The IRTF NMRG network-digital-twin drafts describe this twin-to-simulator interface; they are research drafts, not IETF standards.\n" +
      "Southbound driver: stages candidate-datastore diffs and runs two-phase commits to the Ceragon MH-T261 over RFC 8040 RESTCONF.\n" +
      "ITU-T Y.3090 defines the layered twin reference model (physical network layer, digital twin layer, application layer). 3GPP TS 28.561 defines management of the network digital twin and its closed loop."
    );
  }

  // =====================================================================
  // 4. Standards map
  // =====================================================================
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    title(s, "Standards", "Each interface is anchored in an open specification");

    const cards = [
      [fa.FaBullseye, C_INTENT, "TM Forum TMF921", "BUSINESS & SLA", "Intent Management API",
        "Operator states latency < 1.5 ms and availability. Intent state: Acknowledged → In evaluation → Compliant."],
      [fa.FaSyncAlt, C_SDN, "3GPP TS 28.561", "TWIN LIFECYCLE", "Digital twin management",
        "Governs the 4-gate pre-flight check before any action reaches the network."],
      [fa.FaLayerGroup, NAVY2, "ITU-T Y.3090", "ARCHITECTURE", "Twin network architecture",
        "Layered reference model; keeps the real-time shadow datastore in sync with the physical network."],
      [fa.FaFlask, C_SIM, "IRTF NMRG drafts", "SIMULATION INTERFACE", "Twin–simulator interface",
        "East-west twin ↔ simulator link: inject what-ifs (surge, rain fade), stream telemetry back."],
      [fa.FaBroadcastTower, C_HW, "IETF RFC 8040 / 7950", "HARDWARE INTERFACE", "RESTCONF & YANG 1.1",
        "Southbound to the MH-T261: candidate staging, diff validation, two-phase commit."],
      [fa.FaProjectDiagram, C_SDN, "ETSI TeraFlowSDN", "SDN CONTROL", "Open-source SDN controller",
        "gRPC microservices: Context :10010, Device :10020, Service/PCE :10030, Monitoring :10040."],
    ];
    const cw = 2.85, ch = 1.85, gx = 0.225, gy = 0.2, x0 = 0.5, y0 = 1.35;
    for (let i = 0; i < cards.length; i++) {
      const [I, c, code, tag, name, use] = cards[i];
      const x = x0 + (i % 3) * (cw + gx), y = y0 + Math.floor(i / 3) * (ch + gy);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
      await iconCircle(s, pres, I, x + 0.18, y + 0.18, 0.46, c);
      txt(s, code, { x: x + 0.78, y: y + 0.17, w: cw - 0.9, h: 0.27, fontFace: HEAD, fontSize: 12.5, bold: true, color: NAVY });
      txt(s, tag, { x: x + 0.78, y: y + 0.45, w: cw - 0.9, h: 0.2, fontSize: 8.5, bold: true, color: c, charSpacing: 1 });
      txt(s, name, { x: x + 0.18, y: y + 0.78, w: cw - 0.36, h: 0.25, fontSize: 11, bold: true, color: INK });
      txt(s, use, { x: x + 0.18, y: y + 1.05, w: cw - 0.36, h: 0.72, fontSize: 10, color: MUTED });
    }
    s.addNotes(
      "TMF921: REST API for declarative intent and SLA expectations. The operator sets latency and availability without low-level scripts.\n" +
      "3GPP TS 28.561 (Management aspects of network digital twins): twin lifecycle and closed-loop control. In our platform it governs the 4-phase pre-flight gate: latency envelope, queue stability, hardware capability, spectral mask.\n" +
      "ITU-T Y.3090 (Digital twin network: requirements and architecture): layered reference model and synchronisation between the physical network and the twin. Powers our real-time shadow datastore.\n" +
      "IRTF NMRG (Network Management Research Group) drafts: network-digital-twin architecture and interface drafts. Used as the model for the east-west protocol that injects what-if perturbations and streams discrete telemetry back. Note: these are research drafts, not standards.\n" +
      "RFC 8040 / RFC 7950: RESTCONF over HTTP to YANG datastores. Southbound interface to the Ceragon MH-T261 (192.168.1.225:80).\n" +
      "ETSI TeraFlowSDN: open-source SDN controller hosted by an ETSI Open Source Group (not a standard itself). Context (gRPC :10010), Device driver (:10020), Service/PCE (:10030), Monitoring telemetry (:10040)."
    );
  }

  // =====================================================================
  // 5. Adapter 1 — simulation adapter
  // =====================================================================
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    title(s, "Adapter ② · TFS → ns-3", "The simulation adapter mirrors the live network in ns-3");

    const steps = [
      [fa.FaSyncAlt, "1  Reconcile", "Pull the live 34-node graph and RF metrics (RSSI −58 dBm, MCS 8)"],
      [fa.FaFilter, "2  Scope", "Fidelity governor trims the graph to the domain that matters"],
      [fa.FaCloudShowersHeavy, "3  Perturb", "Inject what-ifs: 3.5× traffic surge, ITU-R P.838 rain fade"],
      [fa.FaStream, "4  Ingest", "Stream queues, drops, latency and the NetAnim trace back to the twin"],
    ];
    const sw = 2.1, sg = 0.2, sy = 1.35, d = 0.5;
    s.addShape(pres.shapes.LINE, { x: 0.5 + d / 2, y: sy + d / 2, w: 3 * (sw + sg), h: 0, line: { color: LINE, width: 2 } });
    for (let i = 0; i < steps.length; i++) {
      const [I, t, dsc] = steps[i];
      const x = 0.5 + i * (sw + sg);
      await iconCircle(s, pres, I, x, sy, d, C_SIM);
      txt(s, t, { x, y: sy + 0.6, w: sw, h: 0.28, fontFace: HEAD, fontSize: 12, bold: true, color: NAVY });
      txt(s, dsc, { x, y: sy + 0.9, w: sw, h: 0.62, fontSize: 10, color: MUTED });
    }

    // governor chart
    s.addChart(pres.charts.BAR, [{
      name: "Fidelity",
      labels: ["T1 Macro-topology", "T2 Queue dynamics", "T3 Physical RF", "T4 URLLC micro-packet", "T5 Green energy"],
      values: [25, 55, 75, 95, 20],
    }], {
      x: 0.4, y: 3.0, w: 5.4, h: 2.3, barDir: "bar",
      chartColors: ["C9CFD6", "C9CFD6", "C9CFD6", C_SIM, "C9CFD6"],
      showTitle: true, title: "Governor resolution tiers (relative fidelity)", titleFontFace: HEAD, titleFontSize: 11, titleColor: NAVY,
      catAxisOrientation: "maxMin", catAxisLabelFontFace: BODY, catAxisLabelFontSize: 10, catAxisLabelColor: INK,
      catAxisLineShow: false, valAxisHidden: true, valAxisMaxVal: 110, valAxisMinVal: 0,
      valGridLine: { style: "none" }, catGridLine: { style: "none" },
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', dataLabelFontSize: 10, dataLabelColor: INK,
      barGapWidthPct: 45, showLegend: false,
    });

    // stats
    const stats = [["34 → 23", "nodes simulated when scoped to Tier 4 (URLLC)"], ["≈ 1.15 s", "per what-if run, instead of minutes for the full graph"]];
    for (let i = 0; i < stats.length; i++) {
      const y = 3.05 + i * 1.15;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.1, y, w: 3.4, h: 1.0, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
      txt(s, stats[i][0], { x: 6.3, y: y + 0.08, w: 3.1, h: 0.5, fontFace: HEAD, fontSize: 24, bold: true, color: C_SIM, valign: "middle" });
      txt(s, stats[i][1], { x: 6.3, y: y + 0.58, w: 3.1, h: 0.36, fontSize: 10, color: INK });
    }
    s.addNotes(
      "Step 1: the adapter queries the TeraFlowSDN Context service for the active 34-node graph, link bandwidths and live Ceragon RF metrics.\n" +
      "Step 2: the simulation resolution governor scopes the graph to the target domain instead of simulating all 34 nodes. Example: Tier 4 (URLLC micro-packet) uses 23 nodes and runs in about 1150 ms.\n" +
      "Step 3: what-if stress is injected into ns-3: a 3.5x bulk traffic surge (bufferbloat) or ITU-R P.838 rain fade (ACM rate falls from 1 Gbps to 100 Mbps).\n" +
      "Step 4: ns-3 simulates microsecond packet events on cersrv-029, writes /tmp/tsn_wifi_ceragon_anim.xml, and the adapter streams queues, drops and latencies back to the twin.\n" +
      "Implementation: remote execution over SSH/JSON to ns-3.45 on cersrv-029, with an analytical local fallback for standalone mode. NetAnim AnimationInterface gives timeline playback (0.000-2.500 s) with 10 buffer slots and tail-drop alerts. Deterministic TSN mapping: 802.1Q PCP class 6 and EDCA AC_VO go to PfifoFast band 0."
    );
  }

  // =====================================================================
  // 6. Storm scenario
  // =====================================================================
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    title(s, "What-if in practice", "A storm on the V-band link, tested before it happens");

    s.addChart(pres.charts.BAR, [{
      name: "Rate", labels: ["Clear sky", "Rain · MCS 0 floor", "Rain · MCS 4 floor"], values: [1000, 100, 500],
    }], {
      x: 0.4, y: 1.25, w: 4.9, h: 3.95, barDir: "col",
      chartColors: ["9AA7B4", "D64545", C_HW],
      showTitle: true, title: "Link rate (Mbps)", titleFontFace: HEAD, titleFontSize: 11, titleColor: NAVY,
      catAxisLabelFontFace: BODY, catAxisLabelFontSize: 10, catAxisLabelColor: INK, catAxisLineShow: true,
      valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1150,
      valGridLine: { style: "none" }, catGridLine: { style: "none" },
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFontBold: true, dataLabelColor: INK,
      barGapWidthPct: 60, showLegend: false,
    });

    txt(s, "Change the twin recommends", { x: 5.7, y: 1.3, w: 3.8, h: 0.3, fontFace: HEAD, fontSize: 12.5, bold: true, color: NAVY });
    const diffs = [["ACM floor", "MCS 0", "MCS 4"], ["QoS queue", "FIFO", "PCP 6"], ["Carrier", "60.48 GHz", "64.80 GHz"]];
    for (let i = 0; i < diffs.length; i++) {
      const y = 1.72 + i * 0.52;
      txt(s, diffs[i][0], { x: 5.7, y, w: 1.1, h: 0.38, fontSize: 11, bold: true, valign: "middle" });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.85, y, w: 1.05, h: 0.38, fill: { color: "ECEFF2" }, line: { color: "ECEFF2" }, rectRadius: 0.06 });
      txt(s, diffs[i][1], { x: 6.85, y, w: 1.05, h: 0.38, fontSize: 10.5, color: MUTED, align: "center", valign: "middle" });
      arrow(s, pres, 7.95, y + 0.19, 8.3, y + 0.19, MUTED);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.35, y, w: 1.15, h: 0.38, fill: { color: "DDF3E8" }, line: { color: "DDF3E8" }, rectRadius: 0.06 });
      txt(s, diffs[i][2], { x: 8.35, y, w: 1.15, h: 0.38, fontSize: 10.5, bold: true, color: "157A50", align: "center", valign: "middle" });
    }

    // latency gauge
    const gy = 3.45;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.7, y: gy, w: 3.8, h: 1.75, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
    txt(s, "Predicted URLLC latency", { x: 5.9, y: gy + 0.15, w: 3.4, h: 0.25, fontSize: 11, bold: true });
    txt(s, [{ text: "1.12 ms", options: { fontSize: 24, bold: true, color: C_SDN } }, { text: "  of a 1.5 ms SLA", options: { fontSize: 11, color: MUTED } }],
      { x: 5.9, y: gy + 0.42, w: 3.4, h: 0.5, fontFace: HEAD, valign: "middle" });
    const bx = 5.9, bw = 3.4, by = gy + 1.05;
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: by, w: bw, h: 0.16, fill: { color: "D5DDE5" }, line: { color: "D5DDE5" } });
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: by, w: bw * (1.12 / 1.5), h: 0.16, fill: { color: C_SDN }, line: { color: C_SDN } });
    txt(s, "25% headroom → change passes the latency gate", { x: bx, y: by + 0.25, w: bw, h: 0.25, fontSize: 9.5, color: MUTED });

    s.addNotes(
      "Scenario values are the ones used in this deck: rain fade (ITU-R P.838) pushes adaptive coding and modulation down so the link rate falls from about 1 Gbps to 100 Mbps. Raising the ACM floor from MCS 0 to MCS 4 keeps at least 500 Mbps.\n" +
      "The same candidate configuration also moves URLLC traffic into IEEE 802.1Q PCP class 6 (PfifoFast band 0) and retunes the carrier from 60.48 GHz to 64.80 GHz.\n" +
      "ns-3 predicts 1.12 ms latency for this configuration, inside the 1.5 ms SLA, so it passes gate 1 of the pre-flight check."
    );
  }

  // =====================================================================
  // 7. Adapter 3 — southbound
  // =====================================================================
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    title(s, "Adapter ③ · TFS → hardware", "Stage → verify → commit, with automatic rollback");

    const stages = [
      [fa.FaEdit, C_INTENT, "STAGE 1", "Stage the candidate", [
        "Decision engine output becomes YANG config", "Held in the TFS candidate datastore, not live", "Diff checked against the running config"]],
      [fa.FaShieldAlt, C_SDN, "STAGE 2 · TS 28.561", "Pre-flight gates", [
        "Latency 1.12 ms ≤ 1.5 ms", "Buffer occupancy < 40%", "MH-T261 supports MCS 4", "64.80 GHz inside V-band mask"]],
      [fa.FaRocket, C_HW, "STAGE 3 · RESTCONF", "Two-phase commit", [
        "Prepare: PATCH the candidate", "Commit: POST ietf-netconf:commit", "Verify: poll telemetry"]],
    ];
    const cw = 2.75, gap = 0.375, cy = 1.3, ch = 2.75;
    for (let i = 0; i < stages.length; i++) {
      const [I, c, k, t, items] = stages[i];
      const x = 0.5 + i * (cw + gap);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: cy, w: cw, h: ch, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.1 });
      await iconCircle(s, pres, I, x + 0.2, cy + 0.2, 0.5, c);
      txt(s, k, { x: x + 0.85, y: cy + 0.22, w: cw - 0.95, h: 0.2, fontSize: 9, bold: true, color: c, charSpacing: 0.5 });
      txt(s, t, { x: x + 0.85, y: cy + 0.42, w: cw - 0.95, h: 0.3, fontFace: HEAD, fontSize: 11.5, bold: true, color: NAVY });
      for (let j = 0; j < items.length; j++) {
        const y = cy + 0.95 + j * 0.42;
        s.addImage({ data: await icon(i === 1 ? fa.FaCheckCircle : fa.FaChevronRight, c), x: x + 0.22, y: y + 0.06, w: 0.18, h: 0.18 });
        txt(s, items[j], { x: x + 0.5, y, w: cw - 0.65, h: 0.36, fontSize: 10.5, color: INK });
      }
      if (i < 2) arrow(s, pres, x + cw + 0.05, cy + ch / 2, x + cw + gap - 0.05, cy + ch / 2, NAVY);
    }
    // rollback loop
    const x1 = 0.5 + cw / 2, x3 = 0.5 + 2 * (cw + gap) + cw / 2, ry = 4.35;
    s.addShape(pres.shapes.LINE, { x: x3, y: cy + ch, w: 0, h: ry - (cy + ch), line: { color: "D64545", width: 1.5, dashType: "dash" } });
    s.addShape(pres.shapes.LINE, { x: x1, y: ry, w: x3 - x1, h: 0, line: { color: "D64545", width: 1.5, dashType: "dash" } });
    arrow(s, pres, x1, ry, x1, cy + ch, "D64545");
    s.addShape(pres.shapes.RECTANGLE, { x: 3.2, y: ry - 0.15, w: 3.6, h: 0.3, fill: { color: WHITE }, line: { color: WHITE } });
    txt(s, "Health check fails → automatic rollback", { x: 3.2, y: ry - 0.15, w: 3.6, h: 0.3, fontSize: 10, bold: true, color: "D64545", align: "center", valign: "middle" });

    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 4.65, w: 9.0, h: 0.6, fill: { color: "DDF3E8" }, line: { color: "DDF3E8" }, rectRadius: 0.08 });
    s.addImage({ data: await icon(fa.FaLock, "157A50"), x: 0.72, y: 4.8, w: 0.3, h: 0.3 });
    txt(s, [
      { text: "Zero-outage by design: ", options: { bold: true, color: "157A50" } },
      { text: "bad configurations are rejected in software before they can touch customer traffic.", options: { color: INK } },
    ], { x: 1.2, y: 4.65, w: 8.1, h: 0.6, fontSize: 11.5, valign: "middle" });

    s.addNotes(
      "Stage 1, candidate datastore staging: the AI decision engine output is translated into YANG configuration and staged in the TeraFlowSDN candidate datastore. Target device: Ceragon MH-T261 (UUID f676623c...). Changes: ACM floor MCS 0 -> MCS 4 (>= 500 Mbps), QoS FIFO -> IEEE 802.1Q PCP class 6 (PfifoFast), carrier 60.48 GHz -> 64.80 GHz.\n" +
      "Stage 2, pre-flight safety gate (3GPP TS 28.561): gate 1 latency (predicted 1.12 ms <= 1.5 ms), gate 2 queue stability (buffer occupancy < 40%), gate 3 hardware capability (MH-T261 supports MCS 4), gate 4 spectral mask (64.80 GHz approved in V-band).\n" +
      "Stage 3, two-phase commit over RESTCONF to 192.168.1.225:80: phase 1 PATCH the candidate datastore, phase 2 POST ietf-netconf:commit, then continuous RFC 8040 telemetry polling with automatic revert if health checks fail."
    );
  }

  // =====================================================================
  // 8. Closed loop
  // =====================================================================
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    title(s, "End-to-end", "One closed loop, seconds instead of hours");

    const cx = 3.1, cy = 3.3, r = 1.4, d = 0.62;
    s.addShape(pres.shapes.OVAL, { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color: WHITE }, line: { color: LINE, width: 2.5 } });
    txt(s, "≈ 2 s", { x: cx - 1, y: cy - 0.4, w: 2, h: 0.5, fontFace: HEAD, fontSize: 26, bold: true, color: NAVY, align: "center", valign: "middle" });
    txt(s, "per loop", { x: cx - 1, y: cy + 0.1, w: 2, h: 0.3, fontSize: 11, color: MUTED, align: "center" });
    const nodes = [
      [fa.FaBullseye, C_INTENT, "1 Intent", "TMF921"],
      [fa.FaFlask, C_SIM, "2 Simulate", "ns-3 what-if"],
      [fa.FaShieldAlt, C_SDN, "3 Verify", "TS 28.561 gates"],
      [fa.FaRocket, C_HW, "4 Actuate", "RESTCONF 2PC"],
      [fa.FaChartLine, NAVY2, "5 Monitor", "live telemetry"],
    ];
    for (let i = 0; i < nodes.length; i++) {
      const [I, c, a, b] = nodes[i];
      const ang = (-90 + i * 72) * Math.PI / 180;
      const nx = cx + r * Math.cos(ang) - d / 2, ny = cy + r * Math.sin(ang) - d / 2;
      await iconCircle(s, pres, I, nx, ny, d, c);
      const right = Math.cos(ang) >= -0.01;
      const lx = right ? nx + d + 0.1 : nx - 1.14;
      txt(s, a, { x: lx, y: ny + 0.04, w: right ? 1.25 : 1.04, h: 0.27, fontFace: HEAD, fontSize: 11, bold: true, color: INK, align: right ? "left" : "right" });
      txt(s, b, { x: lx, y: ny + 0.3, w: right ? 1.25 : 1.04, h: 0.25, fontSize: 9.5, color: MUTED, align: right ? "left" : "right" });
    }

    // where the time goes
    const bx = 5.9, bw = 3.6, total = 2.2;
    txt(s, "Where the ~2 s go", { x: bx, y: 1.35, w: bw, h: 0.3, fontFace: HEAD, fontSize: 12.5, bold: true, color: NAVY });
    const segA = bw * (1.15 / total), segB = bw * ((1.9 - 1.15) / total), segC = bw * (0.3 / total);
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: 1.8, w: segA, h: 0.42, fill: { color: C_SIM }, line: { color: WHITE, width: 1 } });
    s.addShape(pres.shapes.RECTANGLE, { x: bx + segA, y: 1.8, w: segB, h: 0.42, fill: { color: C_SDN }, line: { color: C_SDN, width: 0 } });
    s.addShape(pres.shapes.RECTANGLE, { x: bx + segA + segB, y: 1.8, w: segC, h: 0.42, fill: { color: "7CCAD2" }, line: { color: "7CCAD2", width: 0 } });
    txt(s, "1.15 s", { x: bx, y: 1.8, w: segA, h: 0.42, fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle" });
    txt(s, "+0.3", { x: bx + segA + segB, y: 1.8, w: segC, h: 0.42, fontSize: 9, bold: true, color: WHITE, align: "center", valign: "middle" });
    txt(s, "0.75 s", { x: bx + segA, y: 1.8, w: segB, h: 0.42, fontSize: 10.5, bold: true, color: WHITE, align: "center", valign: "middle" });
    txt(s, "ns-3 what-if (Tier 4)", { x: bx, y: 2.28, w: segA, h: 0.25, fontSize: 9.5, color: MUTED });
    txt(s, "reconcile · verify · commit", { x: bx + segA, y: 2.28, w: segB + segC, h: 0.25, fontSize: 9.5, color: MUTED, align: "right" });

    // hours vs seconds
    const tiles = [["hours", "manual CLI troubleshooting", "9AA7B4"], ["1.9–2.2 s", "autonomous closed loop", C_SDN]];
    for (let i = 0; i < tiles.length; i++) {
      const y = 2.95 + i * 1.2;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y, w: bw, h: 1.05, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
      txt(s, tiles[i][0], { x: bx + 0.2, y: y + 0.08, w: bw - 0.4, h: 0.55, fontFace: HEAD, fontSize: 26, bold: true, color: tiles[i][2], valign: "middle" });
      txt(s, tiles[i][1], { x: bx + 0.2, y: y + 0.65, w: bw - 0.4, h: 0.3, fontSize: 11, color: INK });
    }
    s.addNotes(
      "The loop: SLA intent is ingested (TMF921), simulated in ns-3, safety-verified against the TS 28.561 gates, actuated through TeraFlowSDN over RESTCONF, and monitored through live telemetry. End to end this takes about 1.9 to 2.2 s.\n" +
      "The ns-3 what-if run at Tier 4 is about 1.15 s of that; the remaining 0.75-1.05 s is topology reconciliation, gate checks and the two-phase commit (derived as total minus simulation time).\n" +
      "This replaces hours of manual CLI troubleshooting."
    );
  }

  // =====================================================================
  // 9. Summary
  // =====================================================================
  {
    const s = pres.addSlide();
    s.background = { color: NAVY };
    txt(s, "SUMMARY", { x: 0.5, y: 0.3, w: 9, h: 0.25, fontFace: HEAD, fontSize: 10, bold: true, color: C_SDN, charSpacing: 2 });
    txt(s, "What the adapters deliver", { x: 0.5, y: 0.55, w: 9, h: 0.6, fontFace: HEAD, fontSize: 26, bold: true, color: WHITE, valign: "middle" });

    const vals = [
      [fa.FaShieldAlt, C_SDN, "Zero-risk what-if", "Surges, rain fade and fibre cuts are tested in a µs-accurate twin, never on live circuits."],
      [fa.FaGlobe, C_INTENT, "Multi-vendor, open", "TMF921, 3GPP TS 28.561, ITU-T Y.3090, RESTCONF/YANG: no proprietary lock-in."],
      [fa.FaBolt, C_SIM, "Seconds-scale self-healing", "Intent to verified change in ≈2 s, replacing hours of manual CLI work."],
    ];
    for (let i = 0; i < vals.length; i++) {
      const x = 0.5 + i * 3.1;
      await iconCircle(s, pres, vals[i][0], x, 1.4, 0.55, vals[i][1]);
      txt(s, vals[i][2], { x, y: 2.08, w: 2.8, h: 0.3, fontFace: HEAD, fontSize: 13, bold: true, color: WHITE });
      txt(s, vals[i][3], { x, y: 2.4, w: 2.8, h: 0.75, fontSize: 10.5, color: "C9D6E3" });
    }

    const hdr = { bold: true, color: "8FA3B8", fontSize: 9, fontFace: HEAD, fill: { color: NAVY } };
    const cell = (t, o = {}) => ({ text: t, options: Object.assign({ color: WHITE, fontSize: 10.5, fontFace: BODY, fill: { color: NAVY2 } }, o) });
    s.addTable([
      [{ text: "ADAPTER", options: hdr }, { text: "TRANSLATES", options: hdr }, { text: "STANDARD", options: hdr }],
      [cell("① Intent", { bold: true, color: "B8A8FF" }), cell("Natural language / SLA bounds → TFS service & CSPF requests"), cell("TM Forum TMF921")],
      [cell("② Simulation", { bold: true, color: "F5B96E" }), cell("Live TFS graph → scoped ns-3 run → µs traces"), cell("IRTF NMRG twin drafts")],
      [cell("③ Southbound", { bold: true, color: "6FD3A4" }), cell("Candidate diff → gated two-phase commit on MH-T261"), cell("RFC 8040 / 7950")],
    ], {
      x: 0.5, y: 3.6, w: 9.0, colW: [1.5, 5.3, 2.2], rowH: [0.3, 0.42, 0.42, 0.42],
      border: { type: "solid", pt: 1.5, color: NAVY }, valign: "middle", margin: [0, 0.1, 0, 0.1],
    });
    s.addNotes(
      "Zero-risk what-if validation: operators test severe perturbations (3.5x bulk surges, category-5 tropical rain fade, fibre cuts) in a microsecond-accurate twin without risking live production circuits.\n" +
      "Standardised multi-vendor interoperability: built on open telecom specifications (TMF921, 3GPP TS 28.561, ITU-T Y.3090, IRTF NMRG twin drafts, RFC 8040 RESTCONF) with the open-source ETSI TeraFlowSDN controller.\n" +
      "Seconds-scale closed loop: SLA intent is ingested, simulated, verified and actuated in about 1.9-2.2 s. (The original deck called this 'sub-second'; the measured end-to-end figure is about 2 s.)"
    );
  }

  await pres.writeFile({ fileName: "Telecom_Digital_Twin_Adapters_and_Standards_v2.pptx" });
  console.log("written");
})();
