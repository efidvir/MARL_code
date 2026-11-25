# Bench device catalog, PoC 1 and PoC 2 (TENTATIVE v0)

This is a first pass, to be corrected from lab photos and a tape measure. No repo files were changed.

**Conventions**
- Units are mm, given as W x H x D. For rack devices, W is shown with ears (482.6) and as body width.
- **Front x** is measured from the left edge of the 482.6 mm panel, viewed from the front. **y** is measured up from the bottom of the panel.
- Rear x values use the same front-view frame unless a cell says "seen from rear" (x_rear = 482.6 - x).
- Confidence tags:
  - **[C]** confirmed: a datasheet or drawing was read, or it is visible in a lab photo.
  - **[L]** likely: from a search excerpt, a sibling model, or several third-party listings.
  - **[E]** estimate: the reasoning is given. Positions scaled from lab photos are [E], +/-8 mm in x and +/-5 mm in y.
- Most vendor sites could not be reached from this container, so their facts come from search excerpts. Blocked or unreachable: minicircuits, keysight, ceragon, eaton, se/apc, netgear downloads, startech, meanwell, pasternack, fairview, octopart, digikey, raritan CDN and others. The only vendor PDF read in full is the Westell technical practice (TP).
- Lab photos are in `(session scratch) photos/`.

## 0. Conflicts to settle first
1. **Attenuators.**
   - bench_spec (user note, 2026-11-24) describes them as "small bench boxes with a dial, 4 ch, two per 2U shelf". The twin draws them as 200 x 70 x 160 [E].
   - The model that was researched, ZTDAT-4-18G63S, is a **1U 19-inch rack unit**: 482.6 x 44.45 x 330.2 [L], 3.77 kg [L], with no dial.
   - So either the lab model is something else, or each unit takes 1U in the rack directly: 10U in PoC 2 (PoC 1 has no attenuators since 2026-11-25), instead of 10U of shelves.
2. **IP-50.** bench_spec says IP-50CX (band TBC). No IP-50CX data was found.
   - The 13/15 GHz IP-50C fits the WR-62/SMA chain.
   - The E-band IP-50E and IP-50EX do not fit it; they would need a WR-12 chain.
3. **Spider boxes.**
   - There are three different front panels.
   - bench_spec.mod_x assumes a module pitch of 95 and a tile pitch of 18. The photos show a module pitch of ~78 and a tile pitch of ~19.6-21, so the module centres are off by 40-95 mm.
   - One box has a single DC terminal-block inlet at the rear [L]. If that holds, the PA-01..10 adaptors and the ETH1 PoE feeds may not be needed.
4. **Node board.** The silkscreen reads **N360** (PCB220C_V1); bench_spec says N367.
5. **Twin catalog values contradicted by research.** See the notes at the end.

## A. Spider boxes and their internals (rack, 1U each)

| Device | Part | Form factor | W x H x D (mm) | Mounting | Front | Rear | Confidence |
|---|---|---|---|---|---|---|---|
| SB1 or SB4 ("GTS003 #1/#2"): unit SF06000007, asset sticker '54'. PoC 1: W1+W2 or E1+E2. PoC 2: A1+A3 or B1+B3 | Siklu 1U "Spider" conducted TG test box, lab label GTS003 (no public record). Front **variant A** (wide window) | 19-inch 1U, ears part of the front panel [C photo] | Panel 482.6 x 43.6 [L EIA]. Body ~438 W [E +/-8; scaled from the Pi 4 board, 85 mm] x ~42 H [E] x ~331 D [E, range 322-340; panel face to rear wall]. Add ~3 mm panel [E] and ~10 mm SMA protrusion [E], giving **~345 overall** [E]. Weight 4-5 kg [E] | Front ears only, 2 oblong slots per ear, slot centres 465.1 apart [L EIA]. No rear brackets seen [C]. Rear support (shelf or rear rails) recommended | Brushed natural aluminium [C]. 10 slanted vent slots at x 33-123 (each ~4 x 27, tilted ~15 deg) [E]. 2 countersunk screws per side at x ~30 / ~452 [E]. Window x 128-446, y 14-26 [E]. **16 x SMA-F** edge-launch, gold, 4 per FATJ0498 module [C count and gender]; x positions in A2, axis y ~19 [E]. White labels: 'GTS003' at x 210-290, 'SF06000007' at x 297-375 [E]. No LEDs, switches or display [C]. Jacks are PCB-mounted: hold the box when torquing to 8 in-lb | Seen from inside only (lid-off photo); positions in A3. Galvanised steel wall with one cut-out per node [L]. Per node: SFP+ cage, 2 x RJ-45 (ETH1 10G PoE-in and ETH2 1G, order unknown), a USB-A-like part with 2 red jumpers. Per box: 1 green pluggable terminal block (~4 positions, 2 wired; DC in [L]), 2 x SMA-F bulkheads (GNSS in [L]), 1 x RJ-45 feed-through for the Pi LAN [L] | Front: counts and gender [C], positions [E]. Rear set [L]. Depth [E] |
| SB1 or SB4: unit SF06000031 | GTS003, front **variant C** (narrow slot) | 19-inch 1U | Chassis assumed the same as SF06000007: 482.6 x 43.6 panel, ~438 x ~42 x ~331 body [E] | As above [C photo] | Brushed aluminium. 10 vents at x 34-118 [E]. Screws at x ~28 / ~455 [E]. Slot x 128-444, y 20-34 [E]. **16 x SMA-F, even pitch ~19.6**, axis **y ~27** [E +/-5]; x positions in A2. 'GTS003' above the slot at x 237-292; 'SF06000031' below it at x 140-200 [E]. No LEDs [C] | Not seen. Assumed the same as SF06000007 [E] | Front counts [C]. Everything else [E] |
| SB2, GTST002B: unit SF06000001. PoC 1 HUB, PoC 2 H | Siklu Spider, 1 node x 4 sectors per bench_spec. Front **variant B** (tall window, no vents) | 19-inch 1U | Panel 482.6 x 43.6 [L]. Body assumed ~438 x ~42 x ~331 [E] | As above [C photo] | Brushed aluminium, **no vent slots** [C]. Window x 61-433, ~30 tall [E]. 'P/N FATJ0498 Rev A01' labels readable through it [C]. **16 x SMA-F in 4 groups of 4**: ~21 mm within a group, ~24 mm between groups, axis y ~20 [E]; x positions in A2. Screws at x ~30 / ~452. 'GTST002B' below the window at x 236-284; 'SF06000001' at x 438-471 [E]. No LEDs [C] | Not seen. If it has 1 node: 1 node group (SFP+, 2 x RJ-45, USB), 1 GNSS SMA, DC terminal block and the Pi RJ-45 [E] | Front counts [C]. Rear and the 1 x 4 split [E] |
| SB3, GTST003: unit SF06000058. PoC 1 S+N, PoC 2 A2+B2 | Siklu Spider, 2 nodes x 2 sectors (assumed from the README). Front **variant C** | 19-inch 1U | Assumed ~438 x ~42 x ~331, panel 482.6 x 43.6 [E] | As above | As SF06000031: vents at x 34-118, slot x 130-446 / y 20-34, 16 x SMA-F at even pitch 19.1-19.6 from x ~141 to ~435, axis y ~27 [E]. 'SF06000058' at x 135-174 and 'GTST003' at x 244-299, both below the slot [E] | Not seen. Assumed as SF06000007 [E] | Front counts [C]. Split, rear and depth [E] |
| Kit item 1, "6*4 sector nodes - GTSTU022\23" (not placed in PoC 1 or 2) | GTSTU022 / GTSTU023 lab labels; no web hits | Unknown. Placeholder: Spider 1U [E] | Placeholder copied from the Spider chassis [E] | Placeholder: front ears | Unknown. "6*4" could mean six 4-sector nodes, or 24 SMA. The Siklu Spider document describes a 1RU module holding 2 TUs with 4 RF connectors each [L search excerpt] | Unknown | [E] |
| Node boards inside the boxes. PoC 1: W1, W2, HUB, S, N, E1, E2. PoC 2: A1, A3, H, A2, B2, B1, B3 | Siklu MultiHaul TG board, silkscreen **N360** on PCB220C_V1 [C photo]. bench_spec says N367 | Internal board, rear compartment | ~150 W x ~185 D [E]. For reference only, the outdoor N367 is 195 x 242 x 184 mm, 3.6 kg [C DS] | Standoffs; ports at the box rear wall [C photo] | n/a | SFP+ (1/10G, 3.3 V class-1 modules only) [L]. ETH1 10/5/2.5/1G PoE-in and ETH2 1G PoE-out [L, TG datasheet]. Power 48 W without PoE-out, 90 W with it [L]. Default IP 192.168.0.1 [L] | Connector types [C]. Functions [L]. SKU to confirm |
| Sector modules M1-M4 in each box (tile ports T1-T4; 16 per box) | Siklu FATJ0498 Rev A01 (label read in photos [C]; no public record) | Internal PCB behind the front panel | ~75 W x ~50 D [E]. Module pitch ~78 [E]. ~30 g [E] | 3 screws onto bent brackets on the tray floor [C photo] | 4 x SMA-F edge-launch, gold, through the panel window [C]. Carries IF, assumed ~13-17.4 GHz; DC, LO or control on these ports is unconfirmed | Grey IDC ribbon [C]; micro-coax IF feed [L] | Label and connectors [C]. Function [L]. Size [E] |
| Box controller (seen in SF06000007; assumed in the others) | Raspberry Pi 4 Model B [C silkscreen] | Internal SBC | 85 x 56 x ~17 [C standard] | Front-left of the tray: x 53-133, z 47-106 from the panel [E] | n/a | Ethernet by a white patch cord to the rear RJ-45 feed-through at x ~69 [L]. 2 USB cables into the box. Red/black power leads on GPIO [C photo] | Identity [C]. Role unknown |

**A2. Front SMA x positions per variant** (mm from the left panel edge, [E] +/-8)

| Variant | Units | M1 (SMA 1-4) | M2 (5-8) | M3 (9-12) | M4 (13-16) | SMA axis y | Module centres |
|---|---|---|---|---|---|---|---|
| A | GTS003 SF06000007 | 141, 162, 184, 205 | 224, 244, 264, 284 | 303, 322, 342, 362 | 378, 398, 416, 435 | ~19 +/-4 | ~173 / 254 / 333 / 407 |
| C | GTS003 SF06000031 | 140, 162, 182, 203 | 222, 242, 263, 283 | 302, 321, 341, 360 | 378, 396, 415, 434 | ~27 +/-5 | ~172 / 253 / 331 / 406 |
| C | GTST003 SF06000058 | as the row above, even pitch 19.1-19.6, from ~141 to ~435 | | | | ~27 +/-5 | ~as above |
| B | GTST002B SF06000001 | 95, 117, 138, 159 | 183, 204, 224, 244 | 265, 285, 305, 325 | 346, 365, 384, 403 | ~20 | ~127 / 214 / 295 / 375 |

M1-M4 and SMA 1-16 are assumed to run left to right. For comparison, bench_spec.mod_x places the module centres at ~79 / 174 / 269 / 364 from the panel edge (it measures from the rail line, 8.75 mm in), with a tile pitch of 18. Both should be replaced.

**A3. Rear wall of GTS003 SF06000007** (taken from the inside photo and mirrored). x is in the front-view frame; seen from the rear, x_rear = 482.6 - x. All x [E] +/-10, y [E] +/-5.

| x | y | Item | Type | Conf. |
|---|---|---|---|---|
| 47 | 29 | Bulkhead #1, likely GNSS in for node A | SMA-F, gold, micro-coax pigtail inside | [L] |
| 69 | - | Box controller LAN | RJ-45 shielded coupler (feed-through) | [L] |
| 119 | - | Node A service | vertical USB-A-like part between 2 red jumpers | [E] |
| 150 | 19 | Node A data | SFP+ cage | [C] type |
| 186, 223 | 22 | Node A ETH1 / ETH2 (order unknown) | 2 x RJ-45 | [C] type |
| 245 | 29 | Bulkhead #2, likely GNSS in for node B | SMA-F | [L] |
| 267 | 33 | Box DC input | green pluggable terminal block, ~4 positions, 2 wired | [L] |
| 328 / 365 / 400, 434 | ~19-22 | Node B: USB/jumpers, SFP+, 2 x RJ-45 | as node A | [E] |

**Other Spider facts**
- Inside, a transverse steel wall at z ~130-140 carries 3 fans of ~40 mm at x ~91, 237 and 380 [E]. Airflow direction is unknown; front-to-rear is assumed [E].
- Other boards seen: baseband daughterboards PCB241A_V2 (P/N FAM00068 Rev A102); PCB249B_V1 'P/N SGPS002', which looks like a GPS board; PCB305_V1 boards with u.FL connectors [C photo].
- Finish: brushed silver front, spangled galvanised steel body and lid, gold SMA, white labels with black text [C photo].
- Cable entry: front carries 16 SMA IF cables. Rear carries DC, node RJ-45, SFP+ fibre or DAC, GNSS coax if used, and the Pi LAN.

## B. IF attenuators and small RF parts

| Device | Part | Form factor | W x H x D (mm) | Mounting | Front | Rear | Confidence |
|---|---|---|---|---|---|---|---|
| ATT-01..10 (PoC 2 only; PoC 1 has fixed pads) if the lab uses ZTDAT | Mini-Circuits ZTDAT-4-18G63S: 4 ch, 0.1-18 GHz, 0-63 dB in 0.25 dB steps, case VH3223 | **19-inch rack 1U** [C excerpt] | 482.6 x 44.45 x 330.2 (19 in x 1U x 13 in) [L, siblings 13G60S/13G90S, case drawing 99-01-3223]. Body ~430-440 W [E]. Handles ~35-45 mm and SMA ~9 mm proud of the panel at the front [E]; rear plugs ~10-15 mm [E]. **3.77 kg** [L] | Front ears, 2 holes per ear [E]. Rear support advisable in a 4-post rack [E] | Power switch with LED and cover, left end, x ~50-70, y ~22 [L feature / E position]. **8 x SMA-F**, IN and OUT for ch 1-4, A/B labels [C SMA-F on the front; L labels; which letter is IN unknown], in one row at y ~22: pair centres ~80 apart, IN-OUT ~20-25 apart [E]. Carry handles at both ends [L]. Logo at the right [E] | IEC C14 inlet with a 2 A 250 V fuse, 90-260 V [L siblings]. USB-B [L]. RJ-45 10Base-T (HTTP/Telnet/SSH) [C interface, L rear]. Positions [E]: inlet at one end, USB and RJ-45 together, all at y ~22 | Form and front/rear split [C]. Size, weight and inlet [L]. Positions [E]. Colour unknown (draw grey/black with white legends [E]). Airflow unknown |
| ATT-xx as the user described them (PoC 2 only; PoC 1 has fixed pads) ("small 4-ch bench box with a dial", 2 per 2U shelf) | **Model unknown** | Table-top box on a 1U vented shelf (2U allowed) | Twin placeholder 200 x 70 x 160 [E]. No source | Loose on the shelf | SMA on the front assumed [E] | Unknown | [E]. The model is needed |
| Possible spare or single channel | Mini-Circuits RCDAT-18G-63: 1 ch, 0.1-18 GHz, 0-63 dB, case QF3189 | Small table-top module | ~90 x 25 x 65 [E, from the connector set and 0.25 kg]. 0.25 kg [L] | Shelf or table | RF IN and OUT: 2 x **2.92 mm female** (mate SMA) [C] | USB-C, RJ-45, 15-pin D-sub male (RS232, TTL, DC in) [C]. Faces [E] | Connectors and nickel-plated Al finish [C]. Size [E] |
| SPL-HUBW / SPL-HUBE t1..t4 (PoC 1 only, 8 units on SPL-TRAY) | Mini-Circuits ZN2PD-02183-S+: 2-way 0 deg, 2-18 GHz, DC pass, case UU2386 | Inline block on a tray | 51.9 L x 25.4 W x 10.4 H [L DigiKey]. SMA ~8-9 proud [E]. 95 g [L] | Flat on the tray; screws or Velcro [E] | 3 x SMA-F: S, 1, 2 [C]. S at the centre of one long edge; 1 and 2 at 1/4 and 3/4 of the opposite edge [E] | n/a | Ports [C]. Size [L]. Edges [E]. Blue paint, Pantone 286 [L] |
| CPL-AS, -BE, -DW, -EN t1..t4 (PoC 2, 16 units on CPL-TRAY) | KRYTAR 1821: 10 dB coupler, 1-18 GHz, SMA-F | Inline coax part | ~90-110 L x 13-15 W x 18-20 H [E; sibling 180120 is 88.9 x 13.5 x 18.5]. SMA ~9 each [E]. ~74 g [L] | Tray; tapped base holes [E] | 3 x SMA-F: IN and OUT on the end faces (axial), CPL on the top or side near OUT [C ports / E faces]. Isolated port terminated inside [C] | n/a | Ports [C]. Size [E]. Silver [E] |
| CPL alternative | Mini-Circuits ZUDC10-183+: 10 dB, 0.5-18 GHz, case HT1967 | Inline bar | **152.4 L** (164.3 with connectors) x 18.5 x 12.7 [L]. #4-40 tapped holes, 120 g [L] | Screwed to the tray | SMA-F IN, OUT and CPL, plus probably a 4th port with the supplied 50-ohm termination [L] | n/a | [L]. Plan the tray for 6-inch parts |
| On each coupling arm (PoC 2) | Mini-Circuits BLK-18-S+ DC block, case FF888 | Inline barrel | 29.97 L x 10.41 dia, 7.92 hex [C]. 7.0 g [C] | Hangs on the port | SMA-F to SMA-M [C] | n/a | [C] |
| On each coupling arm (PoC 2); PoC 1: every IF pad stack (PAD-<link>.t<n>) and the 10 dB pad of each IP-50 hop | Mini-Circuits BW-S10W2+ pad (BW-S<n>W2+ family), case FF658 | Inline barrel | 21.59 L x 9.14 dia, 7.92 hex [L]. 4.3 g [L]. **Conflict:** the twin (catalog.PAD) draws each pad 31 mm long and 10 mm across [E], so a two-pad stack is 62 mm there, about 43 mm with these sizes; the P1 / P2 stacks on a splitter would then have about 2.9 mm between bodies instead of 2 mm. The twin keeps the task's estimate until the lab measures a pad | On the port | SMA-M to SMA-F [C] | n/a | [L]. Stainless [C] |
| On unused SMA-F ports | Mini-Circuits ANNE-50+ 50-ohm load, case LL561 | Cap | ~14.7 L x 9.4 dia [L]. ~4 g [L] | On the port | SMA-M [C] | n/a | [L] |
| All IF cables | Mini-Circuits CBL-1FT/2FT/3FT/1M/2M-SMSM+ | Cable | Lengths 305 / 610 / 914 / 1000 / 2000 [C]. OD ~4.5-5.0 [E]. Bend radius ~25 static, ~50 dynamic [E]. Connector body and boot ~25-30 per end [E] | Velcro in the cable managers | SMA-M both ends, passivated stainless [C] | n/a | Blue jacket [L]. bench_spec BEND_MM = 100 is conservative and fine |
| Optional short jumpers | Mini-Circuits 086-2/4/7SM+ Hand-Flex | Formable jumper | 50.8 / 101.6 / 177.8 L, OD 2.18, bend radius 6 [C] | Self-supporting | SMA-M both ends, anti-torque nut [C] | n/a | [C] |

## C. IP-50 chain and bench-table instruments

| Device | Part | Form factor | W x H x D (mm) | Mounting | Front / interface face | Rear / other face | Confidence |
|---|---|---|---|---|---|---|---|
| IP50-<hop>@<site> (8 on hand per parts list; bench_spec choice) | **Ceragon IP-50CX** (band TBC). bench_spec: "compact dual-carrier, 1/10/25 GbE"; source not traced | ODU on the table | **Not found.** Twin uses 270 x 110 x 270 [E]. The IP-20C/IP-50E family envelope is 322 x 227/270 x 86, ~6 kg [L for those models] | Table, laid flat or on a jig | Unknown | Unknown waveguide interface | [E] all |
| IP-50 if it is a 13/15 GHz IP-50C | Ceragon IP-50C (6-42 GHz) | ODU | ~322 x 227 (270 with latches) x 86, ~6 kg [E from IP-20C/PTP 850C] | Table | Interface (bottom) face: DC -48 V (-40.5 to -60), likely a Ceragon 2-pin plug; RJ-45 1/10G PoE-in; 10G SFP+; 1/2.5G SFP; 10G extension SFP+ [L brochure]; ground stud and RSL test point [E]. Positions [E] | Antenna face: Ceragon direct-mount interface. A remote-mount kit gives a standard flange: 15 GHz **WR-62 PBR140**, 13 GHz WR-75 PBR120 [L IP-20C table] | Ports [L]. Size [E]. Light grey [E]. Passive cooling |
| IP-50 if E-band | Ceragon IP-50E (71-86 GHz) | ODU | 322 x 227/270 x 86, 5.5 kg (direct mount) [L] | Table | DC -48 V; RJ-45 1G mgmt/PoE; SFP 1/2.5G; QSFP+ 4x10/40G; 58 W [L] | WR-12-class guide [E]. **Not compatible with the WR-62 chain** | [L] |
| IP-50 if E-band compact | Ceragon IP-50EX (EXP/EXA same size) | ODU | 228 x 233 x 76, 2.77 kg [L] | Table | DC; 1 x 1G RJ-45; 2 x SFP28; QSFP [L] | WR-12-class guide [E]; not compatible | [L] |
| WGT-<hop>-A/B (one per radio port) | Pasternack PEWCA1003: WR-62 UG-419/U to SMA-F, right angle, 12.4-18 GHz | Inline adapter on the radio flange | Flange 33.3 x 33.3 square, 4 holes [L]. Aperture 15.80 x 7.90 [C]. Body ~18 x 10 x 30-35 long [E]. ~78 g [L, Fairview equivalent] | Bolted to the radio or kit flange, 4 screws | Flange face plus side SMA-F [C] | Closed back-short | UG-419/U does **not** mate with PBR140. Use an adapter, or PEWCA1098 (UBR140 flange, SMA-F) [L] |
| Pad on each WGT (2 per hop) | Fairview SA18S10W-30: 30 dB, 10 W, DC-18 GHz | Inline pad | 47.2 L x 15.75 dia [L]. 23.4 g [L]. **Conflict:** the twin (catalog.PAD) draws it 60 L x 25 dia [E]; it keeps that estimate until the lab measures a pad | Screwed onto the WGT SMA | SMA-M in [C] | SMA-F out [C] | [L]. Finned body [E] |
| MWATT (PoC 2 only; PoC 1 uses a fixed BW-S10W2+ 10 dB pad per hop) | Keysight J7204B Opt 002: 4 ch, DC-18 GHz, 0-121 dB | Full-width 2U instrument | **425.6 W x 88.3 H x 578.1 D** without connectors [C datasheet]. 482.6 W with a flange kit (kit part number not found). Add ~10 front (SMA) and 60-80 rear (plugs) [E]. **11 kg** [C] | Table on feet [E]. In a rack it needs flanges plus rear rails | 8 x SMA-F (ch 1-4, Port 1 and Port 2) [L]; one row at y ~44, over the right two-thirds [E]. Standby button (not a line switch) at the lower left [C / E pos]. PWR and per-channel state LEDs [C exist]. No LCD [L] | LAN RJ-45 [C], GPIB [C], AC inlet probably C14 [E], fan grille [E] | Size and weight [C]. Positions [E]. Grey/black [E] |
| SPECAN | R&S FPL1026 (1304.0004K26), 5 kHz-26.5 GHz | Benchtop | **408 W x 186 H x 235 D** [C]. +20 front, +80 rear [E]. 6 kg (7.3 kg with battery) [C] | Table, tilt feet [E] | 10.1 in touch display on the left ~60% (x ~15-250, y ~50-180) [C display / E pos]. Keys and knob on the right [L]. 2 x USB-A at the lower left [C count]. RF IN at the lower right, x ~370, y ~25: test-port adapter, **2.92 mm F standard** (N-F optional) [C] | AC inlet with switch and fuse [C], DC in [C], battery slots [C], LAN [C], 2 x USB 3.0 [C], DVI [C], GPIB option [C], REF IN/OUT BNC [C], TRIG IN BNC [C]. Positions [E] | [C] size and ports. Dark grey front, light grey case [E] |
| PSENS (suggested ID; parts list only) | R&S NRP18S (USB) or NRP18SN (LAN) | Hand-held sensor | 48 x 30 x 138 [L]. <0.2 kg [C] | On the table or hanging on the DUT | N-male [C]; needs NF-SF50+ to take SMA-M | Host socket for the NRP-ZKU USB cable (M12 at the sensor end) [L] | [L] |

## D. Power

| Device | Part | Form factor | W x H x D (mm) | Mounting | Front | Rear | Confidence |
|---|---|---|---|---|---|---|---|
| PDU-A..C (current bench_spec: PoC 1 needs 3, PoC 2 needs 2) | Raritan PX3-5260R: 1U, 230 V 16 A, 12 x C13, switched | 19-inch 1U [C] | 440 x 44 x **183** body [C two listings]. 482.6 with ears [E]. 5.55 kg [L] (one listing says 2.85) | Front ears [E] | 12 x C13 [C count / E face]. Controller with LCD, RJ-45, RS-232, 2 x USB-A, USB-B [L], at one end [E] | IEC 60309 16 A 2P+E cord [C parts.txt], exit [E] | [L]. Black [E] |
| PDU alternative (32 A) | APC AP7922B: 2U, 32 A, 16 x C13 | 19-inch 2U [C] | 445 x 89 x 114 body [C]. 482.6 with ears [E]. 4.49 kg [C] | Front ears; brackets let it face front or rear [L] | 16 x C13 in 2 rows of 8: rows at y ~25/64, pitch ~36, x ~110-395 [E]. LED per outlet [L]. Load display, RJ-45, RJ-12 serial [C AP7922], reset; mgmt module at one end, 2 bank breakers at the other [E] | IEC 60309 32 A 2P+E blue plug, 3.05 m cord [C]. Exit on the face opposite the outlets [E] | [L]. Black [C] |
| DC-RECT (48 V, 2 modules), DC-RECT2 (-48 V for the IP-50, 1 module, +V grounded) | Mean Well RCP-1UI shelf | 19-inch 1U, 3 hot-swap bays [C] | **483.6 (with ears) x 44 x 350.8** [C spec]. Body ~440 [E]. Shelf 2.5-3.5 kg empty [E] | Front ears, M5 screws [C]. Rear support advisable [E] | 3 open bays. Seen from the front: **C left, B middle, A right** [C manual]. Bays at x ~51-178 / 178-305 / 305-432, ~41 tall [E]. DC-RECT uses A+B; DC-RECT2 uses A | **3 x IEC C14**, one per bay [C]. DC +V/-V bus bars [L; position and bolt size unknown]. CN500 D-sub 25 F: remote on/off, sense, alarms [C]; strap +S/-S to +V/-V if unused [L]. Address DIP switch [C, location unknown] | [L]. Airflow front to rear [L]. Silver steel [E] |
| Modules in DC-RECT/RECT2 | Mean Well RCP-1000-48: 48 V, 21 A | Plug-in module | 127 W x 41 H x 295 D [L]. 1.5-2 kg [E] | Bay, blind-mate backplane [L] | Fan grille, AC OK and DC OK LEDs, handle [L] | Backplane connector [L] | [L] |
| DC-DIST (CB1..10 to PA-01..10) | Eaton DCD1M, 3G (suffix -00 or -01 TBC) | 19-inch 1U [C] | 482.6 x ~43.6 x **280** [L brochure]. Body ~430-440 [E]. 4.1 / 4.4 kg [C] | Front ears [E]. Rear support [E] | 12 push-in breakers: 2 battery (AC1R) + 10 load [C]. Battery at the left, CB1-10 at ~19 mm pitch around the centre (x ~130-350) [E] | **Conflict:** the 2009 guide puts M6 input studs, 2 x 10-way barrier strips (live and common), M5 ground and PP45 battery connectors at the rear [L], but the 3G brochure says "full front access" | Front [C]. Terminal face to confirm |
| DC-DIST2 (-48 V, F1..F10 to the IP-50 radios) | Westell N250110-N-0803: 2 buses x 5 GMT positions | 19- or 23-inch 1U [C TP] | **431.8 x 43.7 x 147.8** body [C TP]. 482.6 with brackets [C]. 3.6 kg [C] | 3 reversible brackets, flush or 127 mm offset; #12-24 screws [C] | **All terminations on the front** [C]. x from the left ear (+/-5) [E]: bus A input studs 32-75; bus A output strip 82-150 (RTN upper y ~30, BAT lower y ~13); bus A LEDs ~157; 10 GMT holders 172-237 (A1-5 then B1-5); bus B LEDs and EXT ALARM 247-262; alarm wire-wrap pins 267-308; ground stud 308-327; bus B strip 334-400; bus B input studs 409-454. Clear L-shaped shields [C] | None [C]. Vents on top [L] | [C] (only vendor PDF read). Off-white paint [C] |
| PA-01..PA-10 on PA-SHELF | Siklu EH-PoE-DC-Adaptor (bag of 10) | Small inline adapter | **Not found.** ~45-60 x 20-25 x 18-25 [E]. Twin uses 30 x 25 x 95 [E] | Loose on the shelf | 2-position screw terminal (+/-) [L]. RJ-45 to ETH1, power only [C function]; jack or plug unknown | n/a | [E]. Possibly redundant: see A3 |
| INJ-<node> (e.g. INJ-HUB, PoC 1) | Siklu AX-IN-10G-60W-AC-PoE-xx | Desktop brick | **138.5 L x 62.5 W x 35.3 H** [C]. 0.43 kg [L] | Shelf or table | C6 inlet [L] (fits P014 C14-C5); 2 x RJ-45 (data in, data+power out) [C]; 55 V output [C]. Faces [E] | as front | [L] |
| PSU-<hop>@<site> (one per IP-50 radio off at t0) | Generic 48 V 120 W brick, e.g. Mean Well GST120A48 | Desktop brick | **167 x 67 x 35** [C example]. ~0.6 kg [E] | On the table, flat | C14 inlet [C]. DC cable ~1.2 m [C]; cut and fit the IP-50 DC plug | n/a | [E] part / [C] example dims |
| TABLE-STRIP | Lab strip, part TBC. Typical: 1U IEC strip with a C14 inlet and 8 x C13 | Table-top strip | ~485 x 44 x 44 [C for Secomp/Adam Hall examples]. Twin uses 560 x 45 x 60 | Table edge | 8-10 x C13, pitch ~45-50 [E] | C14 inlet at one end [C example]. 10 A limit | [E] |
| AC leads | Eaton Tripp Lite P004-006/010/015 (C14 to C13) and P014-006/010 (C14 to C5) | Cords | 1830 / 3050 / 4570 [C]. 18 AWG, OD ~7 [E]. C13 body ~26 x 18.5 x 52 (~79 with strain relief) [E] | n/a | n/a | n/a | Type and length [C]. Black [C] |

## E. Network (rack)

| Device | Part | Form factor | W x H x D (mm) | Mounting | Front | Rear | Confidence |
|---|---|---|---|---|---|---|---|
| SW-DATA | Netgear M4350-24F4V (XSM4328FV) | 19-inch 1U | **440 x 43.2 x 400** body [C]. 482.6 with ears [E]. 6.25 kg [C]. ~100 rear clearance [E] | Front ears. Optional RBS1830 rear support (fits 18-30 in rails) [C] | **24 x SFP+ (1-24) and 4 x SFP28 (25-28)** [C]. 2 LEDs per port [C]. POWER, FAN and STACK LEDs [C]. USB-C console and recessed Reset [C]. USB-A [L]. Layout [E]: mgmt cluster x ~60-115; SFP+ in 2 rows x 12 at x ~130-330, pitch ~16.5, odd ports top (y ~29) and even bottom (y ~13) [L convention]; SFP28 2 x 2 at x ~355-395 | OOB RJ-45, C14 with on/off switch, 1 APS PSU bay with blank, 2-3 fans [C]. Order [E]: APS bay 0.05-0.30 of the width, C14 0.35-0.45, fans 0.5-0.85, OOB ~0.9 | Airflow front to back [L]. Dark grey [E] |
| SW-OOB | Netgear M4250-26G4XF-PoE+ (GSM4230PX) | 19-inch 1U, AV line | **440 x 43.2 x 400** body [C]. Twin uses 300: wrong. 5.45 kg [C] | Standard and recessed bracket sets [C]. Can be mounted either way round [L] | Netgear "front": LED mirror of all ports and system LEDs [C]; USB-A [L]. **No data ports on this face** | Netgear "back": **all ports** [C]. 24 x RJ-45 PoE+, 2 x RJ-45, 4 x SFP+, RJ-45 console, USB-C console [L rear], OOB RJ-45, C14 with switch [C]. Layout as fractions of the width, facing the ports [E]: console/OOB 0.06-0.15; RJ-45 1-24 at 0.18-0.62; 25-26 at ~0.65; SFP+ 27-30 at 0.70-0.78; C14 at 0.90-0.95 | Airflow side to side (left to right) [L]. Port numbering [L] |
| Copper SFPs in SW-DATA P1-16 (and SW-OOB if needed) | Netgear AGM734-10000S, 1000BASE-T | SFP module | In-cage 13.7 x 8.5 [L MSA]. ~65-75 long, ~20-30 proud [E] | Cage | RJ-45 [C] | n/a | Allow ~100 mm in front [E] |
| SR optics | Netgear AXM761-10000S | SFP+ module | 55.7 x 13.7 x 11.9 [C]. 17.1 g [C] | Cage | LC duplex [C] | n/a | [C] |
| DACs | Netgear AXC763 (3 m) / AXC761 (1 m) | Cable | 3000 / 1000 [C]. OD ~4.5-5.5, bend ~30-40 [E] | Cage to cage | SFP+ plugs [C] | n/a | [C] / [E] |

## F. Rack, shelves and bench table

| Device | Part | Form factor | W x H x D (mm) | Mounting | Front | Rear | Confidence |
|---|---|---|---|---|---|---|---|
| RACK (bench_spec picks the 42U above 25U) | StarTech 4POSTRACK42 (or 4POSTRACK25U) | 4-post open frame, EIA-310 | 600 W x 2045 H (castors; ~1971-1981 without) x 662 D [C manual]; up to 1048 D at maximum. **Rail depth adjustable 559-1016** [C] (twin: 750). 42U = 1866.9 of rail. U1 bottom ~50 above the floor on feet, ~120-130 on castors [E]; twin uses 100. 38.5 kg [C] | Castors or levelling feet; floor-bolt holes [C] | **Square-hole** L-rails, M6 cage nuts [C]. U markings [C]; U1 at the bottom [L] | Open | Black [C]. The 25U has the same rail-depth range [L] |
| Shelves: AGENTS, PA-SHELF, SPL/CPL-TRAY, attenuator shelves (PoC 2 only) | StarTech CABSHELFV1U (10 in). Variants: 12 in (315) and 16 in | 1U cantilever shelf | 482.6 x ~44 x **254** behind the face [C]; tray ~440 wide [L]. Twin uses 450/300: wrong. 20 kg load [C]; shelf mass ~2-3 kg [E] | Front rails only, 4 x M6 [C] | Plain lip | Open tray end | Black [C] |
| CM | Panduit NCMHF1 | 1U horizontal manager, front only | **482.6 x 43.7 x 95.3** [C] | Front rails | Fingers plus hinged cover [C] | Flat plate | Black [C] |
| Blanks | APC AR8136BLK, toolless | 1U panel | 483 x 45 x **28** with clips [L]. Twin uses 15 | Square-hole rails only [C]; fit this rack | Plain | Clips | Black [C] |
| BENCH-TABLE | Lab table, unknown | Table | Twin: 3.6 m x 0.8 m, top at 750, to the right of the rack [E] | Floor | n/a | n/a | [E] all |

## G. Timing (only if the Spider boxes take GNSS)

| Device | Part | Form factor | W x H x D (mm) | Mounting | Front | Rear | Confidence |
|---|---|---|---|---|---|---|---|
| GNSS-SPL | GPS Networking RMALDCBS1X8 (connector type by order) | 19-inch 1U | 431.8 x 44.5 x **215.9** body, without connectors [C]. 482.6 flange [L]. **5.4 kg** [C] | Front ears [E] | Power LED [C] | 1 in + 8 out (J1 passes DC; J2-J8 DC-blocked with 200-ohm loads) [C]. Face assumed rear [E]. AC inlet type unknown [E] | Connector face [E] |
| GNSS-ANT (roof, off-bench) | Protempis/Trimble Bullet 360, 101898-00 | Mast antenna | 77.5 dia x 66.2 [C]. 200 g [C] | 1"-14 UNS or 3/4" pipe thread [C] | n/a | TNC-F in the base [C] | White [E] |

## H. Compute and operator station (actual lab models unknown; stand-ins used)

| Device | Part (stand-in) | Form factor | W x H x D (mm) | Mounting | Front | Rear | Confidence |
|---|---|---|---|---|---|---|---|
| AGENT-* (PoC 1, 6 units on the AGENTS shelf) | Lab mini PC; stand-in ASUS NUC 14 Pro | Mini PC | 117 x 54 x 112 (tall) or 117 x 37 x 112 (slim) [L stand-in]. Twin uses 120 x 40 x 120 | Shelf, 3 across x 2 deep [E] | Power button, USB [L] | 2.5GbE RJ-45, USB-C, HDMI, DC in [L]. 2nd NIC by USB adapter [E]. Brick with C6 inlet [E] | [E] |
| HOST-* (1U, 10G SFP+ NIC) | Lab server; stand-in Dell R260 (alt. Minisforum MS-01) | 1U rack server | 482 x 42.8 x 461 (497 with bezel) [C R260]. 9.6 kg [C] | 4-post rails | Drives, power button, LEDs [L] | PCIe 2 x SFP+, 2 x 1G LOM, iDRAC RJ-45, PSU C14 at the right [E] | [E] model |
| SRV-E2E (1U), SRV-DPIR (2U, GPU) | Lab servers; stand-ins Dell R660 / R760 | 1U / 2U rack servers | R660: 482 x 42.8 x **809** (823 with bezel), 22.5 kg [C]. R760: 482 x 86.8 x 758 (772), 36.1 kg [L]. Need rail depth ~850-900 [E] | 4-post sliding rails | Drives, ear controls [L] | iDRAC, NICs, 2 x PSU (C14, or C20 on 2400 W units) [E] | [E] model |
| BENCH-CTL, MONITOR, keyboard (table) | Lab PC, unknown | Table-top | Twin: 180 x 55 x 180 PC, 540 x 330 x 60 monitor [E] | Table | n/a | RJ-45 mgmt, video, USB, power [E] | [E] |

## Notes
- **Corrections for `/home/user/MARL_code/testbed_design/twin/catalog.py`.** Not applied; the repo is read-only for this task.

  | Item | Twin value | Corrected value |
  |---|---|---|
  | spider | d=400 | ~331 body, ~345 overall [E] |
  | sw_oob | d=300 | 400 [C]; port face is Netgear's rear |
  | rect | d=330 | 350.8 [C] |
  | fuse (Westell) | d=250, terminals rear | 147.8, all terminals front [C] |
  | dcdist | d=300 | ~280 [L]; terminal face TBC |
  | pdu | d=200 | 183 for PX3-5260R; 114 and 2U for AP7922B |
  | gnss | d=200 | 216 + connectors |
  | cm | d=90 | 95.3 |
  | blank | d=15 | 28 |
  | shelf / tray | 450 / 300 | 254 (10 in SKU) |
  | minipc | 120 x 40 x 120 | ~117 x 54 x 112 |
  | strip | 560 x 45 x 60 | ~485 x 44 x 44 |
  | psu | 170 x 45 x 70 | 167 x 35 x 67, lying flat |
  | injector | 70 x 35 x 140 | 62.5 x 35.3 x 138.5 |
  | specan | marked estimate | 408 x 186 x 235 is [C] |

  The rack rail depth of 750 does not fit R660/R760-class servers.
- **Electrical notes.**
  - The Westell panel is polarity-insensitive, so it suits -48 V with + grounded.
  - The DCD1M is built for positive-earth systems. Siklu says its DC input is floating and recommends + to ground, so this is probably fine [L].
  - The SW-OOB PoE+ (30 W per port) cannot power a node (48 W).
- **Hosts not reached** (no workaround tried): minicircuits, keysight, ceragon, pasternack, fairview, octopart, digikey, eaton, se/apc, netgear downloads, startech, panduit, meanwell, raritan CDN, navtechgps, manualslib, purdi, alliancecorporation and others. The NetBox device-type library repo was refused by the permission classifier.
- **Scratch files** are in `(session scratch) twin_research/`:
  - `spider_photo_measurements.txt` and `crops/`
  - `rf/rf_gear_sources.txt`
  - `power_notes.txt` and `dl/power/tessco/` (Westell TP and front_view.png)
  - `network_rack_notes.txt`
