"""Device catalog for the bench digital twins: outer dimensions, colours and panel layouts.

TENTATIVE: values marked 'estimate' are placeholders until the lab confirms them (photos or a tape
measure); twin_model.py writes every such assumption into the scene and into the installation guide.
Units: mm.  Rack x runs left -> right across the 19-inch panel (0 = left edge of the ears), y is up from the
floor, z runs front -> rear (0 = front rail plane).
"""
U = 44.45                  # one rack unit
PANEL_W = 482.6            # 19-inch panel incl. ears
BODY_W = 440.0             # typical chassis body between the ears

# rack: StarTech 4POSTRACK42 (adjustable depth 22-40 in per retail listings); depth set by us
RACK = dict(part="StarTech 4POSTRACK42", units=42, u1_y=120.0, rail_depth=900.0, frame_w=600.0,
            assumed=["StarTech 4POSTRACK42 (rails adjustable 559-1016 mm, square holes, M6 cage nuts); rail-to-rail depth "
                     "set to 900 mm so 800 mm-class rack servers fit", "U1 bottom 120 mm above the floor (on castors, estimate)"])

# Spider boxes, front SMA x (mm from the left panel edge, tiles M1T1..M4T4) and SMA height, measured from the lab
# photos (+/- 8 mm); which GTS003 serial is #1 is not known: SB1 = SF06000007 assumed. Serials: bench_spec.BOX_SERIAL
# (keyed by the box's lab label, so a box keeps its serial in every PoC) is the source; SPIDER_SERIAL is a fallback
SPIDER_VARIANT = {"SB1": "A", "SB2": "B", "SB3": "C", "SB4": "C"}
SPIDER_SERIAL = {"SB1": "SF06000007 (assumed #1)", "SB2": "SF06000001", "SB3": "SF06000058", "SB4": "SF06000031 (assumed #2)"}
SPIDER_FRONT = {
    "A": ([141, 162, 184, 205, 224, 244, 264, 284, 303, 322, 342, 362, 378, 398, 416, 435], 19.0),
    "B": ([95, 117, 138, 159, 183, 204, 224, 244, 265, 285, 305, 325, 346, 365, 384, 403], 20.0),
    "C": ([140, 162, 182, 203, 222, 242, 263, 283, 302, 321, 341, 360, 378, 396, 415, 434], 27.0),
}
# rear panel (front-view x, mm; seen from inside one GTS003, +/- 10): per node group USB, SFP+, two RJ-45 (ETH1 assumed
# first), a GNSS SMA per node, the internal Raspberry Pi's RJ-45 and a box DC terminal block (functions to confirm)
SPIDER_REAR = dict(node_a=dict(gnss=47, usb=119, sfp=150, eth1=186, eth2=223),
                   node_b=dict(gnss=245, usb=328, sfp=365, eth1=400, eth2=434), pi=69, dcin=267, y=22.0)

def spider_tile_x(box, k, t):
    """x of tile t of module M<k> on box SBn (photo-measured)."""
    return SPIDER_FRONT[SPIDER_VARIANT.get(box, "C")][0][(k - 1) * 4 + (t - 1)]

# bench table next to the rack (right-hand side, operator in front at z < 0)
TABLE = dict(x0=700.0, x1=4300.0, z0=-150.0, z1=650.0, top=750.0,
             assumed=["one 3.6 m x 0.8 m bench table to the right of the rack, top at 750 mm (estimate)"])

# kind -> depth, colour, label colour, confidence and notes; width/height come from the rack units or the table size
KIND = {
    "spider":   dict(d=345, color="#c9ced6", conf="estimate", note="Siklu 1U Spider box: brushed aluminium front, 16 SMA measured from lab photos (+/-8 mm); depth ~331 body / 345 overall and the rear panel (seen from inside one box) to confirm"),
    "ztdat":    dict(d=160, color="#5e81ac", conf="estimate", note="programmable attenuator: small 4-channel box with a dial (user), 200 x 70 x 160 mm assumed, SMA on the front"),
    "attshelf": dict(d=254, color="#4c566a", conf="likely", note="2U of rack space: a 1U vented shelf StarTech CABSHELFV1U (254 mm deep) with 1U clear above for the attenuator boxes"),
    "rackserver": dict(d=500, color="#434c5e", conf="estimate", note="lab rack server, stand-in Dell R260 (1U, 497 mm) / R660 (1U, 823 mm) / R760 (2U, 772 mm)"),
    "tray":     dict(d=254, color="#4c566a", conf="likely", note="1U vented shelf StarTech CABSHELFV1U (254 mm deep) carrying splitters / couplers"),
    "shelf":    dict(d=254, color="#4c566a", conf="likely", note="1U vented shelf StarTech CABSHELFV1U (254 mm)"),
    "dcdist":   dict(d=280, color="#d08770", conf="estimate", note="Eaton DCD1M (~280 mm): breakers front; load terminal face to confirm (rear assumed)"),
    "fuse":     dict(d=148, color="#e5e0d0", conf="likely", note="Westell N250110-N-0803 (147.8 mm, vendor TP): all terminals and GMT fuses on the FRONT"),
    "rect":     dict(d=351, color="#bf616a", conf="likely", note="Mean Well RCP-1UI (350.8 mm): 3 module slots, one C14 inlet per slot, DC bus at the rear; one AC cord per fitted module"),
    "pdu":      dict(d=183, color="#2e3440", conf="estimate", note="1U switched PDU, 12 x C13, 16 A (Raritan PX3-5260R class, 183 mm); mounted on the REAR posts with the outlets and network port facing the rack rear (assumed)"),
    "sw_data":  dict(d=400, color="#4f6f52", conf="likely", note="Netgear M4350-24F4V 440 x 43.2 x 400 mm; ports on the front"),
    "sw_oob":   dict(d=400, color="#4f6f52", conf="likely", note="Netgear GSM4230PX 440 x 43.2 x 400 mm; all ports on Netgear's 'back' face, mounted on the REAR posts with the ports flush with the rack rear"),
    "gnss":     dict(d=216, color="#8fbcbb", conf="likely", note="RMALDCBS1X8 1U, 216 mm, 5.4 kg; connectors at the rear assumed"),
    "cm":       dict(d=95, color="#2e3440", conf="likely", note="Panduit NCMHF1 1U horizontal cable manager"),
    "blank":    dict(d=28, color="#20242c", conf="likely", note="APC AR8136BLK toolless blanking panel"),
}

# table devices: (w, h, d), colour, note
TABLE_DEV = {
    "mwatt":   dict(size=(425.6, 88.3, 578.1), color="#5e81ac", conf="likely", note="Keysight J7204B (parts research dims); SMA ports on the front assumed"),
    "ip50":    dict(size=(322, 86, 270), color="#a3be8c", conf="estimate", note="Ceragon IP-50CX (about 6 kg; size estimate, ceragon.com not reachable) laid flat; waveguide flange on the side toward MWATT"),
    "monitor": dict(size=(540, 330, 60), color="#2e3440", conf="estimate", note="24-inch monitor on a stand"),
    "keyboard": dict(size=(440, 25, 140), color="#4c566a", conf="estimate", note="keyboard + mouse"),
    "server":  dict(size=(440, 88, 600), color="#434c5e", conf="estimate", note="2U lab server lying on the table"),
    "pc":      dict(size=(180, 55, 180), color="#4c566a", conf="estimate", note="small form-factor PC"),
    "specan":  dict(size=(408, 186, 235), color="#81a1c1", conf="likely", note="R&S FPL1026 408 x 186 x 235 mm; RF in (2.92 mm F) at the lower right of the front"),
    "strip":   dict(size=(485, 44, 44), color="#2e3440", conf="estimate", note="bench mains strip, 6 x C13 outlets, C20 inlet, 16 A, with an on/off switch (part TBC); fed from its own wall socket, never from a switched PDU outlet; laid lengthwise front to back on the table"),
    "psu":     dict(size=(167, 35, 67), color="#3b4252", conf="estimate", note="isolated 48 V 120 W desktop supply (e.g. Mean Well GST120A48), output floating, wired -48 V (PSU - to the radio -48 V pin, PSU + to the radio 0 V pin) through a barrel-to-terminal adaptor"),
    "minipc":  dict(size=(117, 54, 112), color="#4c566a", conf="estimate", note="agent mini PC (NUC-class stand-in)"),
}

# small parts that sit on shelves / trays
SMALL = {
    "adaptor":  dict(size=(30, 25, 95), color="#ebcb8b", note="Siklu EH-PoE-DC-Adaptor (size estimate)"),
    "injector": dict(size=(62.5, 35.3, 138.5), color="#bf616a", note="Siklu AX-IN-10G-60W-AC-PoE 138.5 x 62.5 x 35.3 mm, C6 inlet"),
    # tray parts lie flat with their long axis along x: input end to the LEFT, output end to the RIGHT (seen from the front)
    "splitter": dict(size=(51.9, 10.4, 25.4), color="#3b6ea5", note="Mini-Circuits ZN2PD-02183-S+ 51.9 x 25.4 x 10.4 mm (blue), lying flat: S on the left end face, ports 1 and 2 on the right end face, label up"),
    "coupler":  dict(size=(90, 14, 19), color="#b0b7c3", note="10 dB directional coupler KRYTAR 1821 (~90 x 19 x 14 mm, size estimate), lying flat: IN on the left end, THRU on the right end, CPL on the top face"),
    # standing on each coupler's top CPL port (PoC 2 coupling arms): DC block, then (input side) the arm pad
    # height per stack from twin_model (input side: DC block + pad, 52 mm; output side: DC block only, 30 mm)
    "armstack": dict(size=(10, 52, 10), color="#d8a657", note="BLK-18-S+ DC block (about 30 mm), on the arm input side topped by the arm's BW-S pad (about 22 mm), 10 mm across (estimates), standing on the coupler's CPL port"),
    "spacer":   dict(size=(300, 25, 25), color="#8f9aa8", note="25 x 25 mm aluminium spacer bar, 300 mm (part TBC): two per stacked IP-50 radio, at the front and rear edges of the lower lid"),
    "earthbar": dict(size=(30, 200, 10), color="#2e7d32", note="rack earth bar with M6 studs on the left rear post (part TBC, in the grounding kit)"),
    "arrestor": dict(size=(80, 40, 25), color="#2e7d32", note="GNSS surge arrestor, DC pass (the splitter powers the antenna through it), N-female both ends, ground lug (part TBC)"),
    "bridge":   dict(size=(135, 6, 140), color="#6b7785", note="rack-to-table cable bridge: wire-basket cable tray section with a divider (data in front, power behind) on two brackets, rack right post to the table edge, its floor at table height (part TBC)"),
    "transition": dict(size=(33.3, 33.3, 35), color="#d8dee9", note="Pasternack PEWCA1003 WR-62 to SMA-F (33.3 mm flange); band to confirm"),
}

# colours by cable class (light theme)
CABLE = {"AC": "#c0392b", "DC": "#d4a017", "POE": "#e67e22", "FIB": "#16a085", "ETH": "#2980b9", "RF": "#8e44ad",
         "GNSS": "#27ae60", "WG": "#c2185b", "AUX": "#7f8c8d", "PE": "#9acd32"}
CABLE_NAME = {"AC": "AC mains", "DC": "DC power", "POE": "PoE (power only) to node ETH1", "FIB": "data to SW-DATA",
              "ETH": "management to SW-OOB", "RF": "IF coax (SMA)", "GNSS": "GNSS coax", "WG": "IP-50 chain",
              "AUX": "operator station (video, USB)", "PE": "earth bonding (green/yellow)"}
CABLE_DIA = {"AC": 7.0, "DC": 5.0, "POE": 6.0, "FIB": 3.0, "ETH": 6.0, "RF": 3.6, "GNSS": 4.0, "WG": 4.0, "AUX": 5.0,
             "PE": 6.0}
POWER_CLS = ("AC", "DC", "POE", "PE")      # dressed on the LEFT of the rack (seen from the front), own table lane

# standard-length ladders: the twin picks the shortest that covers the routed length + 10 % (DC is cut from bulk)
LADDER = {
    "AC_C13": [("P004-006", 1830, "Eaton Tripp Lite P004-006 C14-C13 lead 1.83 m", "confirmed"),
               ("P004-010", 3050, "Eaton Tripp Lite P004-010 C14-C13 lead 3.05 m", "confirmed"),
               ("P004-015", 4570, "Eaton Tripp Lite P004-015 C14-C13 lead 4.57 m", "likely")],
    "AC_C5": [("P014-006", 1830, "Eaton Tripp Lite P014-006 C14-C5 'Mickey' lead 1.83 m", "confirmed"),
              ("P014-010", 3050, "Eaton Tripp Lite P014-010 C14-C5 lead 3.05 m", "likely"),
              ("C14-C5-5M", 5000, "C14-C5 lead 5 m (part TBC)", "to confirm")],
    "CAT6A": [("STP6X1MBU", 1000, "Panduit STP6X1MBU Cat6A S/FTP 1 m", "confirmed"),
              ("STP6X2MBU", 2000, "Panduit STP6X2MBU Cat6A S/FTP 2 m", "confirmed"),
              ("STP6X3MBU", 3000, "Panduit STP6X3MBU Cat6A S/FTP 3 m", "confirmed"),
              ("STP6X5MBU", 5000, "Panduit STP6X5MBU Cat6A S/FTP 5 m", "likely"),
              ("STP6X7MBU", 7000, "Panduit STP6X7MBU Cat6A S/FTP 7 m", "likely")],
    "DAC": [("AXC761-10000S", 1000, "Netgear AXC761-10000S SFP+ DAC 1 m", "confirmed"),
            ("AXC763-10000S", 3000, "Netgear AXC763-10000S SFP+ DAC 3 m", "confirmed")],
    "AC_C19": [("CORD-60309-C19-3M", 3000, "CORD-60309-C19-3M power cord IEC 60309 16 A (2P+E) plug to IEC C19, 3 m (part TBC)",
                "to confirm"),
               ("CORD-60309-C19-5M", 5000, "CORD-60309-C19-5M power cord IEC 60309 16 A (2P+E) plug to IEC C19, 5 m (part TBC)",
                "to confirm")],
    "OM4": [("FZ2ERLNLNSNM001", 1000, "Panduit FZ2ERLNLNSNM001 OM4 LC-LC 1 m", "likely"),
            ("FZ2ERLNLNSNM002", 2000, "Panduit FZ2ERLNLNSNM002 OM4 LC-LC 2 m", "confirmed"),
            ("FZ2ERLNLNSNM003", 3000, "Panduit FZ2ERLNLNSNM003 OM4 LC-LC 3 m", "likely"),
            ("FZ2ERLNLNSNM005", 5000, "Panduit FZ2ERLNLNSNM005 OM4 LC-LC 5 m", "likely"),
            ("FZ2ERLNLNSNM007", 7000, "Panduit FZ2ERLNLNSNM007 OM4 LC-LC 7 m", "to confirm")],
    "RG316": [("RG-316 SMA 1 m", 1000, "RG-316 SMA coax 1 m (part TBC)", "to confirm"),
              ("RG-316 SMA 1.5 m", 1500, "RG-316 SMA coax 1.5 m (part TBC)", "to confirm"),
              ("RG-316 SMA 2 m", 2000, "RG-316 SMA coax 2 m (part TBC)", "to confirm"),
              ("RG-316 SMA 2.5 m", 2500, "RG-316 SMA coax 2.5 m (part TBC)", "to confirm"),
              ("RG-316 SMA 3 m", 3000, "RG-316 SMA coax 3 m (part TBC)", "to confirm")],
    # GNSS arrestor (equipment side) to the splitter input
    "LMR240": [("LMR-240 N 1 m", 1000, "LMR-240 coax jumper, N-male (arrestor) to the splitter input (connector TBC), 1 m (part TBC)",
                "to confirm"),
               ("LMR-240 N 2 m", 2000, "LMR-240 coax jumper, N-male (arrestor) to the splitter input (connector TBC), 2 m (part TBC)",
                "to confirm"),
               ("LMR-240 N 3 m", 3000, "LMR-240 coax jumper, N-male (arrestor) to the splitter input (connector TBC), 3 m (part TBC)",
                "to confirm"),
               ("LMR-240 N 5 m", 5000, "LMR-240 coax jumper, N-male (arrestor) to the splitter input (connector TBC), 5 m (part TBC)",
                "to confirm")],
}
# mounting: kind -> (where it mounts, M6 cage nut + screw sets at the front posts, at the rear posts, extra hardware)
# rule: 4 sets per mounted item per post pair (2 per ear); rail-kit servers and toolless blanks take none
# UNIRAILS1U (StarTech listing): 420-1054 mm mounting depth, ships with its own M5 cage nuts and panel screws, and each
# rail bolts to its front AND rear post with that kit hardware; the M6 sets counted here are for the ears only
RAILS = "StarTech UNIRAILS1U adjustable 1U rails, bolted to front and rear posts with the kit's own M5 cage nuts and screws"
MOUNT = {
    "spider":     ("front posts (ears) + rails front and rear", 4, 0, RAILS + " (345 mm box)"),
    "attshelf":   ("front posts (2-post shelf, 1U clear above)", 4, 0, ""),
    "rackserver": ("4-post rail kit (shipped with the server)", 0, 0, "toolless square-hole rails"),
    "tray":       ("front posts (2-post shelf)", 4, 0, ""),
    "shelf":      ("front posts (2-post shelf)", 4, 0, ""),
    "dcdist":     ("front posts", 4, 0, ""),
    "fuse":       ("front posts", 4, 0, ""),
    "rect":       ("front posts (ears) + rails front and rear", 4, 0, RAILS + " (351 mm shelf, up to 3 modules)"),
    "pdu":        ("rear posts, outlets facing the rear", 0, 4, ""),
    "sw_data":    ("front posts (ears), ports facing the front + rails front and rear", 4, 0,
                   RAILS + " (400 mm, 6.25 kg; Netgear RBS1830 fits 18-30 in rails only, not 900 mm)"),
    "sw_oob":     ("rear posts, ports facing the rear", 0, 4, ""),
    "gnss":       ("front posts, connectors facing the rear", 4, 0, ""),
    "cm":         ("front posts", 4, 0, ""),
    "blank":      ("front posts, toolless", 0, 0, ""),
}
CAGE_PACK = 50          # M6 cage nut + screw + washer sets per pack (StarTech CABSCREWM6 class)

LADDER_CONF = {pn: conf for v in LADDER.values() for pn, _, _, conf in v}

# typical mains draw per load, W (TENTATIVE estimates for the PDU / strip budget; replace with nameplate or measured
# values). Rectifier modules are computed from their DC load in twin_model (each module rated for the full load).
LOAD_W = {"SW-DATA": 120, "SW-OOB": 60, "GNSS-SPL": 10, "ATT": 15, "AGENT": 65, "HOST": 250, "SRV-E2E": 350,
          "SRV-DPIR": 800, "INJ": 70, "PSU": 135, "BENCH-CTL": 150, "MONITOR": 30, "SPECAN": 60, "MWATT": 50}
NODE_W, IP50_W, RECT_EFF = 48, 58, 0.9
BOX_AUX_W = 15                      # Spider box fans + Raspberry Pi on its DCIN terminal block (estimate)
RCP_MODULE_A = 1000 / 48 * 1.05     # RCP-1000-48 output at its current limit, about 22 A per module (estimate)
CIRCUIT_W = 16 * 230 * 0.8          # 16 A circuit at 230 V, loaded to 80 %
SLACK = 1.10
