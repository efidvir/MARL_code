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
# photos (+/- 8 mm); which GTS003 serial is #1 is not known: SB1 = SF06000007 assumed
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
    "attshelf": dict(d=450, color="#4c566a", conf="estimate", note="2U of rack space with a 1U vented shelf carrying two attenuator boxes"),
    "rackserver": dict(d=500, color="#434c5e", conf="estimate", note="lab rack server, stand-in Dell R260 (1U, 497 mm) / R660 (1U, 823 mm) / R760 (2U, 772 mm)"),
    "tray":     dict(d=254, color="#4c566a", conf="likely", note="1U vented shelf StarTech CABSHELFV1U (254 mm) carrying splitters / couplers"),
    "shelf":    dict(d=254, color="#4c566a", conf="likely", note="1U vented shelf StarTech CABSHELFV1U (254 mm)"),
    "dcdist":   dict(d=280, color="#d08770", conf="estimate", note="Eaton DCD1M (~280 mm): breakers front; load terminal face to confirm (rear assumed)"),
    "fuse":     dict(d=148, color="#e5e0d0", conf="likely", note="Westell N250110-N-0803 (147.8 mm, vendor TP): all terminals and GMT fuses on the FRONT"),
    "rect":     dict(d=351, color="#bf616a", conf="likely", note="Mean Well RCP-1UI (350.8 mm): 3 C14 inlets and the DC bus at the rear"),
    "pdu":      dict(d=183, color="#2e3440", conf="estimate", note="1U switched PDU, 12 x C13, 16 A (Raritan PX3-5260R class, 183 mm); outlets on the rear assumed"),
    "sw_data":  dict(d=400, color="#4f6f52", conf="likely", note="Netgear M4350-24F4V 440 x 43.2 x 400 mm; ports on the front"),
    "sw_oob":   dict(d=400, color="#4f6f52", conf="likely", note="Netgear GSM4230PX 440 x 43.2 x 400 mm; all ports on Netgear's 'back' face, mounted facing the rack rear"),
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
    "strip":   dict(size=(485, 44, 44), color="#2e3440", conf="estimate", note="bench mains strip, C13 (part TBC)"),
    "psu":     dict(size=(167, 35, 67), color="#3b4252", conf="estimate", note="isolated 48 V 120 W desktop supply (e.g. Mean Well GST120A48)"),
    "minipc":  dict(size=(117, 54, 112), color="#4c566a", conf="estimate", note="agent mini PC (NUC-class stand-in)"),
}

# small parts that sit on shelves / trays
SMALL = {
    "adaptor":  dict(size=(30, 25, 95), color="#ebcb8b", note="Siklu EH-PoE-DC-Adaptor (size estimate)"),
    "injector": dict(size=(62.5, 35.3, 138.5), color="#bf616a", note="Siklu AX-IN-10G-60W-AC-PoE 138.5 x 62.5 x 35.3 mm, C6 inlet"),
    "splitter": dict(size=(25.4, 10.4, 51.9), color="#3b6ea5", note="Mini-Circuits ZN2PD-02183-S+ 51.9 x 25.4 x 10.4 mm (blue), S / 1 / 2 SMA-F"),
    "coupler":  dict(size=(14, 19, 90), color="#b0b7c3", note="10 dB coupler KRYTAR 1821 (~90 x 14 x 19 mm estimate); ZUDC10-183+ is 152 mm long"),
    "transition": dict(size=(33.3, 33.3, 35), color="#d8dee9", note="Pasternack PEWCA1003 WR-62 to SMA-F (33.3 mm flange); band to confirm"),
}

# colours by cable class (light theme)
CABLE = {"AC": "#c0392b", "DC": "#d4a017", "POE": "#e67e22", "FIB": "#16a085", "ETH": "#2980b9", "RF": "#8e44ad",
         "GNSS": "#27ae60", "WG": "#c2185b", "AUX": "#7f8c8d"}
CABLE_NAME = {"AC": "AC mains", "DC": "DC power", "POE": "PoE (power only) to node ETH1", "FIB": "data to SW-DATA",
              "ETH": "management to SW-OOB", "RF": "IF coax (SMA)", "GNSS": "GNSS coax", "WG": "IP-50 chain",
              "AUX": "operator station (video, USB)"}
CABLE_DIA = {"AC": 7.0, "DC": 5.0, "POE": 6.0, "FIB": 3.0, "ETH": 6.0, "RF": 3.6, "GNSS": 4.0, "WG": 4.0, "AUX": 5.0}

# standard-length ladders: the twin picks the shortest that covers the routed length + 10 % (DC is cut from bulk)
LADDER = {
    "AC_C13": [("P004-006", 1830, "Eaton Tripp Lite P004-006 C14-C13 lead 1.83 m", "confirmed"),
               ("P004-010", 3050, "Eaton Tripp Lite P004-010 C14-C13 lead 3.05 m", "confirmed"),
               ("P004-015", 4570, "Eaton Tripp Lite P004-015 C14-C13 lead 4.57 m", "likely")],
    "AC_C5": [("P014-006", 1830, "Eaton Tripp Lite P014-006 C14-C5 'Mickey' lead 1.83 m", "confirmed"),
              ("P014-010", 3050, "Eaton Tripp Lite P014-010 C14-C5 lead 3.05 m", "likely")],
    "CAT6A": [("STP6X1MBU", 1000, "Panduit STP6X1MBU Cat6A S/FTP 1 m", "confirmed"),
              ("STP6X2MBU", 2000, "Panduit STP6X2MBU Cat6A S/FTP 2 m", "confirmed"),
              ("STP6X3MBU", 3000, "Panduit STP6X3MBU Cat6A S/FTP 3 m", "confirmed"),
              ("STP6X5MBU", 5000, "Panduit STP6X5MBU Cat6A S/FTP 5 m", "likely"),
              ("STP6X7MBU", 7000, "Panduit STP6X7MBU Cat6A S/FTP 7 m", "likely")],
    "DAC": [("AXC761-10000S", 1000, "Netgear AXC761-10000S SFP+ DAC 1 m", "confirmed"),
            ("AXC763-10000S", 3000, "Netgear AXC763-10000S SFP+ DAC 3 m", "confirmed")],
    "OM4": [("FZ2ERLNLNSNM001", 1000, "Panduit FZ2ERLNLNSNM001 OM4 LC-LC 1 m", "likely"),
            ("FZ2ERLNLNSNM002", 2000, "Panduit FZ2ERLNLNSNM002 OM4 LC-LC 2 m", "confirmed"),
            ("FZ2ERLNLNSNM003", 3000, "Panduit FZ2ERLNLNSNM003 OM4 LC-LC 3 m", "likely"),
            ("FZ2ERLNLNSNM005", 5000, "Panduit FZ2ERLNLNSNM005 OM4 LC-LC 5 m", "likely")],
}
LADDER_CONF = {pn: conf for v in LADDER.values() for pn, _, _, conf in v}
SLACK = 1.10
