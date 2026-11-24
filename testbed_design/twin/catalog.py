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
RACK = dict(part="StarTech 4POSTRACK42", units=42, u1_y=100.0, rail_depth=750.0, frame_w=600.0,
            assumed=["rail-to-rail depth set to 750 mm (adjustable)", "U1 bottom 100 mm above the floor (estimate)"])

# bench table next to the rack (right-hand side, operator in front at z < 0)
TABLE = dict(x0=700.0, x1=4300.0, z0=-150.0, z1=650.0, top=750.0,
             assumed=["one 3.6 m x 0.8 m bench table to the right of the rack, top at 750 mm (estimate)"])

# kind -> depth, colour, label colour, confidence and notes; width/height come from the rack units or the table size
KIND = {
    "spider":   dict(d=400, color="#3b4252", conf="estimate", note="Siklu 1U Spider box: depth and rear panel unknown"),
    "ztdat":    dict(d=160, color="#5e81ac", conf="estimate", note="programmable attenuator: small 4-channel box with a dial (user), 200 x 70 x 160 mm assumed, SMA on the front"),
    "attshelf": dict(d=450, color="#4c566a", conf="estimate", note="2U of rack space with a 1U vented shelf carrying two attenuator boxes"),
    "rackserver": dict(d=600, color="#434c5e", conf="estimate", note="1U/2U rack server (lab), 600 mm deep assumed"),
    "tray":     dict(d=300, color="#4c566a", conf="estimate", note="1U vented shelf (CABSHELFV1U) carrying splitters / couplers"),
    "shelf":    dict(d=450, color="#4c566a", conf="estimate", note="1U vented shelf"),
    "dcdist":   dict(d=300, color="#d08770", conf="estimate", note="Eaton DCD1M: breakers front, terminals rear assumed"),
    "fuse":     dict(d=250, color="#d08770", conf="estimate", note="Westell N250110-N-0803: fuses front, terminals rear assumed"),
    "rect":     dict(d=330, color="#bf616a", conf="estimate", note="Mean Well RCP-1UI: AC C14 and DC output at the rear assumed"),
    "pdu":      dict(d=200, color="#2e3440", conf="estimate", note="1U switched PDU, 12 x C13 (user: 1U); outlets on the rear assumed"),
    "sw_data":  dict(d=400, color="#4f6f52", conf="likely", note="Netgear M4350-24F4V 440 x 43.2 x 400 mm (parts research)"),
    "sw_oob":   dict(d=300, color="#4f6f52", conf="estimate", note="Netgear GSM4230PX: depth estimate"),
    "gnss":     dict(d=200, color="#8fbcbb", conf="estimate", note="RMALDCBS1X8 1U assumed"),
    "cm":       dict(d=90, color="#2e3440", conf="likely", note="Panduit NCMHF1 1U horizontal cable manager"),
    "blank":    dict(d=15, color="#20242c", conf="likely", note="APC AR8136BLK blanking panel"),
}

# table devices: (w, h, d), colour, note
TABLE_DEV = {
    "mwatt":   dict(size=(425.6, 88.3, 578.1), color="#5e81ac", conf="likely", note="Keysight J7204B (parts research dims); SMA ports on the front assumed"),
    "ip50":    dict(size=(270, 110, 270), color="#a3be8c", conf="estimate", note="Ceragon IP-50CX (about 6 kg; size estimate, ceragon.com not reachable) laid flat; waveguide flange on the side toward MWATT"),
    "monitor": dict(size=(540, 330, 60), color="#2e3440", conf="estimate", note="24-inch monitor on a stand"),
    "keyboard": dict(size=(440, 25, 140), color="#4c566a", conf="estimate", note="keyboard + mouse"),
    "server":  dict(size=(440, 88, 600), color="#434c5e", conf="estimate", note="2U lab server lying on the table"),
    "pc":      dict(size=(180, 55, 180), color="#4c566a", conf="estimate", note="small form-factor PC"),
    "specan":  dict(size=(408, 186, 235), color="#81a1c1", conf="estimate", note="R&S FPL1026 (estimate)"),
    "strip":   dict(size=(560, 45, 60), color="#2e3440", conf="estimate", note="bench mains strip, C13"),
    "psu":     dict(size=(170, 45, 70), color="#3b4252", conf="estimate", note="isolated 48 V desktop supply"),
    "minipc":  dict(size=(120, 40, 120), color="#4c566a", conf="estimate", note="agent mini PC"),
}

# small parts that sit on shelves / trays
SMALL = {
    "adaptor":  dict(size=(30, 25, 95), color="#ebcb8b", note="Siklu EH-PoE-DC-Adaptor (size estimate)"),
    "injector": dict(size=(70, 35, 140), color="#bf616a", note="Siklu AX-IN-10G-60W-AC-PoE (size estimate)"),
    "splitter": dict(size=(16, 14, 32), color="#b48ead", note="Mini-Circuits ZN2PD-02183-S+ (size estimate)"),
    "coupler":  dict(size=(14, 14, 36), color="#b48ead", note="10 dB coupler KRYTAR 1821 / ZUDC10-183+ (size estimate)"),
    "transition": dict(size=(40, 40, 45), color="#d8dee9", note="Pasternack PEWCA1003 (size estimate)"),
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
