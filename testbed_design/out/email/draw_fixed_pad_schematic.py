import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

RED, GREEN, GREY, INK, MUTED = "#c0392b", "#1e8449", "#34495e", "#1b1f24", "#5d6d7e"
fig = plt.figure(figsize=(16, 10.4), dpi=160)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 104); ax.axis("off")
fig.patch.set_facecolor("white")

def box(x, y, w, h, title, sub=None, ec=GREY, fc="#f4f6f7", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", ec=ec, fc=fc, lw=1.3))
    ax.text(x + w / 2, y + h / 2 + (1.0 if sub else 0), title, ha="center", va="center", fontsize=fs, weight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 1.4, sub, ha="center", va="center", fontsize=8, color=MUTED)

def pad(x, y, label, col):
    ax.add_patch(FancyBboxPatch((x, y - 1.6), 11, 3.2, boxstyle="round,pad=0.1,rounding_size=0.4", ec=col, fc="white", lw=1.6))
    ax.text(x + 5.5, y, label, ha="center", va="center", fontsize=9, weight="bold", color=col)

def splitter(x, y, name, port, col):
    ax.add_patch(Polygon([[x, y - 2], [x, y + 2], [x + 5, y]], closed=True, ec=col, fc="white", lw=1.5))
    ax.text(x + 2.5, y + 3.0, f"{name} {port}", ha="center", va="bottom", fontsize=7.5, color=MUTED)

ax.text(3, 100.5, "PoC 1 (splitter bench): IF wiring with fixed pads, no programmable attenuators", fontsize=17, weight="bold", color=INK)
ax.text(3, 97.2, "One row per link. Each link is 4 parallel SMA paths (tiles T1-T4 of the two sector modules), each with its own pad "
        "stack (and, on the hub W / E links, its own splitter). Values in dB, pads Mini-Circuits BW-SxW2+ (DC-18 GHz, 2 W). TENTATIVE.",
        fontsize=9.5, color=MUTED)

# column headers
for x, t in ((3, "A end: Spider box . module"), (28, "pad on the A tile"), (45, "splitter (hub W / E only)"),
             (70, "pad on the B tile"), (85, "B end: Spider box . module"), (118, "what happens at t0")):
    ax.text(x, 92.5, t, fontsize=9, weight="bold", color=GREY)

rows = [
    ("HUB-W1", "SB2 . M1", "HUB . W", "SPL-HUBW", "P1", "20 + 8", "SB1 . M1", "W1 . E", RED, "down: hub powered off"),
    ("HUB-W2", "SB2 . M1", "HUB . W", "SPL-HUBW", "P2", "20 + 3", "SB1 . M3", "W2 . E", RED, "down: hub powered off"),
    ("HUB-E1", "SB2 . M2", "HUB . E", "SPL-HUBE", "P1", "20 + 8", "SB4 . M1", "E1 . W", RED, "down: hub powered off"),
    ("HUB-E2", "SB2 . M2", "HUB . E", "SPL-HUBE", "P2", "20 + 3", "SB4 . M3", "E2 . W", RED, "down: hub powered off"),
    ("HUB-S", "SB2 . M3", "HUB . S", None, None, "30 + 5", "SB3 . M1", "S . N", RED, "down: hub powered off"),
    ("HUB-N", "SB2 . M4", "HUB . N", None, None, "30 + 7", "SB3 . M3", "N . S", RED, "down: hub powered off"),
    ("E2-W2", "SB4 . M4", "E2 . N", None, None, "30 + 1", "SB1 . M4", "W2 . S", GREEN, "bridge: held down, enabled by MARL"),
    ("E1-N", "SB4 . M2", "E1 . N", None, None, "30 + 4", "SB3 . M4", "N . W", GREEN, "bridge: held down, enabled by MARL"),
    ("W1-S", "SB1 . M2", "W1 . S", None, None, "30 + 12", "SB3 . M2", "S . E", GREEN, "bridge: held down, enabled by MARL"),
]
y = 87.5
for name, a_box, a_lab, spl, port, padv, b_box, b_lab, col, fate in rows:
    box(3, y - 2.6, 24, 5.2, f"{a_box}", a_lab, ec=col)
    ax.text(41, y + 2.9, name, fontsize=9.5, weight="bold", color=col)
    # each pad stack is drawn on the tile it screws onto: the hub W / E links' on the B tile (after the splitter and the
    # cable), the others' on the A tile (the splitter tray carries only splitters)
    if spl:
        ax.plot([27.5, 48], [y, y], color=col, lw=2)
        splitter(48, y, spl, port, col)
        ax.plot([53, 73.5], [y, y], color=col, lw=2)
        pad(73.5, y, padv, col)
    else:
        pad(27.8, y, padv, col)
        ax.plot([38.8, 85], [y, y], color=col, lw=2)
    box(85, y - 2.6, 24, 5.2, f"{b_box}", b_lab, ec=col)
    ax.text(111, y, "x 4", fontsize=9, color=MUTED, va="center")
    ax.text(118, y, fate, fontsize=9.5, color=col, va="center")
    y -= 7.0

# IP-50 hops
y0 = y - 1.0
ax.text(3, y0, "IP-50 hops on the bench table (one chain per hop, the Keysight J7204B is replaced by a fixed 10 dB pad)", fontsize=11,
        weight="bold", color=INK)
chain = ["radio", "WR-62\ntransition", "30 dB\n10 W", "10 dB", "SMA cable", "30 dB\n10 W", "WR-62\ntransition", "radio"]
xs = [3, 15, 27, 39, 51, 63, 75, 87]
yc = y0 - 6
for i, (x, t) in enumerate(zip(xs, chain)):
    ax.add_patch(FancyBboxPatch((x, yc - 2.6), 10, 5.2, boxstyle="round,pad=0.2,rounding_size=0.6", ec=GREY,
                                fc="#eef5ea" if "radio" in t else "white", lw=1.2))
    ax.text(x + 5, yc, t, ha="center", va="center", fontsize=8.5, color=INK)
    if i:
        ax.plot([xs[i - 1] + 10.2, x - 0.2], [yc, yc], color=GREY, lw=1.6)
hops = [("W1-CORE", RED, "down (CORE radio off)"), ("CORE-E1", RED, "down (CORE radio off)"),
        ("W1-W2", GREY, "unchanged"), ("E1-E2", GREY, "unchanged")]
for i, (h, col, f) in enumerate(hops):
    ax.text([3, 27, 51, 68][i], yc - 6.0, f"{h}: {f}", fontsize=9, color=col, weight="bold" if col != GREY else "normal")

# right panel: the t0 event
px, py = 118, y0 - 0.5
ax.add_patch(FancyBboxPatch((px - 1, py - 19), 41, 21, boxstyle="round,pad=0.4,rounding_size=1", ec=RED, fc="#fdf2f0", lw=1.3))
ax.text(px + 1, py - 0.2, "t0: switched PDU outlets turned OFF", fontsize=10.5, weight="bold", color=RED)
for i, t in enumerate(["INJ-HUB (PoE injector of the hub node)", "PSU of IP-50 W1-CORE @ CORE",
                       "PSU of IP-50 CORE-E1 @ CORE", "SRV-E2E (E2E controller)", "SRV-DPIR (planner)"]):
    ax.text(px + 1, py - 4.2 - i * 3.0, "- " + t, fontsize=9.3, color=INK)

# legend / notes at the bottom left
ny = 6.0
ax.plot([3, 9], [ny, ny], color=RED, lw=2.4); ax.text(10, ny, "goes down at t0 because a device is powered off", va="center", fontsize=9, color=INK)
ax.plot([52, 58], [ny, ny], color=GREEN, lw=2.4); ax.text(59, ny, "bridge: link disabled in Terragraph until MARL enables it", va="center", fontsize=9, color=INK)
ax.text(3, 2.6, "No attenuator moves in the scenario, so fixed pads set every path loss (values = the former attenuator settings, "
        "rounded to what two pads make; corrected at the port check).", fontsize=8.8, color=MUTED)
ax.text(3, 0.6, "Every pad stack screws onto a Spider tile: the hub W / E links' on the B-end tile (splitter P1 / P2 cabled to it), "
        "the others' on the A-end tile; the splitter tray carries only splitters. Torque SMA to 8 in-lb (backing-wrench rule: install guide).",
        fontsize=8.8, color=MUTED)
fig.savefig("/home/user/MARL_code/testbed_design/out/email/poc1_schematic_fixed_pads.png", dpi=160)
print("ok")
