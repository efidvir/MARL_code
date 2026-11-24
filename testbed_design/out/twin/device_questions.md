# Open questions for the lab (what would firm up the twin)

The twin is TENTATIVE. Answers already given (2026-11-24): attenuators are small 4-channel boxes with a dial; PDUs are
1U; a standard 42U rack; Ceragon IP-50CX radios; servers and hosts are 1U/2U rack servers with a PC and monitor on the
table. What still moves cables and lengths, most important first:

1. **Spider boxes, rear panels (all four).** One straight-on photo of each rear, from outside, with a tape along the
   bottom edge, and a close-up of the green terminal block. Only one box was seen, from inside: per node an SFP+ cage,
   two RJ-45 (which is ETH1?), a USB part and a GNSS SMA, plus one RJ-45 for the internal Raspberry Pi and one DC
   terminal block per box.
2. **Spider boxes, power.** What goes into the terminal block (voltage, polarity, plug)? Is each node powered from it,
   or does each node still need PoE on ETH1? If the box has its own DC inlet, the ten DC/PoE adaptors, the breaker
   feeds and the PoE patch cords change.
3. **Spider boxes, front and identity.** Which GTS003 serial is #1 (SF06000007 is assumed) and which tile SMA belongs to
   which module and node. Front SMA positions were measured from photos to about 8 mm.
4. **Internal Raspberry Pi.** Does it need a LAN port and an IP on the management switch? (Up to 4 more SW-OOB ports;
   PoC 1 would then outgrow the GSM4230PX.)
5. **IP-50CX.** Band (the 23 GHz version would need the 18-26.5 GHz attenuator chain instead of the J7204B), the
   waveguide or antenna interface, the DC connector, and the ports used (1G RJ-45, 10G SFP+, 25G SFP28).
6. **Where the bench table stands** relative to the rack (side, gap, height) and where the wall sockets are. This sets
   every rack-to-table cable length.
7. **Switches as mounted.** The GSM4230PX has all its ports on Netgear's 'back' face; the twin mounts it ports-to-the-
   rear. Confirm, and confirm the M4350 port numbering (odd ports top).
8. **DC drawer.** Which face of the Eaton DCD1M carries the load terminals (sources disagree); the Westell -48 V panel has
   all terminals on the front.
9. **Attenuator boxes, when chosen.** Make and model, SMA positions, IN/OUT labelling, power (brick or C14) and LAN.
10. **GTSTU022 / GTSTU023** (kit item 1): what they are and how many.
