# Open questions for the lab (what would firm up the twin)

The twin is TENTATIVE. Answers already given (2026-11-24): attenuators are small 4-channel boxes with a dial; PDUs are
1U; a standard 42U rack; Ceragon IP-50CX radios; servers and hosts are 1U/2U rack servers with a PC and monitor on the
table. What still moves cables and lengths, most important first:

1. **Spider boxes, rear panels (photo received 2026-11-25).** The lab photo of the four rears is in the twin: per node
   window two RJ-45, an SFP+ cage and a USB-A, one green 4-pole DC header with a screw flange per box, an RJ-45 in its own
   bracket (internal Raspberry Pi assumed) and the GNSS SMA. Still open: which of the two RJ-45 in each node window is
   ETH1, and which SMA serves which node. The hub box shows no SMA next to its node window: does the hub node take GNSS
   at all? If not, {gnss_change}.
2. **Spider boxes, green 4-pole DC header (DCIN).** The headers are bare in the photo (no plug fitted). What does the
   header feed: the nodes, or only the box fans and the Raspberry Pi? At what voltage, which pin is which (pin order,
   polarity), and what is the pitch (5.08 or 7.62 mm)? If it powers the nodes, {dcin_change}.
3. **Spider boxes, front and identity.** Which GTS003 serial is #1 (SF06000007 is assumed) and which tile SMA belongs to
   which module and node. Front SMA positions were measured from photos to about 8 mm.
4. **Internal Raspberry Pi.** Does it need a LAN port and an IP on the management switch? (Up to 4 more SW-OOB ports;
   PoC 1 uses 19 of 26 RJ-45 since the fixed-pad change, so they would fit.)
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
