# -*- coding: utf-8 -*-
"""Tests for the single definition of "this site can bridge".

The engine, the severance generator and the attainable-optimum solver must all
agree, because they disagreed before: the engine can only steer a site that
owns a PHYMACState, while the other two asked has_multihaul, a topology flag
also set on nodes with no radio hardware. On the evaluation topology that was
26 flagged against 22 actionable, and it made three of ten severances appear
bridgeable to a single component when no arm could achieve it.

Run:  python test_actionable_bridge.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sixg_sim.simulation import (RADIO_AGENT_TYPES, site_can_steer,
                                 node_hosts_agent, STEER_MODEL)
from sixg_sim.topology import NodeType


class FakeNode:
    def __init__(self, node_type, has_multihaul=True, is_survivor=True):
        self.node_type = node_type
        self.has_multihaul = has_multihaul
        self.is_survivor = is_survivor


FAILED = []


def check(cond, msg):
    if cond:
        print("  PASS  %s" % msg)
    else:
        print("  FAIL  %s" % msg)
        FAILED.append(msg)


print("STEER_MODEL =", STEER_MODEL)
print()

print("1. radio-capable node types with the flag CAN steer")
for t in (NodeType.O_RU, NodeType.O_DU, NodeType.RELAY):
    check(site_can_steer(FakeNode(t)), "%s steers" % t.value)

print()
NONRADIO = (NodeType.UPF, NodeType.AMF, NodeType.SMO,
            NodeType.NEAR_RT_RIC, NodeType.O_CU_CP, NodeType.O_CU_UP)
if STEER_MODEL == 'actionable':
    print("2. the flag alone is NOT enough -- this is the EdgeUPF case")
    for t in NONRADIO:
        check(not site_can_steer(FakeNode(t, has_multihaul=True)),
              "%s flagged has_multihaul does NOT steer" % t.value)
else:
    print("2. legacy mode DOES let the flag alone confer steerability")
    for t in NONRADIO:
        check(site_can_steer(FakeNode(t, has_multihaul=True)),
              "%s steers on the flag alone, as published" % t.value)

print()
print("3. radio hardware without the flag cannot steer either")
for t in (NodeType.O_RU, NodeType.O_DU, NodeType.RELAY):
    check(not site_can_steer(FakeNode(t, has_multihaul=False)),
          "%s without the flag does not steer" % t.value)

print()
print("4. every steerable site also hosts an agent")
print("   (a site that can act must have someone to decide the action)")
for t in RADIO_AGENT_TYPES:
    n = FakeNode(t)
    if site_can_steer(n):
        check(node_hosts_agent(n), "%s steers and hosts an agent" % t.value)

print()
ALL = (NodeType.O_RU, NodeType.O_DU, NodeType.RELAY, NodeType.UPF,
       NodeType.AMF, NodeType.SMO, NodeType.NEAR_RT_RIC,
       NodeType.O_CU_CP, NodeType.O_CU_UP)
if STEER_MODEL == 'actionable':
    print("5. no site can steer without hosting an agent")
    for t in ALL:
        n = FakeNode(t)
        check(not (site_can_steer(n) and not node_hosts_agent(n)),
              "%s: steerable implies agent-hosting" % t.value)
else:
    print("5. legacy mode is EXPECTED to violate steerable-implies-agent")
    orphans = [t.value for t in ALL
               if site_can_steer(FakeNode(t)) and not node_hosts_agent(FakeNode(t))]
    check(bool(orphans),
          "legacy mode produces steerable non-agent sites: %s" % orphans)

print()
print("6. the legacy mode still reproduces the published behaviour")
if STEER_MODEL == 'actionable':
    print("  (set MARL_STEER_MODEL=flagged to exercise this)")
else:
    check(site_can_steer(FakeNode(NodeType.UPF)),
          "flagged mode lets a non-radio node steer, as published")

print()
if FAILED:
    print("%d CHECK(S) FAILED" % len(FAILED))
    sys.exit(1)
print("ALL CHECKS PASSED")
