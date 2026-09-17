"""
Multi-eNB IOPS Controller — ETSI TS 122 346 V16.0.0

Manages Multi-eNB island clusters formed after core network severance:
  - Island detection via connected-components on surviving infra graph
  - Anchor eNB election (hosts local EPC functions)
  - Xn mesh formation for inter-eNB signaling
  - Nomadic eNB (NeNB) deployment by rescue teams
  - UE-to-UE multi-hop routing through surviving RAN
  - Island merge when IAB relay connects separate islands
  - MCPTT group call support within island

Integration:
  Instantiated by Simulator alongside IOPSManager.
  Called each tick to update island state and route UE-to-UE traffic.
"""

import uuid
import random
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class NomadiceNB:
    """A Nomadic eNB deployed by rescue forces (TS 22.346 §5.2)."""
    nenb_id: str
    deployed_at_tick: int
    coverage_area: Optional[str] = None
    connected_to_island: Optional[str] = None
    tx_power_dbm: float = 30.0
    capacity_ues: int = 20
    is_active: bool = True


@dataclass
class LocalEPC:
    """
    Island-local EPC providing authentication and session management
    when the macro core (AMF/UDM/AUSF) is unreachable (TS 22.346 §5.3).

    Hosted on the anchor eNB of each Multi-eNB island.
    """
    island_id: str
    anchor_enb: str
    max_capacity: int = 50       # scales with island size
    _registrations: Dict[str, dict] = field(default_factory=dict)
    _pdu_sessions: Dict[str, dict] = field(default_factory=dict)
    _group_calls: List[dict] = field(default_factory=list)
    _total_admitted: int = 0
    _total_denied: int = 0

    def authenticate_ue(self, ue_id: str, is_emergency: bool,
                        tick: int) -> Optional[str]:
        """Issue IOPS local credential. Returns token or None if full."""
        if ue_id in self._registrations:
            return self._registrations[ue_id]['token']

        if len(self._registrations) >= self.max_capacity:
            if is_emergency:
                # Evict oldest non-emergency
                non_emrg = [uid for uid, r in self._registrations.items()
                            if not r.get('is_emergency', False)]
                if non_emrg:
                    del self._registrations[non_emrg[0]]
                else:
                    self._total_denied += 1
                    return None
            else:
                self._total_denied += 1
                return None

        token = str(uuid.uuid4())[:8]
        self._registrations[ue_id] = {
            'token': token,
            'is_emergency': is_emergency,
            'registered_at': tick,
        }
        self._total_admitted += 1
        return token

    def setup_pdu_session(self, src_ue: str, dst_ue: str,
                          session_type: str = 'ue_to_ue') -> Optional[str]:
        """Create a UE-to-UE PDU session within the island."""
        if src_ue not in self._registrations:
            return None
        if dst_ue not in self._registrations:
            return None
        sid = f"pdu_{src_ue}_{dst_ue}_{len(self._pdu_sessions)}"
        self._pdu_sessions[sid] = {
            'src': src_ue, 'dst': dst_ue,
            'type': session_type, 'active': True,
        }
        return sid

    def multicast_group_call(self, group_ues: List[str],
                             call_type: str = 'mcptt_emergency') -> bool:
        """Initiate MCPTT group call within island (TS 22.179)."""
        registered = [u for u in group_ues if u in self._registrations]
        if len(registered) < 2:
            return False
        self._group_calls.append({
            'participants': registered,
            'type': call_type,
            'active': True,
        })
        return True

    @property
    def registered_count(self) -> int:
        return len(self._registrations)

    @property
    def capacity_fraction(self) -> float:
        return len(self._registrations) / max(1, self.max_capacity)

    def is_registered(self, ue_id: str) -> bool:
        return ue_id in self._registrations

    def revoke_all(self):
        self._registrations.clear()
        self._pdu_sessions.clear()
        self._group_calls.clear()

    def stats(self) -> dict:
        return {
            'admitted': self._total_admitted,
            'denied': self._total_denied,
            'registered': self.registered_count,
            'pdu_sessions': len(self._pdu_sessions),
            'group_calls': len(self._group_calls),
            'capacity_pct': 100.0 * self.capacity_fraction,
        }


@dataclass
class IOPSIsland:
    """
    One Multi-eNB IOPS island cluster (TS 22.346 §5.1).

    Formed by connected surviving infrastructure nodes after core severance.
    """
    island_id: str
    member_enbs: Set[str] = field(default_factory=set)
    local_epc: Optional[LocalEPC] = None
    nenb_nodes: Set[str] = field(default_factory=set)
    registered_ues: Dict[str, str] = field(default_factory=dict)
    # Xn mesh: (enb_a, enb_b) -> capacity
    xn_mesh: Dict[Tuple[str, str], float] = field(default_factory=dict)
    iops_mode: str = 'initiating'   # initiating | operational | terminating
    anchor_enb: Optional[str] = None
    formed_at_tick: int = 0

    @property
    def size(self) -> int:
        return len(self.member_enbs)

    @property
    def total_ue_capacity(self) -> int:
        """Island-wide UE capacity = 50 × num eNBs."""
        return 50 * max(1, self.size)


# ── Controller ────────────────────────────────────────────────────────────────

class MultiENBIOPSController:
    """
    Orchestrates Multi-eNB IOPS lifecycle (TS 22.346).

    Per-tick call sequence:
        1. update_islands(topology, tick)  — detect/update island clusters
        2. process_ue_registrations(...)   — IOPS credential issuance
        3. route_ue_to_ue(src, dst)        — multi-hop routing
    """

    # Node types that participate as "eNBs" in the IOPS island
    ENB_TYPES = {'O-RU', 'O-DU', 'O-CU-CP', 'O-CU-UP', 'O-CU',
                 'gNB-Site', 'GNBSite', 'DU', 'CU', 'Near-RT-RIC',
                 'Relay', 'EdgeUPF'}

    # Core types that, when severed, trigger IOPS
    CORE_TYPES = {'Core', 'SMO', 'Non-RT-RIC', 'AMF', 'UPF'}

    def __init__(self):
        self.islands: Dict[str, IOPSIsland] = {}
        self.nenbs: Dict[str, NomadiceNB] = {}
        self._node_to_island: Dict[str, str] = {}  # node_id -> island_id
        self._routing_cache: Dict[Tuple[str, str], Optional[List[str]]] = {}
        self._last_update_tick: int = -1

    # ── Island detection ─────────────────────────────────────────────────

    def update_islands(self, topology, tick: int) -> List[IOPSIsland]:
        """
        Detect Multi-eNB islands from surviving infrastructure graph.
        Uses NetworkX connected_components on infra-only subgraph.
        """
        if tick == self._last_update_tick:
            return list(self.islands.values())
        self._last_update_tick = tick

        # Build infra-only graph (exclude UEs and down links)
        infra_g = nx.Graph()
        for nid, node in topology.nodes.items():
            ntype = node.node_type.value
            if ntype == 'UE':
                continue
            if ntype in self.CORE_TYPES:
                continue  # core is severed
            if not node.is_survivor:
                continue
            infra_g.add_node(nid)

        for link in topology.links.values():
            if not link.is_up:
                continue
            iface = getattr(link, 'interface_type', None)
            if iface and iface.value == 'Uu':
                continue  # skip air interface
            ep_a, ep_b = link.endpoints
            if ep_a in infra_g and ep_b in infra_g:
                infra_g.add_edge(ep_a, ep_b)

        # Find connected components
        components = list(nx.connected_components(infra_g))

        # Filter to components with >= 2 eNB-type nodes
        new_islands: Dict[str, IOPSIsland] = {}
        self._node_to_island.clear()

        for comp in components:
            enb_nodes = {n for n in comp
                         if topology.nodes[n].node_type.value in self.ENB_TYPES}
            if len(enb_nodes) < 2:
                continue

            # Check if this component matches an existing island
            existing_id = None
            for nid in enb_nodes:
                if nid in self._node_to_island:
                    existing_id = self._node_to_island[nid]
                    break

            if existing_id and existing_id in self.islands:
                island = self.islands[existing_id]
                island.member_enbs = enb_nodes
            else:
                island_id = f"island_{len(new_islands)}_{tick}"
                island = IOPSIsland(
                    island_id=island_id,
                    member_enbs=enb_nodes,
                    formed_at_tick=tick,
                )
                # Elect anchor and create local EPC
                anchor = self._elect_anchor(enb_nodes, topology)
                island.anchor_enb = anchor
                island.local_epc = LocalEPC(
                    island_id=island.island_id,
                    anchor_enb=anchor,
                    max_capacity=island.total_ue_capacity,
                )
                island.iops_mode = 'operational'

            new_islands[island.island_id] = island
            for nid in enb_nodes:
                self._node_to_island[nid] = island.island_id

            # Form Xn mesh
            self._form_xn_mesh(island, infra_g)

        self.islands = new_islands
        self._routing_cache.clear()
        return list(self.islands.values())

    def _elect_anchor(self, enb_nodes: Set[str], topology) -> str:
        """Elect anchor eNB: highest degree + energy in infra graph."""
        best = None
        best_score = -1.0
        for nid in enb_nodes:
            node = topology.nodes.get(nid)
            if not node:
                continue
            degree = len(topology.get_neighbors(nid))
            energy = getattr(node, 'energy_soc', 0.5)
            score = degree * 0.6 + energy * 0.4
            if score > best_score:
                best_score = score
                best = nid
        return best or next(iter(enb_nodes))

    def _form_xn_mesh(self, island: IOPSIsland, infra_g: nx.Graph):
        """Create logical Xn links between all eNBs in the island."""
        island.xn_mesh.clear()
        enbs = list(island.member_enbs)
        for i, a in enumerate(enbs):
            for b in enbs[i+1:]:
                if infra_g.has_edge(a, b) or nx.has_path(infra_g, a, b):
                    try:
                        path_len = nx.shortest_path_length(infra_g, a, b)
                        cap = max(10.0, 1000.0 / max(1, path_len))
                    except nx.NetworkXError:
                        cap = 10.0
                    island.xn_mesh[(a, b)] = cap

    # ── UE registration ──────────────────────────────────────────────────

    def get_island_for_node(self, node_id: str) -> Optional[IOPSIsland]:
        """Return the island a node belongs to."""
        iid = self._node_to_island.get(node_id)
        return self.islands.get(iid) if iid else None

    def get_island_for_ue(self, ue_id: str, topology) -> Optional[IOPSIsland]:
        """Find which island a UE can register to (via its connected O-RU)."""
        for link in topology.links.values():
            if not link.is_up:
                continue
            iface = getattr(link, 'interface_type', None)
            if not iface or iface.value != 'Uu':
                continue
            if ue_id not in link.endpoints:
                continue
            other = link.endpoints[0] if link.endpoints[1] == ue_id else link.endpoints[1]
            island = self.get_island_for_node(other)
            if island:
                return island
        return None

    def admit_ue(self, ue_id: str, is_emergency: bool,
                 tick: int, topology) -> bool:
        """Admit UE to its island's local EPC."""
        island = self.get_island_for_ue(ue_id, topology)
        if not island or not island.local_epc:
            return False
        token = island.local_epc.authenticate_ue(ue_id, is_emergency, tick)
        if token:
            island.registered_ues[ue_id] = token
            return True
        return False

    # ── UE-to-UE routing ─────────────────────────────────────────────────

    def route_ue_to_ue(self, src_ue: str, dst_ue: str,
                       topology) -> Optional[List[str]]:
        """
        Multi-hop UE-to-UE routing through surviving Multi-eNB RAN.

        Path: src_UE → O-RU → [O-DU → ... → O-DU] → O-RU → dst_UE
        Uses Xn mesh for inter-eNB hops within the same island.
        """
        cache_key = (src_ue, dst_ue)
        if cache_key in self._routing_cache:
            return self._routing_cache[cache_key]

        # Find O-RUs for each UE
        src_orus = self._get_ue_orus(src_ue, topology)
        dst_orus = self._get_ue_orus(dst_ue, topology)

        if not src_orus or not dst_orus:
            self._routing_cache[cache_key] = None
            return None

        # Check same island
        for s_oru in src_orus:
            s_island = self.get_island_for_node(s_oru)
            if not s_island:
                continue
            for d_oru in dst_orus:
                d_island = self.get_island_for_node(d_oru)
                if d_island and d_island.island_id == s_island.island_id:
                    # Same island — find path through Xn mesh
                    path = self._find_intra_island_path(
                        s_oru, d_oru, s_island, topology)
                    if path:
                        full_path = [src_ue] + path + [dst_ue]
                        self._routing_cache[cache_key] = full_path
                        return full_path

        self._routing_cache[cache_key] = None
        return None

    def _get_ue_orus(self, ue_id: str, topology) -> List[str]:
        """Get O-RUs connected to a UE via Uu links."""
        orus = []
        for link in topology.links.values():
            if not link.is_up:
                continue
            iface = getattr(link, 'interface_type', None)
            if not iface or iface.value != 'Uu':
                continue
            if ue_id not in link.endpoints:
                continue
            other = link.endpoints[0] if link.endpoints[1] == ue_id \
                else link.endpoints[1]
            node = topology.nodes.get(other)
            if node and node.is_survivor:
                orus.append(other)
        return orus

    def _find_intra_island_path(self, src_enb: str, dst_enb: str,
                                island: IOPSIsland,
                                topology) -> Optional[List[str]]:
        """Find shortest path between two eNBs within an island."""
        if src_enb == dst_enb:
            return [src_enb]

        # Build island subgraph from Xn mesh
        g = nx.Graph()
        for (a, b), cap in island.xn_mesh.items():
            g.add_edge(a, b, weight=1.0 / max(cap, 0.1))

        # Also add direct infra edges
        for nid in island.member_enbs:
            for neighbor in topology.get_neighbors(nid):
                if neighbor in island.member_enbs:
                    if not g.has_edge(nid, neighbor):
                        g.add_edge(nid, neighbor, weight=1.0)

        try:
            path = nx.shortest_path(g, src_enb, dst_enb, weight='weight')
            return path
        except (nx.NetworkXError, nx.NodeNotFound):
            return None

    # ── Nomadic eNB ──────────────────────────────────────────────────────

    def deploy_nenb(self, topology, tick: int,
                    coverage_area: Optional[str] = None,
                    connect_to: Optional[List[str]] = None) -> Optional[str]:
        """Deploy a Nomadic eNB and add it to the nearest island."""
        from ..topology import Node, NodeType, Link, LinkType, InterfaceType

        nenb_id = f"NeNB_{len(self.nenbs)}_{tick}"
        nenb = NomadiceNB(
            nenb_id=nenb_id,
            deployed_at_tick=tick,
            coverage_area=coverage_area,
        )

        # Add to topology as O-RU type
        node = Node(
            id=nenb_id,
            node_type=NodeType.O_RU,
            initial_energy=1.0,
            coverage_area=coverage_area,
        )
        topology.add_node(node)

        # Connect to nearby surviving O-RUs/O-DUs
        if connect_to is None:
            candidates = [
                nid for nid, n in topology.nodes.items()
                if n.node_type.value in ('O-RU', 'O-DU', 'Relay')
                and n.is_survivor and nid != nenb_id
            ]
            if coverage_area:
                area_cands = [c for c in candidates
                              if topology.nodes[c].coverage_area == coverage_area]
                if area_cands:
                    candidates = area_cands
            connect_to = random.sample(candidates,
                                       min(3, len(candidates)))

        link_counter = len(topology.links) + 1
        for target in connect_to:
            lid = f"IAB_NeNB_{nenb_id}_{target}_{link_counter}"
            link_counter += 1
            topology.add_link(Link(
                id=lid,
                endpoints=(nenb_id, target),
                capacity=random.choice([200, 400, 600]),
                latency=random.randint(1, 3),
                link_type=LinkType.TRANSPORT_RELAY,
                interface_type=InterfaceType.BACKHAUL,
                is_up=True,
            ))

        topology.invalidate_infrastructure_cache()
        self.nenbs[nenb_id] = nenb

        # Join nearest island
        island = self.get_island_for_node(
            connect_to[0] if connect_to else '')
        if island:
            island.member_enbs.add(nenb_id)
            island.nenb_nodes.add(nenb_id)
            island.local_epc.max_capacity = island.total_ue_capacity
            nenb.connected_to_island = island.island_id
            self._node_to_island[nenb_id] = island.island_id

        return nenb_id

    # ── Island merge ─────────────────────────────────────────────────────

    def try_merge_islands(self, island_a_id: str,
                          island_b_id: str) -> Optional[IOPSIsland]:
        """Merge two islands when IAB relay connects them."""
        a = self.islands.get(island_a_id)
        b = self.islands.get(island_b_id)
        if not a or not b or a.island_id == b.island_id:
            return None

        # Merge into island_a
        a.member_enbs |= b.member_enbs
        a.nenb_nodes |= b.nenb_nodes
        a.registered_ues.update(b.registered_ues)
        a.xn_mesh.update(b.xn_mesh)
        a.local_epc.max_capacity = a.total_ue_capacity

        # Update mappings
        for nid in b.member_enbs:
            self._node_to_island[nid] = a.island_id

        del self.islands[island_b_id]
        self._routing_cache.clear()
        return a

    # ── IOPS termination ─────────────────────────────────────────────────

    def terminate_iops(self, island_id: str):
        """Graceful IOPS teardown when core reconnects."""
        island = self.islands.get(island_id)
        if not island:
            return
        island.iops_mode = 'terminating'
        if island.local_epc:
            island.local_epc.revoke_all()
        island.registered_ues.clear()
        island.xn_mesh.clear()

    def terminate_all(self):
        """Terminate all IOPS islands."""
        for iid in list(self.islands.keys()):
            self.terminate_iops(iid)
        self.islands.clear()
        self._node_to_island.clear()
        self._routing_cache.clear()

    # ── Observation helpers ──────────────────────────────────────────────

    def get_node_obs(self, node_id: str, topology) -> dict:
        """
        Return 8 normalised observation features for a node's IOPS state.
        Used to populate the extended ConnectivityState block.
        """
        island = self.get_island_for_node(node_id)
        if not island:
            return {
                'island_member_count_norm': 0.0,
                'island_ue_load_balance': 0.5,
                'xn_mesh_density': 0.0,
                'local_epc_health': 0.0,
                'nenb_count_norm': 0.0,
                'peer_avg_reward': 0.0,
                'peer_best_relay_hint': 0.0,
                'multi_island_bridge': 0.0,
            }

        n_enbs = island.size
        max_possible = 80  # max O-RUs in topology
        member_norm = min(1.0, n_enbs / max(1, max_possible))

        # UE load balance: my UEs / island avg
        my_ues = 0
        for link in topology.links.values():
            if not link.is_up:
                continue
            iface = getattr(link, 'interface_type', None)
            if not iface or iface.value != 'Uu':
                continue
            if node_id in link.endpoints:
                my_ues += 1
        avg_ues = max(1, len(island.registered_ues) / max(1, n_enbs))
        load_bal = min(2.0, my_ues / max(1.0, avg_ues)) / 2.0

        # Xn mesh density
        max_xn = n_enbs * (n_enbs - 1) / 2
        xn_density = len(island.xn_mesh) / max(1, max_xn)

        # Local EPC health
        epc_health = 1.0 if island.local_epc and \
            island.local_epc.capacity_fraction < 0.9 else 0.5

        # NeNB count
        nenb_norm = min(1.0, len(island.nenb_nodes) / 5.0)

        # Bridge detection: does this node connect to nodes in
        # another island?
        is_bridge = 0.0
        for neighbor in topology.get_neighbors(node_id):
            n_island = self.get_island_for_node(neighbor)
            if n_island and n_island.island_id != island.island_id:
                is_bridge = 1.0
                break

        return {
            'island_member_count_norm': member_norm,
            'island_ue_load_balance': load_bal,
            'xn_mesh_density': min(1.0, xn_density),
            'local_epc_health': epc_health,
            'nenb_count_norm': nenb_norm,
            'peer_avg_reward': 0.0,     # filled by LearningPostcardExchanger
            'peer_best_relay_hint': 0.0,
            'multi_island_bridge': is_bridge,
        }

    def stats(self) -> dict:
        """Global IOPS stats for logging."""
        return {
            'num_islands': len(self.islands),
            'total_enbs': sum(i.size for i in self.islands.values()),
            'total_nenbs': len(self.nenbs),
            'total_registered_ues': sum(
                len(i.registered_ues) for i in self.islands.values()),
            'islands': {
                iid: {
                    'size': i.size,
                    'mode': i.iops_mode,
                    'anchor': i.anchor_enb,
                    'ues': len(i.registered_ues),
                    'xn_links': len(i.xn_mesh),
                    'nenbs': len(i.nenb_nodes),
                    'members': list(i.member_enbs),
                    'xn_density': len(i.xn_mesh) / max(1, i.size * (i.size - 1) / 2),
                    'epc': i.local_epc.stats() if i.local_epc else {},
                }
                for iid, i in self.islands.items()
            },
        }

    def route_ue_to_ue(self, source_ue: str, target_ue: str,
                       topology) -> 'Optional[List[str]]':
        """Find a multi-hop path through IOPS island infrastructure.

        Returns list of node IDs forming the path, or None if no path exists.
        Uses the Xn mesh and NeNB links within each island.
        """
        import networkx as nx

        # Find which island(s) are connected to each UE via their anchor eNBs
        def _ue_anchors(ue_id):
            """Return infra nodes directly connected to a UE."""
            anchors = []
            for link in topology.links.values():
                if not getattr(link, 'is_up', True):
                    continue
                ep0, ep1 = link.endpoints
                if ep0 == ue_id:
                    n = topology.nodes.get(ep1)
                    if n and n.node_type.value != 'UE':
                        anchors.append(ep1)
                elif ep1 == ue_id:
                    n = topology.nodes.get(ep0)
                    if n and n.node_type.value != 'UE':
                        anchors.append(ep0)
            return anchors

        src_anchors = _ue_anchors(source_ue)
        tgt_anchors = _ue_anchors(target_ue)
        if not src_anchors or not tgt_anchors:
            return None

        # Check if any src anchor can reach any tgt anchor through
        # the live topology graph (includes Xn mesh + NeNB links)
        try:
            for sa in src_anchors:
                for ta in tgt_anchors:
                    if topology.has_infrastructure_path(sa, ta):
                        path = nx.shortest_path(topology.graph,
                                                source_ue, target_ue)
                        return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return None
