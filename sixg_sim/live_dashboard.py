"""
Live Dashboard Server — Flask-SocketIO
======================================
Runs in a background daemon thread.  The simulation main loop calls:
    server.push_state(snap)     — every tick
    server.push_topology(topo)  — once at startup
    server.push_episode_end(ep) — end of each episode

Clients connect to http://localhost:5050 and receive real-time updates.
"""

import threading
import webbrowser
import time
import os
from typing import Optional

from flask import Flask, send_from_directory
from flask_socketio import SocketIO

_app  = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
_sio  = SocketIO(_app, cors_allowed_origins="*", async_mode="threading", logger=False, engineio_logger=False)
_PORT = 5050
_thread: Optional[threading.Thread] = None
_ready  = threading.Event()


# ── Routes ────────────────────────────────────────────────────────────────────

@_app.route("/")
def index():
    return send_from_directory(_app.static_folder, "dashboard.html")

@_app.route("/internal_push", methods=["POST"])
def internal_push():
    from flask import request
    import pickle
    try:
        snap = pickle.loads(request.data)
        push_state(snap)
    except Exception:
        pass
    return {"status": "ok"}, 200


_last_topology_data = None   # cached for reconnect


@_sio.on("connect")
def on_connect():
    """Push cached topology to a newly connected client."""
    if _last_topology_data is not None:
        # emit() with no room/sid broadcasts to current request's socket
        from flask_socketio import emit as _emit
        _emit("topology", _last_topology_data)


@_sio.on("request_topology")
def on_request_topology():
    """Client explicitly asks for topology re-push (e.g. after page refresh)."""
    if _last_topology_data is not None:
        from flask_socketio import emit as _emit
        _emit("topology", _last_topology_data)


# ── Public API (called from simulation thread) ────────────────────────────────

def push_topology(topology):
    """
    Send the full topology (nodes + links) once so the dashboard can
    build the network graph layout.

    topology: object with .nodes (dict id→node) and .links (dict id→link)
    """
    nodes_data = []
    for nid, node in topology.nodes.items():
        nt = getattr(node, "node_type", None)
        nt_str = nt.value if hasattr(nt, 'value') else str(nt)
        nodes_data.append({
            "id":   nid,
            "type": nt_str,
            "is_rescue_service": getattr(node, "is_rescue_service", False),
            "coverage_area": getattr(node, "coverage_area", ""),
            "zone": getattr(node, "zone", None) or getattr(node, "coverage_area", ""),
            "has_multihaul": getattr(node, "has_multihaul", False),
            "x_pos": getattr(node, "x_pos", 0.0),
            "y_pos": getattr(node, "y_pos", 0.0),
        })

    links_data = []
    for lid, link in topology.links.items():
        # Try both .endpoints tuple and .source/.dest attributes
        ep = getattr(link, "endpoints", None)
        if ep and len(ep) >= 2:
            src, dst = ep[0], ep[1]
        else:
            src = getattr(link, "source", None)
            dst = getattr(link, "dest",   None)
        if src and dst:
            lt = getattr(link, "link_type", "")
            lt_str = lt.value if hasattr(lt, 'value') else str(lt)
            links_data.append({
                "id":   lid,
                "a":    src,
                "b":    dst,
                "type": lt_str,
                "capacity": getattr(link, "capacity", 0),
            })

    _sio.emit("topology", {"nodes": nodes_data, "links": links_data})
    global _last_topology_data
    _last_topology_data = {"nodes": nodes_data, "links": links_data}


def push_state(snap):
    """
    Push a TickSnapshot to all connected browser clients.
    Called every tick (or every N ticks) from the simulation loop.
    """
    import math

    # Build per-node state dict
    node_states = {}
    relay_paths = []  # active re-routing paths for visualization
    for nid, rm in snap.node_relay_modes.items():
        node_states[nid] = {
            "relay_mode": rm,
            "tx_power":   snap.node_tx_power.get(nid),
            "island":     nid in snap.node_is_island,
            "iops_reg":   nid in snap.node_iops_reg,
            "severed":    False,
        }
        # Capture active relay re-routing paths
        rm_upper = rm.upper() if rm else ''
        if 'REROUTE' in rm_upper or 'BOOST' in rm_upper:
            peer = snap.node_relay_peers.get(nid)
            if peer:
                relay_paths.append({
                    "from": nid,
                    "to": peer,
                    "mode": 'boost' if 'BOOST' in rm_upper else 'reroute',
                    "capacity": snap.node_relay_capacity.get(nid, 0),
                })
    for nid in snap.node_is_island:
        if nid not in node_states:
            node_states[nid] = {"relay_mode": "", "island": True, "severed": False, "iops_reg": False}
    # Mark nodes in severed links as severed
    for pair in snap.severed_links:
        for nid in pair:
            if nid in node_states:
                node_states[nid]["severed"] = True

    transport_relay_count = sum(
        1 for rm in snap.node_relay_modes.values()
        if any(k in (rm or '').upper() for k in ('REROUTE', 'BOOST', 'TRANSPORT'))
    )

    payload = {
        "tick":              snap.tick,
        "episode":           snap.episode,
        "phase":             snap.phase,
        "island_mode":       snap.island_mode,
        "ue_pairs_routed":   snap.ue_pairs_routed,
        "reachable_frac":    snap.reachable_frac,
        "iops_admitted":     snap.iops_admitted,
        "policy_loss":       snap.policy_loss if not math.isnan(snap.policy_loss) else None,
        "value_loss":        snap.value_loss  if not math.isnan(snap.value_loss)  else None,
        "entropy":           snap.entropy     if not math.isnan(snap.entropy)     else None,
        "episode_reward":    snap.episode_reward,
        "global_policy":     snap.global_policy,
        "severed_links":     snap.severed_links,
        "transport_links":         snap.transport_links,
        "relay_paths":       relay_paths,
        "postcard_senders":  snap.postcard_senders,
        "postcard_receivers":snap.postcard_receivers,
        "node_states":       node_states,
        "transport_relay_count":   transport_relay_count,
        # Multi-eNB IOPS (ETSI TS 22.346)
        "iops_island_count":   getattr(snap, 'iops_island_count', 0),
        "iops_nenb_count":     getattr(snap, 'iops_nenb_count', 0),
        "iops_peer_exchanges": getattr(snap, 'iops_peer_exchanges', 0),
        "iops_xn_density":     getattr(snap, 'iops_xn_density', 0.0),
        "node_island_ids":     getattr(snap, 'node_island_ids', {}),
        "scenario_type":       getattr(snap, 'scenario_type', ''),
        "learning_postcard_pairs": getattr(snap, 'learning_postcard_pairs', []),
        "optimal_paths":       getattr(snap, 'optimal_paths', []),
        "optimal_actions":     getattr(snap, 'optimal_actions', []),
    }
    _sio.emit("state_update", payload)


def push_episode_end(episode: int, avg_reward: float, connectivity_rate: float):
    _sio.emit("episode_end", {
        "episode":           episode,
        "avg_reward":        avg_reward,
        "connectivity_rate": connectivity_rate,
    })


def push_training_progress(episode: int, total_episodes: int,
                           scenario_type: str, metrics: dict):
    """Push training progress for dashboard progress bars and learning state."""
    import time
    _sio.emit("training_progress", {
        "episode":          episode,
        "total_episodes":   total_episodes,
        "progress_pct":     round(100.0 * episode / max(1, total_episodes), 1),
        "scenario_type":    scenario_type,
        "timestamp":        time.time(),
        # Learning metrics
        "policy_loss":      metrics.get("policy_loss"),
        "critic_loss":      metrics.get("critic_loss"),
        "entropy":          metrics.get("entropy"),
        "avg_reward":       metrics.get("avg_reward"),
        "ue_connectivity":  metrics.get("ue_connectivity"),
        "relay_count":      metrics.get("relay_count"),
        "peak_relays":      metrics.get("peak_relays"),
        "lr_actor":         metrics.get("lr_actor"),
        "lr_critic":        metrics.get("lr_critic"),
        # Convergence
        "reward_trend":     metrics.get("reward_trend"),  # slope of last 5 ep rewards
        # Optimal connectivity
        "optimal_frac":     metrics.get("optimal_frac"),
        "efficiency":       metrics.get("efficiency"),
        "n_components":     metrics.get("n_components"),
    })


# ── Server lifecycle ──────────────────────────────────────────────────────────

def _run_server():
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    _ready.set()
    _sio.run(_app, host="0.0.0.0", port=_PORT, allow_unsafe_werkzeug=True)


def start(open_browser: bool = True):
    """Start the dashboard server in a background thread."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return   # already running

    _thread = threading.Thread(target=_run_server, daemon=True)
    _thread.start()
    _ready.wait(timeout=5)
    time.sleep(0.5)   # let the socket bind

    url = f"http://localhost:{_PORT}"
    print(f"\n[Dashboard] Live dashboard -> {url}")
    if open_browser:
        webbrowser.open(url)


def stop():
    """No-op — server exits automatically when main process ends (daemon thread)."""
    pass
