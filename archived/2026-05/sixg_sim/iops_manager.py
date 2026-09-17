"""
IOPS Manager — Isolated E-UTRAN Operation for Public Safety
3GPP TS 23.380 / TS 22.346

In normal operation:
    New UE → PRACH → RRC Setup → AMF registration (AUSF/UDM auth) → PDU session

After core severance (AMF/AUSF/UDM offline):
    New UE → PRACH → O-CU-CP issues LOCAL credential (IOPS token)
           → UE can communicate within the island without core auth
           → Token valid until island mode ends or UE detaches

This module manages the island-local credential store and controls capacity.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Maximum UEs that can be locally registered per island
DEFAULT_MAX_CAPACITY   = 50
MAX_PENDING_QUEUE_SIZE = 20


@dataclass
class IOPSRegistration:
    """One locally-registered UE in island mode."""
    ue_id:              str
    is_emergency:       bool
    registered_at_tick: int
    local_token:        str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    can_communicate:    bool = True    # set False if island re-fragments


@dataclass
class IOPSRequest:
    """Pending UE registration request waiting for agent decision."""
    ue_id:         str
    is_emergency:  bool
    request_tick:  int


class IOPSManager:
    """
    Manages local UE credentials in island (core-severed) mode.

    Agents interact through:
        request_registration(ue_id, is_emergency, tick) — called by sim when new UE arrives
        admit(ue_id, is_emergency, tick)                — called when MARL agent decides ADMIT
        deny(ue_id)                                     — called when MARL agent decides DENY
        flush_pending()                                 — returns pending list at each tick

    Observation features (two normalised floats, [0, 1]):
        capacity_fraction()   — how full the local credential store is
        pending_norm()        — how many UEs are waiting / max_pending
    """

    def __init__(self, max_capacity: int = DEFAULT_MAX_CAPACITY):
        self.max_capacity   = max_capacity
        self.registrations: Dict[str, IOPSRegistration] = {}
        self.pending:       List[IOPSRequest]            = []
        self._total_admitted   = 0
        self._total_denied     = 0
        self._total_emergency_admitted = 0

    # ── Registration lifecycle ─────────────────────────────────────────────────

    def request_registration(self, ue_id: str, is_emergency: bool, tick: int):
        """Queue an IOPS registration request (UE arrived, AMF unavailable)."""
        if ue_id in self.registrations:
            return  # Already registered
        # Avoid duplicate pending
        if any(r.ue_id == ue_id for r in self.pending):
            return
        if len(self.pending) < MAX_PENDING_QUEUE_SIZE:
            self.pending.append(IOPSRequest(ue_id, is_emergency, tick))

    def admit(self, ue_id: str, is_emergency: bool, tick: int) -> bool:
        """
        Immediately admit a UE with an island-local credential.
        Emergency UEs may evict oldest non-emergency if at capacity.
        Returns True if admitted.
        """
        if ue_id in self.registrations:
            return True  # already registered

        if len(self.registrations) >= self.max_capacity:
            if not is_emergency:
                self._total_denied += 1
                return False
            # Emergency: evict oldest non-emergency UE to make room
            non_emrg = [uid for uid, r in self.registrations.items()
                        if not r.is_emergency]
            if non_emrg:
                del self.registrations[non_emrg[0]]
            else:
                # Even emergency slots are full — deny
                self._total_denied += 1
                return False

        self.registrations[ue_id] = IOPSRegistration(
            ue_id            = ue_id,
            is_emergency     = is_emergency,
            registered_at_tick = tick,
        )
        self._total_admitted += 1
        if is_emergency:
            self._total_emergency_admitted += 1
        # Remove from pending if present
        self.pending = [r for r in self.pending if r.ue_id != ue_id]
        return True

    def deny(self, ue_id: str):
        """Record a DENY decision and remove from pending queue."""
        self.pending = [r for r in self.pending if r.ue_id != ue_id]
        self._total_denied += 1

    def revoke(self, ue_id: str):
        """Revoke a UE's island credential (e.g. detachment or island split)."""
        self.registrations.pop(ue_id, None)

    def revoke_all(self):
        """Clear all credentials (core reconnected)."""
        self.registrations.clear()

    # ── State queries ──────────────────────────────────────────────────────────

    @property
    def total_admitted(self) -> int:
        """Public alias for _total_admitted (used by dashboard / main loop)."""
        return self._total_admitted

    @property
    def total_denied(self) -> int:
        """Public alias for _total_denied."""
        return self._total_denied


    def is_registered(self, ue_id: str) -> bool:
        return ue_id in self.registrations

    def registered_count(self) -> int:
        return len(self.registrations)

    def emergency_count(self) -> int:
        return sum(1 for r in self.registrations.values() if r.is_emergency)

    def pending_count(self) -> int:
        return len(self.pending)

    def capacity_fraction(self) -> float:
        """[0, 1] — fraction of local credential slots in use."""
        return len(self.registrations) / max(1, self.max_capacity)

    def pending_norm(self) -> float:
        """[0, 1] — normalised pending queue depth."""
        return min(1.0, len(self.pending) / MAX_PENDING_QUEUE_SIZE)

    def get_obs_features(self) -> Tuple[float, float]:
        """Return (capacity_fraction, pending_norm) for agent observation."""
        return self.capacity_fraction(), self.pending_norm()

    # ── Tick interface ─────────────────────────────────────────────────────────

    def flush_pending(self) -> List[IOPSRequest]:
        """
        Return all pending registration requests for this tick
        (without clearing — only cleared by admit/deny).
        """
        return list(self.pending)

    def get_next_pending(self) -> Optional[IOPSRequest]:
        """Return highest-priority pending request (emergency first)."""
        emrg = [r for r in self.pending if r.is_emergency]
        if emrg:
            return emrg[0]
        return self.pending[0] if self.pending else None

    def expire_old_pending(self, current_tick: int, max_wait: int = 50):
        """Drop requests that have been waiting too long (timed out)."""
        self.pending = [
            r for r in self.pending
            if (current_tick - r.request_tick) <= max_wait or r.is_emergency
        ]

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'admitted':            self._total_admitted,
            'denied':              self._total_denied,
            'emergency_admitted':  self._total_emergency_admitted,
            'currently_registered': len(self.registrations),
            'capacity_pct':        100.0 * self.capacity_fraction(),
        }
