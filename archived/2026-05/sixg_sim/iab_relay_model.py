"""
Backward-compatibility shim — redirects to transport_relay_model.py

This module previously contained IAB (Integrated Access and Backhaul)
terminology. All functionality has been moved to transport_relay_model.py
with corrected naming that reflects Ceragon's transport-layer role.
"""
from .transport_relay_model import (        # noqa: F401
    TransportRelayLink as IABLink,
    TransportRelayModel as IABRelayModel,
)
