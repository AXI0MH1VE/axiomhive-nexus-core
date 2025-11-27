"""Axiom Hive Nexus Core - Deterministic AI with Cryptographic Proof."""

__version__ = "1.1.0"
__author__ = "Alexis M. Adams"
__email__ = "devdollzai@gmail.com"
__url__ = "https://axiomhive.org"

from .axiomhive_nexus import AxiomHiveNexus
from .deoxys_fhe import DeoxysFHE
from .elc_core import ELCCore
from .ailock_proxy import AILockProxy
from .hybrid_ssm import HybridStateSpaceModel
from .bitcoin_lightning import BitcoinLightningNode

__all__ = [
    "AxiomHiveNexus",
    "DeoxysFHE",
    "ELCCore",
    "AILockProxy",
    "HybridStateSpaceModel",
    "BitcoinLightningNode",
]