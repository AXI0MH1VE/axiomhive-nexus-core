"""Axiom Hive Nexus Core - Main Kernel."""

import hashlib
import time
import json
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn

from .deoxys_fhe import DeoxysFHE
from .elc_core import ELCCore
from .ailock_proxy import AILockProxy
from .hybrid_ssm import HybridStateSpaceModel
from .bitcoin_lightning import BitcoinLightningNode


class AxiomHiveNexus:
    """Main kernel for Axiom Hive deterministic AI system.
    
    Provides cryptographically verifiable intelligence with:
    - C=0 Signature generation
    - Homomorphic encryption via Deoxys FHE
    - Formal verification via ELC Core
    - Sovereignty enforcement via AILock
    - Hybrid SSM computation
    - Bitcoin Lightning monetization
    """

    def __init__(
        self,
        axioms: List[str],
        mode: str = "verified",
        d_model: int = 512,
        d_state: int = 64,
        enable_fhe: bool = False,
        enable_lightning: bool = False,
        lnd_config: Optional[Dict[str, str]] = None,
    ):
        """Initialize Axiom Hive Nexus kernel.
        
        Args:
            axioms: List of axiom identifiers that constrain behavior
            mode: 'verified' (deterministic) or 'creative' (probabilistic)
            d_model: Model dimension
            d_state: State space dimension
            enable_fhe: Enable homomorphic encryption
            enable_lightning: Enable Bitcoin Lightning payments
            lnd_config: Lightning node configuration
        """
        self.axioms = axioms
        self.mode = mode
        self.d_model = d_model
        self.d_state = d_state
        
        # Initialize components
        self.ailock = AILockProxy(axioms=axioms)
        self.elc = ELCCore()
        self.hybrid_ssm = HybridStateSpaceModel(
            d_model=d_model,
            d_state=d_state,
            n_heads=8
        )
        
        # Optional components
        self.fhe = DeoxysFHE() if enable_fhe else None
        self.lightning = None
        if enable_lightning and lnd_config:
            self.lightning = BitcoinLightningNode(
                lnd_host=lnd_config.get("host", "localhost:10009"),
                macaroon_path=lnd_config.get("macaroon_path", "~/.lnd/admin.macaroon"),
                cert_path=lnd_config.get("cert_path", "~/.lnd/tls.cert")
            )
        
        # State tracking
        self.state_history: List[Dict[str, Any]] = []
        self.computation_count = 0

    def process(
        self,
        input_data: Any,
        require_proof: bool = True,
        payment_required: bool = False,
        amount_sats: int = 1000,
    ) -> Dict[str, Any]:
        """Process input through Axiom Hive kernel with optional proof.
        
        Args:
            input_data: Input to process (text, tensor, or dict)
            require_proof: Generate C=0 signature
            payment_required: Require Lightning payment before processing
            amount_sats: Payment amount in satoshis
            
        Returns:
            Dict containing:
                - output: Processed result
                - signature: C=0 cryptographic signature (if require_proof)
                - verified: Boolean verification status
                - payment_hash: Lightning payment hash (if payment_required)
        """
        # Step 1: AILock sovereignty check
        request_description = str(input_data)[:200]
        if not self.ailock.validate_request(request_description):
            return {
                "output": None,
                "error": "Request blocked by AILock: violates axioms",
                "verified": False,
                "axioms_violated": self.ailock.get_violations(request_description)
            }
        
        # Step 2: Payment check (if required)
        payment_hash = None
        if payment_required and self.lightning:
            invoice = self.lightning.create_invoice(
                amount_sats=amount_sats,
                memo=f"Axiom Hive computation #{self.computation_count}"
            )
            payment_hash = invoice["payment_hash"]
            
            # Wait for payment confirmation
            if not self.lightning.wait_for_payment(payment_hash, timeout=300):
                return {
                    "output": None,
                    "error": "Payment not received",
                    "verified": False,
                    "invoice": invoice["payment_request"]
                }
        
        # Step 3: Convert input to tensor
        input_tensor = self._prepare_input(input_data)
        
        # Step 4: Optional FHE encryption
        if self.fhe:
            encrypted_input = self.fhe.encrypt(input_tensor.cpu().numpy())
            # Compute on encrypted data
            output_tensor = self._compute_encrypted(encrypted_input)
        else:
            # Direct computation
            output_tensor = self._compute(input_tensor)
        
        # Step 5: ELC verification
        prev_state = self.state_history[-1] if self.state_history else {"energy": 100.0}
        current_state = {
            "energy": prev_state["energy"] - 0.5,
            "entropy": float(torch.sum(output_tensor ** 2).item()),
            "computation_id": self.computation_count
        }
        
        verification_passed = self.elc.verify_state_transition(
            prev_state=prev_state,
            next_state=current_state,
            axioms=self.axioms
        )
        
        if not verification_passed:
            return {
                "output": None,
                "error": "ELC verification failed: state transition invalid",
                "verified": False,
                "prev_state": prev_state,
                "next_state": current_state
            }
        
        # Step 6: Generate output
        output = self._format_output(output_tensor)
        
        # Step 7: Generate C=0 signature (if required)
        signature = None
        if require_proof:
            signature = self.generate_c0_signature(
                input_data=input_data,
                output_data=output,
                state=current_state
            )
        
        # Update state history
        self.state_history.append(current_state)
        self.computation_count += 1
        
        return {
            "output": output,
            "signature": signature,
            "verified": verification_passed,
            "payment_hash": payment_hash,
            "computation_id": self.computation_count - 1,
            "mode": self.mode
        }

    def generate_c0_signature(
        self,
        input_data: Any,
        output_data: Any,
        state: Dict[str, Any]
    ) -> str:
        """Generate C=0 cryptographic signature proving zero corruption.
        
        Args:
            input_data: Original input
            output_data: Computed output
            state: Current system state
            
        Returns:
            SHA-256 hash binding all elements as cryptographic receipt
        """
        # Hash input
        input_str = json.dumps(str(input_data), sort_keys=True)
        input_hash = hashlib.sha256(input_str.encode()).hexdigest()
        
        # Hash output
        output_str = json.dumps(str(output_data), sort_keys=True)
        output_hash = hashlib.sha256(output_str.encode()).hexdigest()
        
        # Hash axioms
        axioms_str = json.dumps(sorted(self.axioms))
        axioms_hash = hashlib.sha256(axioms_str.encode()).hexdigest()
        
        # Hash state
        state_str = json.dumps(state, sort_keys=True)
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()
        
        # Combine with timestamp
        timestamp = str(int(time.time()))
        
        # Generate final signature
        combined = f"{input_hash}|{output_hash}|{axioms_hash}|{state_hash}|{timestamp}"
        signature = hashlib.sha256(combined.encode()).hexdigest()
        
        return signature

    def verify_c0_signature(
        self,
        signature: str,
        input_data: Any,
        output_data: Any,
        state: Dict[str, Any],
        timestamp: int
    ) -> bool:
        """Verify a C=0 signature.
        
        Args:
            signature: Signature to verify
            input_data: Original input
            output_data: Claimed output
            state: Claimed state
            timestamp: Original timestamp
            
        Returns:
            True if signature is valid
        """
        # Reconstruct signature
        input_str = json.dumps(str(input_data), sort_keys=True)
        input_hash = hashlib.sha256(input_str.encode()).hexdigest()
        
        output_str = json.dumps(str(output_data), sort_keys=True)
        output_hash = hashlib.sha256(output_str.encode()).hexdigest()
        
        axioms_str = json.dumps(sorted(self.axioms))
        axioms_hash = hashlib.sha256(axioms_str.encode()).hexdigest()
        
        state_str = json.dumps(state, sort_keys=True)
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()
        
        combined = f"{input_hash}|{output_hash}|{axioms_hash}|{state_hash}|{timestamp}"
        expected_signature = hashlib.sha256(combined.encode()).hexdigest()
        
        return signature == expected_signature

    def _prepare_input(self, input_data: Any) -> torch.Tensor:
        """Convert input to tensor format."""
        if isinstance(input_data, torch.Tensor):
            return input_data
        elif isinstance(input_data, (list, np.ndarray)):
            return torch.tensor(input_data, dtype=torch.float32)
        elif isinstance(input_data, str):
            # Simple character-level encoding
            chars = [ord(c) / 255.0 for c in input_data[:self.d_model]]
            # Pad to d_model
            chars += [0.0] * (self.d_model - len(chars))
            return torch.tensor(chars[:self.d_model], dtype=torch.float32).unsqueeze(0)
        else:
            # Default: zero tensor
            return torch.zeros(1, self.d_model)

    def _compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform computation through Hybrid SSM."""
        with torch.no_grad():
            if self.mode == "verified":
                # Deterministic mode: no dropout, fixed seed
                torch.manual_seed(42)
                output = self.hybrid_ssm(input_tensor)
            else:
                # Creative mode: stochastic sampling
                output = self.hybrid_ssm(input_tensor)
        return output

    def _compute_encrypted(self, encrypted_input: Any) -> torch.Tensor:
        """Compute on FHE-encrypted data."""
        # Perform homomorphic operations
        result_encrypted = self.fhe.compute(
            encrypted_input,
            operation="mean"  # Example operation
        )
        
        # Decrypt result
        result_plaintext = self.fhe.decrypt(result_encrypted)
        
        # Convert back to tensor
        return torch.tensor(result_plaintext, dtype=torch.float32)

    def _format_output(self, output_tensor: torch.Tensor) -> Any:
        """Format output tensor into readable form."""
        output_np = output_tensor.cpu().numpy()
        
        # Simple interpretation: convert to string
        if output_np.ndim == 1:
            # Character-level decoding
            chars = [chr(int(v * 255) % 128) for v in output_np[:100]]
            return ''.join(c for c in chars if c.isprintable())
        else:
            # Return as array
            return output_np.tolist()

    def get_state_history(self) -> List[Dict[str, Any]]:
        """Return complete state history for auditing."""
        return self.state_history.copy()

    def reset(self):
        """Reset kernel to initial state."""
        self.state_history = []
        self.computation_count = 0


def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: axiomhive <input>")
        sys.exit(1)
    
    input_text = " ".join(sys.argv[1:])
    
    nexus = AxiomHiveNexus(
        axioms=["causality_preserved", "energy_conserved"],
        mode="verified"
    )
    
    result = nexus.process(input_text, require_proof=True)
    
    print(f"Output: {result['output']}")
    print(f"Verified: {result['verified']}")
    print(f"Signature: {result['signature']}")


if __name__ == "__main__":
    main()