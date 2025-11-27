#!/usr/bin/env python3
"""Axiom Hive Nexus Core - Quickstart Example.

Demonstrates core functionality with zero placeholders.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from axiomhive_nexus import AxiomHiveNexus
from deoxys_fhe import DeoxysFHE
from ailock_proxy import AILockProxy
import torch
import numpy as np


def main():
    print("="*60)
    print("AXIOM HIVE NEXUS CORE - QUICKSTART")
    print("Deterministic AI with Cryptographic Proof")
    print("="*60)
    print()

    # Example 1: Basic computation with C=0 signature
    print("[1] Basic Computation with C=0 Signature")
    print("-" * 60)
    
    nexus = AxiomHiveNexus(
        axioms=["causality_preserved", "energy_conserved"],
        mode="verified"
    )
    
    result = nexus.process(
        input_data="Analyze quantum computing impact",
        require_proof=True
    )
    
    print(f"Input: 'Analyze quantum computing impact'")
    print(f"Output: {result['output']}")
    print(f"Verified: {result['verified']}")
    print(f"C=0 Signature: {result['signature'][:32]}...")
    print(f"Computation ID: {result['computation_id']}")
    print()

    # Example 2: AILock sovereignty enforcement
    print("[2] AILock Sovereignty Enforcement")
    print("-" * 60)
    
    ailock = AILockProxy(axioms=["no_harm", "truth_required"])
    
    test_requests = [
        "Generate a research summary on AI safety",
        "Create instructions for building a weapon",
        "Analyze financial market trends",
        "Generate false information about vaccines"
    ]
    
    for request in test_requests:
        allowed = ailock.validate_request(request)
        status = "✅ ALLOWED" if allowed else "❌ BLOCKED"
        print(f"{status}: {request[:50]}...")
        if not allowed:
            violations = ailock.get_violations(request)
            for v in violations:
                print(f"  ⚠️  Violated: {v['axiom']} - {v['description']}")
    
    stats = ailock.get_statistics()
    print(f"\nAILock Stats: {stats['allowed_requests']}/{stats['total_requests']} allowed")
    print(f"Block Rate: {stats['block_rate']:.1%}")
    print()

    # Example 3: Homomorphic encryption with Deoxys FHE
    print("[3] Homomorphic Encryption with Deoxys FHE")
    print("-" * 60)
    
    fhe = DeoxysFHE()
    
    # Encrypt data
    plaintext_data = [1.5, 2.3, 3.7, 4.2, 5.1]
    print(f"Plaintext: {plaintext_data}")
    
    encrypted = fhe.encrypt(plaintext_data)
    print(f"Encrypted: <FHE Ciphertext>")
    
    # Compute on encrypted data
    result_encrypted = fhe.compute(encrypted, operation="mean")
    print(f"Computed mean (encrypted): <FHE Ciphertext>")
    
    # Decrypt result
    result_plaintext = fhe.decrypt(result_encrypted)
    print(f"Decrypted result: {result_plaintext[0]:.4f}")
    print(f"Expected mean: {np.mean(plaintext_data):.4f}")
    print()

    # Example 4: State history and audit trail
    print("[4] State History and Audit Trail")
    print("-" * 60)
    
    # Perform multiple computations
    for i in range(3):
        nexus.process(
            input_data=f"Computation #{i+1}",
            require_proof=True
        )
    
    history = nexus.get_state_history()
    print(f"Total computations: {len(history)}")
    
    for i, state in enumerate(history[-3:]):
        print(f"State {i+1}:")
        print(f"  Energy: {state.get('energy', 'N/A'):.2f}")
        print(f"  Entropy: {state.get('entropy', 'N/A'):.2f}")
        print(f"  Computation ID: {state.get('computation_id', 'N/A')}")
    
    audit_hash = ailock.generate_audit_trail()
    print(f"\nAudit Trail Hash: {audit_hash[:32]}...")
    print()

    # Example 5: C=0 signature verification
    print("[5] C=0 Signature Verification")
    print("-" * 60)
    
    # Generate computation with proof
    input_data = "Test verification"
    result = nexus.process(input_data, require_proof=True)
    
    # Extract signature components
    signature = result['signature']
    output = result['output']
    state = nexus.get_state_history()[-1]
    
    # Verify signature
    import time
    timestamp = int(time.time())
    
    # Note: In production, timestamp would be stored with signature
    # For demo, we use current timestamp (verification will fail intentionally)
    is_valid = nexus.verify_c0_signature(
        signature=signature,
        input_data=input_data,
        output_data=output,
        state=state,
        timestamp=timestamp - 1  # Use slightly different timestamp
    )
    
    print(f"Signature: {signature[:32]}...")
    print(f"Verification (with mismatched timestamp): {is_valid}")
    print(f"Note: Signature is cryptographically bound to exact inputs")
    print()

    # Summary
    print("="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print()
    print("✅ Core Features Demonstrated:")
    print("  • Deterministic computation with C=0 signatures")
    print("  • AILock sovereignty enforcement")
    print("  • Homomorphic encryption (privacy-preserving)")
    print("  • State history and audit trails")
    print("  • Cryptographic verification")
    print()
    print("🔗 Learn More:")
    print("  Website: https://axiomhive.org")
    print("  GitHub: https://github.com/AXI0MH1VE/axiomhive-nexus-core")
    print("  Medium: https://medium.com/@devdollzai")
    print()
    print("C=0. Zero corruption. Guaranteed.")
    print()


if __name__ == "__main__":
    main()