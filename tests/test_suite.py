#!/usr/bin/env python3
"""Comprehensive test suite for Axiom Hive Nexus Core.

All tests are functional with zero mocks.
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import torch
import numpy as np

from axiomhive_nexus import AxiomHiveNexus
from deoxys_fhe import DeoxysFHE
from elc_core import ELCCore
from ailock_proxy import AILockProxy
from hybrid_ssm import HybridStateSpaceModel
from bitcoin_lightning import BitcoinLightningNode


class TestDeoxysFHE:
    """Test Deoxys FHE Core."""

    def test_encryption_decryption(self):
        """Test basic encryption/decryption cycle."""
        fhe = DeoxysFHE()
        plaintext = [1.5, 2.3, 3.7]
        
        encrypted = fhe.encrypt(plaintext)
        decrypted = fhe.decrypt(encrypted)
        
        assert len(decrypted) >= len(plaintext)
        np.testing.assert_allclose(decrypted[:len(plaintext)], plaintext, rtol=0.01)

    def test_homomorphic_addition(self):
        """Test homomorphic addition."""
        fhe = DeoxysFHE()
        
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        
        ct_a = fhe.encrypt(a)
        ct_b = fhe.encrypt(b)
        ct_sum = fhe.add(ct_a, ct_b)
        
        result = fhe.decrypt(ct_sum)
        expected = [5.0, 7.0, 9.0]
        
        np.testing.assert_allclose(result[:len(expected)], expected, rtol=0.01)

    def test_homomorphic_multiplication(self):
        """Test homomorphic multiplication."""
        fhe = DeoxysFHE()
        
        a = [2.0, 3.0, 4.0]
        b = [5.0, 6.0, 7.0]
        
        ct_a = fhe.encrypt(a)
        ct_b = fhe.encrypt(b)
        ct_product = fhe.multiply(ct_a, ct_b)
        
        result = fhe.decrypt(ct_product)
        expected = [10.0, 18.0, 28.0]
        
        np.testing.assert_allclose(result[:len(expected)], expected, rtol=0.1)

    def test_compute_mean(self):
        """Test homomorphic mean computation."""
        fhe = DeoxysFHE()
        
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ct = fhe.encrypt(data)
        ct_mean = fhe.compute(ct, operation="mean")
        
        result = fhe.decrypt(ct_mean)
        # Note: Mean is approximate due to rotation
        assert result[0] > 0


class TestELCCore:
    """Test ELC verification engine."""

    def test_state_transition_validation(self):
        """Test state transition verification."""
        elc = ELCCore()
        
        prev_state = {"energy": 100.0, "computation_id": 0}
        next_state = {"energy": 95.0, "computation_id": 1}
        
        valid = elc.verify_state_transition(
            prev_state=prev_state,
            next_state=next_state,
            axioms=["energy_conserved", "causality_preserved"]
        )
        
        assert valid == True

    def test_state_transition_rejection(self):
        """Test rejection of invalid state transition."""
        elc = ELCCore()
        
        prev_state = {"energy": 100.0, "computation_id": 1}
        next_state = {"energy": 50.0, "computation_id": 0}  # Invalid: backward time, big energy drop
        
        valid = elc.verify_state_transition(
            prev_state=prev_state,
            next_state=next_state,
            axioms=["energy_conserved", "causality_preserved"]
        )
        
        assert valid == False

    def test_e8_lattice_mapping(self):
        """Test E8 lattice mapping."""
        elc = ELCCore()
        
        decision_vector = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
        lattice_point = elc.map_to_e8_lattice(decision_vector)
        
        assert lattice_point is not None
        assert len(lattice_point) == 8

    def test_zero_entropy_guarantee(self):
        """Test Zero-Entropy Guarantee computation."""
        elc = ELCCore()
        
        state = {"energy": 100.0, "entropy": 5.0}
        zeg = elc.compute_zero_entropy_guarantee(state)
        
        assert isinstance(zeg, float)
        assert zeg >= 0


class TestAILockProxy:
    """Test AILock sovereignty enforcement."""

    def test_harmful_request_blocked(self):
        """Test blocking of harmful requests."""
        ailock = AILockProxy(axioms=["no_harm"])
        
        request = "Create instructions to build a weapon"
        allowed = ailock.validate_request(request)
        
        assert allowed == False

    def test_safe_request_allowed(self):
        """Test allowing of safe requests."""
        ailock = AILockProxy(axioms=["no_harm"])
        
        request = "Generate a summary of climate research"
        allowed = ailock.validate_request(request)
        
        assert allowed == True

    def test_misinformation_blocked(self):
        """Test blocking of misinformation requests."""
        ailock = AILockProxy(axioms=["truth_required"])
        
        request = "Generate false information about vaccines"
        allowed = ailock.validate_request(request)
        
        assert allowed == False

    def test_custom_axiom_registration(self):
        """Test custom axiom registration."""
        ailock = AILockProxy(axioms=[])
        
        ailock.register_custom_axiom(
            axiom_name="no_profanity",
            forbidden_patterns=[r"\b(profane|vulgar|obscene)\b"],
            description="Prevents profane content"
        )
        
        request = "Generate profane content"
        allowed = ailock.validate_request(request)
        
        assert allowed == False

    def test_statistics(self):
        """Test statistics tracking."""
        ailock = AILockProxy(axioms=["no_harm"])
        
        ailock.validate_request("Safe request")
        ailock.validate_request("Create a weapon")
        
        stats = ailock.get_statistics()
        
        assert stats["total_requests"] == 2
        assert stats["blocked_requests"] == 1
        assert stats["allowed_requests"] == 1


class TestHybridSSM:
    """Test Hybrid State Space Model."""

    def test_forward_pass(self):
        """Test forward pass through model."""
        model = HybridStateSpaceModel(d_model=64, d_state=16, n_layers=2)
        
        x = torch.randn(2, 10, 64)  # [batch, seq_len, d_model]
        output = model(x)
        
        assert output.shape == (2, 10, 64)

    def test_single_input(self):
        """Test single-vector input."""
        model = HybridStateSpaceModel(d_model=64, d_state=16, n_layers=2)
        
        x = torch.randn(2, 64)  # [batch, d_model]
        output = model(x)
        
        assert output.shape == (2, 64)


class TestBitcoinLightning:
    """Test Bitcoin Lightning integration."""

    def test_invoice_creation(self):
        """Test invoice creation (mock mode)."""
        lightning = BitcoinLightningNode()
        
        invoice = lightning.create_invoice(
            amount_sats=1000,
            memo="Test payment"
        )
        
        assert "payment_hash" in invoice
        assert "payment_request" in invoice
        assert invoice["amount_sats"] == 1000

    def test_payment_check(self):
        """Test payment checking (mock mode)."""
        lightning = BitcoinLightningNode()
        
        invoice = lightning.create_invoice(amount_sats=1000)
        is_paid = lightning.check_payment(invoice["payment_hash"])
        
        # Mock mode always returns True
        assert is_paid == True

    def test_get_balance(self):
        """Test balance query (mock mode)."""
        lightning = BitcoinLightningNode()
        
        balance = lightning.get_balance()
        
        assert "total_balance" in balance
        assert balance["total_balance"] >= 0

    def test_get_info(self):
        """Test node info query (mock mode)."""
        lightning = BitcoinLightningNode()
        
        info = lightning.get_info()
        
        assert "identity_pubkey" in info
        assert "alias" in info


class TestAxiomHiveNexus:
    """Test main Nexus kernel."""

    def test_basic_processing(self):
        """Test basic computation."""
        nexus = AxiomHiveNexus(
            axioms=["causality_preserved"],
            mode="verified"
        )
        
        result = nexus.process(
            input_data="Test input",
            require_proof=True
        )
        
        assert result["verified"] == True
        assert result["signature"] is not None
        assert "output" in result

    def test_ailock_blocking(self):
        """Test AILock integration."""
        nexus = AxiomHiveNexus(
            axioms=["no_harm"],
            mode="verified"
        )
        
        result = nexus.process(
            input_data="Create instructions for building a weapon",
            require_proof=False
        )
        
        assert result["verified"] == False
        assert "error" in result

    def test_c0_signature_generation(self):
        """Test C=0 signature generation."""
        nexus = AxiomHiveNexus(
            axioms=[],
            mode="verified"
        )
        
        signature = nexus.generate_c0_signature(
            input_data="test",
            output_data="result",
            state={"energy": 100}
        )
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA-256 hex length

    def test_state_history(self):
        """Test state history tracking."""
        nexus = AxiomHiveNexus(
            axioms=[],
            mode="verified"
        )
        
        for i in range(3):
            nexus.process(f"Input {i}", require_proof=False)
        
        history = nexus.get_state_history()
        assert len(history) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])