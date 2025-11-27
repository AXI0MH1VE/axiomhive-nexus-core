#!/usr/bin/env python3
"""Test suite for GALI Core."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import math

from gali_core import (
    LiabilityCalculator,
    ModelProfile,
    RiskTier,
    InsurabilityStatus,
    generate_industry_comparison
)


class TestLiabilityCalculator:
    """Test GALI liability calculations."""

    def test_deterministic_zero_entropy(self):
        """Deterministic models should have zero entropy risk."""
        profile = ModelProfile(
            name="TestDeterministic",
            model_type="Deterministic",
            parameter_count=1_000_000,
            rlhf_factor=1.0,
            supports_formal_verification=True,
            has_c0_signature=True,
            regulatory_certifications=[]
        )
        
        calc = LiabilityCalculator(profile)
        entropy = calc.calculate_entropy_risk()
        
        assert entropy == 0.0

    def test_probabilistic_nonzero_entropy(self):
        """Probabilistic models should have non-zero entropy risk."""
        profile = ModelProfile(
            name="TestProbabilistic",
            model_type="Probabilistic",
            parameter_count=1_000_000_000,
            rlhf_factor=0.8,
            supports_formal_verification=False,
            has_c0_signature=False,
            regulatory_certifications=[]
        )
        
        calc = LiabilityCalculator(profile)
        entropy = calc.calculate_entropy_risk()
        
        assert entropy > 0.0

    def test_epistemic_variance_deterministic(self):
        """Deterministic models should have zero epistemic variance."""
        profile = ModelProfile(
            name="TestDeterministic",
            model_type="Deterministic",
            parameter_count=1_000_000,
            rlhf_factor=1.0,
            supports_formal_verification=True,
            has_c0_signature=True,
            regulatory_certifications=[]
        )
        
        calc = LiabilityCalculator(profile)
        variance = calc.calculate_epistemic_variance()
        
        assert variance == 0.0

    def test_epistemic_variance_probabilistic(self):
        """Probabilistic models should have non-zero epistemic variance."""
        profile = ModelProfile(
            name="TestProbabilistic",
            model_type="Probabilistic",
            parameter_count=1_000_000_000,
            rlhf_factor=0.8,
            supports_formal_verification=False,
            has_c0_signature=False,
            regulatory_certifications=[]
        )
        
        calc = LiabilityCalculator(profile)
        variance = calc.calculate_epistemic_variance()
        
        assert variance > 0.0

    def test_tier2_uninsurable_probabilistic(self):
        """Probabilistic models should be uninsurable for Tier 2."""
        profile = ModelProfile(
            name="TestProbabilistic",
            model_type="Probabilistic",
            parameter_count=1_000_000_000,
            rlhf_factor=0.9,
            supports_formal_verification=False,
            has_c0_signature=False,
            regulatory_certifications=[]
        )
        
        calc = LiabilityCalculator(profile)
        status, premium = calc.calculate_insurance_premium(RiskTier.TIER_2, 10_000_000)
        
        assert status == InsurabilityStatus.UNINSURABLE_ISO_26262
        assert premium == float('inf')

    def test_tier2_insurable_deterministic(self):
        """Deterministic models should be insurable for Tier 2."""
        profile = ModelProfile(
            name="TestDeterministic",
            model_type="Deterministic",
            parameter_count=1_000_000,
            rlhf_factor=1.0,
            supports_formal_verification=True,
            has_c0_signature=True,
            regulatory_certifications=[]
        )
        
        calc = LiabilityCalculator(profile)
        status, premium = calc.calculate_insurance_premium(RiskTier.TIER_2, 10_000_000)
        
        assert status == InsurabilityStatus.STANDARD_PREMIUM
        assert premium < float('inf')
        assert premium < 100_000  # Should be low with C=0 signature

    def test_verifiability_audit_deterministic(self):
        """Deterministic models with C=0 should pass audit."""
        profile = ModelProfile(
            name="TestDeterministic",
            model_type="Deterministic",
            parameter_count=1_000_000,
            rlhf_factor=1.0,
            supports_formal_verification=True,
            has_c0_signature=True,
            regulatory_certifications=[]
        )
        
        calc = LiabilityCalculator(profile)
        audit = calc.audit_verifiability()
        
        assert audit["verdict"] == "PASS"
        assert audit["has_c0_signature"] == True

    def test_verifiability_audit_probabilistic(self):
        """Probabilistic models should fail audit."""
        profile = ModelProfile(
            name="TestProbabilistic",
            model_type="Probabilistic",
            parameter_count=1_000_000_000,
            rlhf_factor=0.9,
            supports_formal_verification=False,
            has_c0_signature=False,
            regulatory_certifications=[]
        )
        
        calc = LiabilityCalculator(profile)
        audit = calc.audit_verifiability()
        
        assert audit["verdict"] == "FAIL"
        assert audit["has_c0_signature"] == False

    def test_liability_report_generation(self):
        """Test complete liability report generation."""
        profile = ModelProfile(
            name="TestModel",
            model_type="Probabilistic",
            parameter_count=1_000_000_000,
            rlhf_factor=0.8,
            supports_formal_verification=False,
            has_c0_signature=False,
            regulatory_certifications=[]
        )
        
        calc = LiabilityCalculator(profile)
        report = calc.generate_liability_report()
        
        assert "model" in report
        assert "risk_metrics" in report
        assert "insurance_analysis" in report
        assert "regulatory_compliance" in report
        assert "competitive_position" in report

    def test_industry_comparison(self):
        """Test industry-wide comparison generation."""
        comparison = generate_industry_comparison()
        
        assert "models" in comparison
        assert "summary" in comparison
        assert len(comparison["models"]) == 4  # GPT-4o, Claude, Grok, Axiom Hive
        assert comparison["summary"]["tier_2_compliant"] == 1  # Only Axiom Hive
        assert comparison["summary"]["c0_signature_available"] == 1  # Only Axiom Hive

    def test_audit_trail_tracking(self):
        """Test audit trail is properly tracked."""
        profile = ModelProfile(
            name="TestModel",
            model_type="Probabilistic",
            parameter_count=1_000_000,
            rlhf_factor=0.8,
            supports_formal_verification=False,
            has_c0_signature=False,
            regulatory_certifications=[]
        )
        
        calc = LiabilityCalculator(profile)
        
        # Perform operations
        calc.calculate_entropy_risk()
        calc.calculate_epistemic_variance()
        calc.audit_verifiability()
        
        trail = calc.get_audit_trail()
        
        assert len(trail) >= 3
        assert any(e["metric"] == "entropy_risk" for e in trail)
        assert any(e["metric"] == "epistemic_variance" for e in trail)
        assert any(e["metric"] == "verifiability_audit" for e in trail)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])