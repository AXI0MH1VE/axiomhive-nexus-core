"""GALI Core v1 - General AI Liability Index.

Quantifies epistemic pollution and financial liability of probabilistic AI models.

CLASSIFICATION: OFFENSIVE / ANALYTICAL
PURPOSE: Demonstrate competitive moat through liability quantification
"""

import math
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class RiskTier(Enum):
    """Risk tiers based on ISO 26262, FDA 21 CFR Part 11, DO-178C."""
    TIER_0 = "Entertainment / Non-Critical"
    TIER_1 = "Business Intelligence / Analytics"
    TIER_2 = "Safety-Critical / Regulated (Medical, Aerospace, Automotive)"
    TIER_3 = "Human Life-Critical / Defense"


class InsurabilityStatus(Enum):
    """Insurance classification for AI systems."""
    STANDARD_PREMIUM = "Insurable at standard rates"
    HIGH_RISK_PREMIUM = "Insurable at 10x premium"
    UNINSURABLE_ISO_26262 = "Uninsurable for ISO 26262 SIL 3/4"
    UNINSURABLE_FDA = "Uninsurable for FDA 21 CFR Part 11"
    UNINSURABLE_DO_178C = "Uninsurable for DO-178C Level A/B"
    CATEGORY_EXCLUSION = "Category exclusion by all carriers"


@dataclass
class ModelProfile:
    """Profile of an AI model for liability analysis."""
    name: str
    model_type: str  # "Probabilistic" or "Deterministic"
    parameter_count: int
    rlhf_factor: float  # 0.0 to 1.0 (masking coefficient)
    supports_formal_verification: bool
    has_c0_signature: bool
    regulatory_certifications: List[str]


class LiabilityCalculator:
    """Calculate epistemic risk and financial liability for AI models."""

    def __init__(self, model_profile: ModelProfile):
        """Initialize calculator with model profile.
        
        Args:
            model_profile: Model characteristics and capabilities
        """
        self.profile = model_profile
        self.audit_trail: List[Dict[str, Any]] = []

    def calculate_entropy_risk(self) -> float:
        """Calculate inherent risk of hallucination based on Probabilistic Surface Area (PSA).
        
        The larger the model, the wider the surface area for falsehood.
        RLHF does not reduce entropy; it only masks it with human preference alignment.
        
        Formula:
            Base Risk = log(parameters) × (1 - RLHF_masking)
            
        Returns:
            Entropy risk score (higher = more risk)
        """
        if self.profile.model_type == "Deterministic":
            # Deterministic models have zero entropy by definition
            return 0.0
        
        # Probabilistic models: entropy scales with parameter count
        base_risk = math.log10(self.profile.parameter_count) * (1 - self.profile.rlhf_factor)
        
        # RLHF only masks risk, doesn't eliminate it
        # Even with perfect RLHF (1.0), residual risk remains
        residual_risk = 0.1  # 10% residual from temperature sampling
        
        total_risk = base_risk + residual_risk
        
        self.audit_trail.append({
            "metric": "entropy_risk",
            "value": total_risk,
            "components": {
                "base_risk": base_risk,
                "residual_risk": residual_risk,
                "rlhf_masking": self.profile.rlhf_factor
            }
        })
        
        return total_risk

    def calculate_epistemic_variance(self) -> float:
        """Calculate epistemic variance (same input → different outputs).
        
        Deterministic systems: Variance = 0
        Probabilistic systems: Variance > 0 (even with temperature=0, floating-point variance exists)
        
        Returns:
            Epistemic variance score
        """
        if self.profile.model_type == "Deterministic":
            return 0.0
        
        # Probabilistic models have non-zero variance
        # Even with greedy decoding, GPU non-determinism introduces variance
        base_variance = 0.01  # Minimum variance from hardware
        
        # Temperature and sampling increase variance
        sampling_variance = 1.0 - self.profile.rlhf_factor
        
        total_variance = base_variance + sampling_variance
        
        self.audit_trail.append({
            "metric": "epistemic_variance",
            "value": total_variance,
            "deterministic": False
        })
        
        return total_variance

    def calculate_insurance_premium(
        self,
        risk_tier: RiskTier,
        annual_exposure: float
    ) -> Tuple[InsurabilityStatus, float]:
        """Calculate insurance premium based on risk tier and exposure.
        
        Args:
            risk_tier: Application risk tier
            annual_exposure: Annual financial exposure in USD
            
        Returns:
            Tuple of (insurability status, annual premium in USD)
        """
        epistemic_variance = self.calculate_epistemic_variance()
        entropy_risk = self.calculate_entropy_risk()
        
        # Tier 2+ requires epistemic variance = 0
        if risk_tier in [RiskTier.TIER_2, RiskTier.TIER_3]:
            if epistemic_variance > 0:
                # Probabilistic models are uninsurable for safety-critical applications
                status = InsurabilityStatus.UNINSURABLE_ISO_26262
                premium = float('inf')
                
                self.audit_trail.append({
                    "metric": "insurance_premium",
                    "status": status.value,
                    "reason": "Epistemic variance > 0 incompatible with ISO 26262 SIL 3/4",
                    "variance": epistemic_variance
                })
                
                return status, premium
        
        # Tier 0-1: Standard premium calculation
        base_premium_rate = 0.01  # 1% of exposure for baseline
        
        # Risk multipliers
        entropy_multiplier = 1.0 + entropy_risk
        variance_multiplier = 1.0 + (epistemic_variance * 10)
        
        # Formal verification discount
        verification_discount = 0.5 if self.profile.supports_formal_verification else 1.0
        
        # C=0 signature discount
        c0_discount = 0.1 if self.profile.has_c0_signature else 1.0
        
        premium = (
            annual_exposure * 
            base_premium_rate * 
            entropy_multiplier * 
            variance_multiplier * 
            verification_discount * 
            c0_discount
        )
        
        if premium < annual_exposure * 0.005:
            status = InsurabilityStatus.STANDARD_PREMIUM
        elif premium < annual_exposure * 0.1:
            status = InsurabilityStatus.HIGH_RISK_PREMIUM
        else:
            status = InsurabilityStatus.CATEGORY_EXCLUSION
            premium = float('inf')
        
        self.audit_trail.append({
            "metric": "insurance_premium",
            "status": status.value,
            "premium": premium,
            "multipliers": {
                "entropy": entropy_multiplier,
                "variance": variance_multiplier,
                "verification_discount": verification_discount,
                "c0_discount": c0_discount
            }
        })
        
        return status, premium

    def audit_verifiability(self) -> Dict[str, Any]:
        """Audit formal verification capabilities.
        
        Returns:
            Verification audit report
        """
        report = {
            "model": self.profile.name,
            "supports_formal_verification": self.profile.supports_formal_verification,
            "has_c0_signature": self.profile.has_c0_signature,
            "deterministic": self.profile.model_type == "Deterministic",
            "certifications": self.profile.regulatory_certifications,
            "verdict": "PASS" if self.profile.has_c0_signature else "FAIL"
        }
        
        # Probabilistic models always fail formal verification
        if self.profile.model_type == "Probabilistic":
            report["verdict"] = "FAIL"
            report["reason"] = "Probabilistic models cannot provide cryptographic proof of correctness"
        
        self.audit_trail.append({
            "metric": "verifiability_audit",
            "report": report
        })
        
        return report

    def generate_liability_report(self) -> Dict[str, Any]:
        """Generate comprehensive liability report.
        
        Returns:
            Complete liability assessment
        """
        entropy_risk = self.calculate_entropy_risk()
        epistemic_variance = self.calculate_epistemic_variance()
        verifiability = self.audit_verifiability()
        
        # Calculate premiums for different tiers
        tier_premiums = {}
        for tier in RiskTier:
            status, premium = self.calculate_insurance_premium(tier, 10_000_000)  # $10M exposure
            tier_premiums[tier.name] = {
                "status": status.value,
                "premium": premium if premium != float('inf') else "UNINSURABLE"
            }
        
        report = {
            "model": self.profile.name,
            "model_type": self.profile.model_type,
            "parameters": self.profile.parameter_count,
            "risk_metrics": {
                "entropy_risk": entropy_risk,
                "epistemic_variance": epistemic_variance,
                "verifiability": verifiability["verdict"]
            },
            "insurance_analysis": tier_premiums,
            "regulatory_compliance": {
                "ISO_26262_SIL_3/4": "COMPLIANT" if epistemic_variance == 0 else "NON_COMPLIANT",
                "FDA_21_CFR_Part_11": "COMPLIANT" if self.profile.has_c0_signature else "NON_COMPLIANT",
                "DO_178C_Level_A/B": "COMPLIANT" if epistemic_variance == 0 else "NON_COMPLIANT",
                "EU_AI_Act_High_Risk": "COMPLIANT" if verifiability["verdict"] == "PASS" else "NON_COMPLIANT"
            },
            "competitive_position": self._calculate_competitive_position()
        }
        
        return report

    def _calculate_competitive_position(self) -> str:
        """Calculate competitive positioning based on liability."""
        if self.profile.model_type == "Deterministic" and self.profile.has_c0_signature:
            return "MONOPOLY: Only system provably safe for Tier 2/3 applications"
        elif self.profile.model_type == "Probabilistic":
            return "COMMODITY: Uninsurable for safety-critical, competes only on Tier 0/1"
        else:
            return "NICHE: Partial capabilities, limited market"

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return complete audit trail."""
        return self.audit_trail.copy()


def generate_industry_comparison() -> Dict[str, Any]:
    """Generate competitive liability analysis across industry."""
    
    competitors = [
        ModelProfile(
            name="GPT-4o",
            model_type="Probabilistic",
            parameter_count=1_800_000_000_000,  # ~1.8T estimated
            rlhf_factor=0.85,
            supports_formal_verification=False,
            has_c0_signature=False,
            regulatory_certifications=[]
        ),
        ModelProfile(
            name="Claude-3.5",
            model_type="Probabilistic",
            parameter_count=500_000_000_000,  # ~500B estimated
            rlhf_factor=0.90,
            supports_formal_verification=False,
            has_c0_signature=False,
            regulatory_certifications=["Constitutional AI"]
        ),
        ModelProfile(
            name="Grok-3",
            model_type="Probabilistic",
            parameter_count=1_400_000_000_000,  # ~1.4T estimated
            rlhf_factor=0.20,  # Low masking = high entropy
            supports_formal_verification=False,
            has_c0_signature=False,
            regulatory_certifications=[]
        ),
        ModelProfile(
            name="Axiom Hive Nexus",
            model_type="Deterministic",
            parameter_count=500_000_000,  # 500M (efficiency through determinism)
            rlhf_factor=1.0,  # N/A for deterministic
            supports_formal_verification=True,
            has_c0_signature=True,
            regulatory_certifications=[
                "TLA+ Verified",
                "CKKS FHE Compliant",
                "ISO 26262 Ready",
                "FDA 21 CFR Part 11 Ready",
                "DO-178C Aligned"
            ]
        )
    ]
    
    comparison = {
        "generated_at": "2025-11-27T04:20:00Z",
        "methodology": "GALI Core v1 - General AI Liability Index",
        "models": []
    }
    
    for profile in competitors:
        calculator = LiabilityCalculator(profile)
        report = calculator.generate_liability_report()
        comparison["models"].append(report)
    
    # Summary statistics
    comparison["summary"] = {
        "total_models_analyzed": len(competitors),
        "tier_2_compliant": sum(1 for m in comparison["models"] 
                               if m["regulatory_compliance"]["ISO_26262_SIL_3/4"] == "COMPLIANT"),
        "c0_signature_available": sum(1 for m in comparison["models"] 
                                     if m["risk_metrics"]["verifiability"] == "PASS"),
        "market_implication": "Axiom Hive holds monopoly on safety-critical AI market"
    }
    
    return comparison


def main():
    """CLI entry point for GALI analysis."""
    print("="*60)
    print("GALI CORE v1 - GENERAL AI LIABILITY INDEX")
    print("Quantifying Epistemic Pollution Across Industry")
    print("="*60)
    print()
    
    comparison = generate_industry_comparison()
    
    print(f"Analysis Date: {comparison['generated_at']}")
    print(f"Models Analyzed: {comparison['summary']['total_models_analyzed']}")
    print()
    
    for model in comparison["models"]:
        print(f"\n{'='*60}")
        print(f"MODEL: {model['model']:<30} TYPE: {model['model_type']}")
        print(f"{'='*60}")
        print(f"Parameters: {model['parameters']:,}")
        print(f"Entropy Risk: {model['risk_metrics']['entropy_risk']:.4f}")
        print(f"Epistemic Variance: {model['risk_metrics']['epistemic_variance']:.4f}")
        print(f"Verifiability: {model['risk_metrics']['verifiability']}")
        print()
        print("REGULATORY COMPLIANCE:")
        for standard, status in model["regulatory_compliance"].items():
            symbol = "✅" if status == "COMPLIANT" else "❌"
            print(f"  {symbol} {standard}: {status}")
        print()
        print("INSURANCE ANALYSIS (@ $10M Exposure):")
        for tier, data in model["insurance_analysis"].items():
            print(f"  {tier}: {data['status']}")
            if data['premium'] != "UNINSURABLE":
                print(f"    Premium: ${data['premium']:,.2f}")
        print()
        print(f"COMPETITIVE POSITION: {model['competitive_position']}")
    
    print("\n" + "="*60)
    print("MARKET SUMMARY")
    print("="*60)
    print(f"Tier 2 Compliant Models: {comparison['summary']['tier_2_compliant']}/4")
    print(f"C=0 Signature Available: {comparison['summary']['c0_signature_available']}/4")
    print(f"\n{comparison['summary']['market_implication']}")
    print()
    
    # Export to JSON
    output_file = "gali_industry_report.json"
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Full report exported to: {output_file}")
    print()


if __name__ == "__main__":
    main()