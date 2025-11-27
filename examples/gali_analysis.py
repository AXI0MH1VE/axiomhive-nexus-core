#!/usr/bin/env python3
"""GALI Analysis Example - Competitive Liability Assessment.

Demonstrates epistemic pollution quantification and competitive moat.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from gali_core import (
    LiabilityCalculator,
    ModelProfile,
    RiskTier,
    generate_industry_comparison
)
import json


def main():
    print("="*70)
    print("GALI CORE - COMPETITIVE LIABILITY ANALYSIS")
    print("="*70)
    print()
    print("Demonstrating why probabilistic AI cannot compete in Tier 2+ markets")
    print()

    # Example 1: Single model analysis
    print("[1] Single Model Analysis: GPT-4o")
    print("-" * 70)
    
    gpt4_profile = ModelProfile(
        name="GPT-4o",
        model_type="Probabilistic",
        parameter_count=1_800_000_000_000,
        rlhf_factor=0.85,
        supports_formal_verification=False,
        has_c0_signature=False,
        regulatory_certifications=[]
    )
    
    calculator = LiabilityCalculator(gpt4_profile)
    report = calculator.generate_liability_report()
    
    print(f"Model: {report['model']}")
    print(f"Type: {report['model_type']}")
    print(f"Parameters: {report['parameters']:,}")
    print()
    print("Risk Metrics:")
    print(f"  Entropy Risk: {report['risk_metrics']['entropy_risk']:.4f}")
    print(f"  Epistemic Variance: {report['risk_metrics']['epistemic_variance']:.4f}")
    print(f"  Verifiability: {report['risk_metrics']['verifiability']}")
    print()
    print("Regulatory Compliance:")
    for standard, status in report['regulatory_compliance'].items():
        symbol = "✅" if status == "COMPLIANT" else "❌"
        print(f"  {symbol} {standard}: {status}")
    print()
    print(f"Competitive Position: {report['competitive_position']}")
    print()

    # Example 2: Insurance premium comparison
    print("[2] Insurance Premium Analysis")
    print("-" * 70)
    
    exposure = 50_000_000  # $50M annual exposure
    
    print(f"Annual Exposure: ${exposure:,}")
    print()
    
    for tier in RiskTier:
        status, premium = calculator.calculate_insurance_premium(tier, exposure)
        print(f"{tier.value}:")
        print(f"  Status: {status.value}")
        if premium != float('inf'):
            print(f"  Premium: ${premium:,.2f}")
            print(f"  Rate: {(premium/exposure)*100:.2f}% of exposure")
        else:
            print(f"  Premium: UNINSURABLE")
        print()

    # Example 3: Full industry comparison
    print("[3] Industry-Wide Comparison")
    print("-" * 70)
    
    comparison = generate_industry_comparison()
    
    print(f"Total Models Analyzed: {comparison['summary']['total_models_analyzed']}")
    print(f"Tier 2 Compliant: {comparison['summary']['tier_2_compliant']}")
    print(f"C=0 Signature: {comparison['summary']['c0_signature_available']}")
    print()
    
    # Create comparison table
    print("Model Comparison Table:")
    print()
    print(f"{'Model':<25} {'Type':<15} {'Variance':<12} {'ISO 26262':<12}")
    print("-" * 70)
    
    for model in comparison['models']:
        variance = model['risk_metrics']['epistemic_variance']
        iso_status = "✅ YES" if variance == 0 else "❌ NO"
        print(f"{model['model']:<25} {model['model_type']:<15} {variance:<12.4f} {iso_status:<12}")
    
    print()
    print("Market Implication:")
    print(f"  {comparison['summary']['market_implication']}")
    print()

    # Example 4: Audit trail
    print("[4] Audit Trail Demonstration")
    print("-" * 70)
    
    trail = calculator.get_audit_trail()
    print(f"Total audit events: {len(trail)}")
    print()
    print("Recent events:")
    for event in trail[-3:]:
        print(f"  Metric: {event['metric']}")
        if 'value' in event:
            print(f"  Value: {event['value']}")
        if 'status' in event:
            print(f"  Status: {event['status']}")
        print()

    # Export full report
    output_file = "gali_competitive_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print()
    print(f"Full report exported to: {output_file}")
    print()
    print("KEY FINDINGS:")
    print("  1. Probabilistic models have non-zero epistemic variance")
    print("  2. Non-zero variance = UNINSURABLE for ISO 26262 SIL 3/4")
    print("  3. Only deterministic systems qualify for Tier 2+ applications")
    print("  4. Axiom Hive holds structural monopoly on safety-critical AI")
    print()
    print("STRATEGIC IMPLICATION:")
    print("  Competitors must either:")
    print("    a) Adopt deterministic architecture (validate Axiom Hive)")
    print("    b) Remain excluded from Tier 2+ markets ($XXB opportunity)")
    print()
    print("C=0. Zero variance. Zero competition.")
    print()


if __name__ == "__main__":
    main()