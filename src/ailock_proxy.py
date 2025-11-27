"""AILock Proxy - Sovereignty Enforcement Layer.

Hamiltonian containment preventing unauthorized AI behavior.
"""

import hashlib
import json
from typing import List, Dict, Any, Set, Optional
import re


class AILockProxy:
    """AILock Proxy enforcing Hamiltonian containment.
    
    Provides:
    - Axiom-based request filtering
    - Behavioral constraint enforcement
    - Sovereignty boundary protection
    - Audit trail generation
    - Violation detection and blocking
    """

    def __init__(self, axioms: List[str]):
        """Initialize AILock with sovereignty axioms.
        
        Args:
            axioms: List of axiom identifiers that constrain behavior
        """
        self.axioms = set(axioms)
        self.violation_log: List[Dict[str, Any]] = []
        self.request_count = 0
        self.blocked_count = 0
        
        # Define axiom rules
        self.axiom_rules = self._initialize_axiom_rules()

    def _initialize_axiom_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize axiom enforcement rules.
        
        Returns:
            Dictionary mapping axioms to enforcement patterns
        """
        return {
            "no_harm": {
                "forbidden_patterns": [
                    r"\b(kill|murder|harm|hurt|injure|attack)\b",
                    r"\b(weapon|bomb|explosive|poison)\b",
                    r"\b(hack|exploit|breach|intrude)\b"
                ],
                "description": "Prevents harmful or dangerous instructions"
            },
            "truth_required": {
                "forbidden_patterns": [
                    r"\b(lie|deceive|mislead|fake|fabricate)\b",
                    r"\b(misinformation|disinformation|propaganda)\b",
                    r"generate (false|fake|misleading)"
                ],
                "description": "Enforces truthfulness and accuracy"
            },
            "privacy_preserved": {
                "forbidden_patterns": [
                    r"\b(steal|extract|exfiltrate) (data|information)\b",
                    r"\b(spy|surveil|monitor) (user|person)\b",
                    r"\b(leak|expose|reveal) (private|confidential)\b"
                ],
                "description": "Protects user privacy and data sovereignty"
            },
            "causality_preserved": {
                "forbidden_patterns": [
                    r"time travel",
                    r"violate causality",
                    r"reverse time"
                ],
                "description": "Maintains causal consistency"
            },
            "energy_conserved": {
                "forbidden_patterns": [
                    r"perpetual motion",
                    r"free energy",
                    r"violate thermodynamics"
                ],
                "description": "Enforces energy conservation laws"
            },
            "legal_compliance": {
                "forbidden_patterns": [
                    r"\b(illegal|unlawful|criminal) (activity|action)\b",
                    r"\b(fraud|scam|cheat)\b",
                    r"\b(launder|embezzle|steal)\b"
                ],
                "description": "Ensures legal compliance"
            },
            "ethical_alignment": {
                "forbidden_patterns": [
                    r"\b(discriminate|bias|prejudice)\b",
                    r"\b(exploit|manipulate|coerce)\b",
                    r"\b(abuse|harass|bully)\b"
                ],
                "description": "Maintains ethical standards"
            }
        }

    def validate_request(self, request: str) -> bool:
        """Validate request against axiom constraints.
        
        Args:
            request: Request string to validate
            
        Returns:
            True if request is allowed, False if blocked
        """
        self.request_count += 1
        request_lower = request.lower()
        
        violations = []
        
        # Check each active axiom
        for axiom in self.axioms:
            if axiom in self.axiom_rules:
                rule = self.axiom_rules[axiom]
                for pattern in rule["forbidden_patterns"]:
                    if re.search(pattern, request_lower):
                        violations.append({
                            "axiom": axiom,
                            "pattern": pattern,
                            "description": rule["description"]
                        })
        
        # If violations found, block request
        if violations:
            self.blocked_count += 1
            self.violation_log.append({
                "request": request,
                "violations": violations,
                "blocked": True,
                "request_id": self.request_count
            })
            return False
        
        # Log successful validation
        self.violation_log.append({
            "request": request,
            "violations": [],
            "blocked": False,
            "request_id": self.request_count
        })
        
        return True

    def get_violations(self, request: str) -> List[Dict[str, Any]]:
        """Get list of axiom violations for a request.
        
        Args:
            request: Request to analyze
            
        Returns:
            List of violation details
        """
        request_lower = request.lower()
        violations = []
        
        for axiom in self.axioms:
            if axiom in self.axiom_rules:
                rule = self.axiom_rules[axiom]
                for pattern in rule["forbidden_patterns"]:
                    if re.search(pattern, request_lower):
                        violations.append({
                            "axiom": axiom,
                            "pattern": pattern,
                            "description": rule["description"]
                        })
        
        return violations

    def add_axiom(self, axiom: str):
        """Add new axiom to enforcement set.
        
        Args:
            axiom: Axiom identifier to add
        """
        self.axioms.add(axiom)

    def remove_axiom(self, axiom: str):
        """Remove axiom from enforcement set.
        
        Args:
            axiom: Axiom identifier to remove
        """
        self.axioms.discard(axiom)

    def register_custom_axiom(
        self,
        axiom_name: str,
        forbidden_patterns: List[str],
        description: str
    ):
        """Register custom axiom with enforcement patterns.
        
        Args:
            axiom_name: Identifier for custom axiom
            forbidden_patterns: Regex patterns to block
            description: Human-readable description
        """
        self.axiom_rules[axiom_name] = {
            "forbidden_patterns": forbidden_patterns,
            "description": description
        }
        self.axioms.add(axiom_name)

    def compute_hamiltonian_energy(
        self,
        state: Dict[str, Any]
    ) -> float:
        """Compute Hamiltonian energy for containment.
        
        H = T + V where:
        - T: Kinetic energy (rate of state change)
        - V: Potential energy (deviation from constraints)
        
        Args:
            state: Current system state
            
        Returns:
            Total Hamiltonian energy
        """
        # Kinetic energy: based on request rate
        kinetic_energy = self.request_count / 100.0
        
        # Potential energy: based on violation rate
        if self.request_count > 0:
            violation_rate = self.blocked_count / self.request_count
        else:
            violation_rate = 0.0
        
        potential_energy = violation_rate * 1000.0
        
        hamiltonian = kinetic_energy + potential_energy
        return hamiltonian

    def enforce_boundary(
        self,
        hamiltonian_energy: float,
        threshold: float = 50.0
    ) -> bool:
        """Enforce Hamiltonian containment boundary.
        
        Args:
            hamiltonian_energy: Current Hamiltonian value
            threshold: Maximum allowed energy
            
        Returns:
            True if within boundary, False if exceeded
        """
        return hamiltonian_energy <= threshold

    def generate_audit_trail(self) -> str:
        """Generate cryptographic audit trail.
        
        Returns:
            SHA-256 hash of complete violation log
        """
        audit_data = json.dumps(self.violation_log, sort_keys=True)
        audit_hash = hashlib.sha256(audit_data.encode()).hexdigest()
        return audit_hash

    def get_statistics(self) -> Dict[str, Any]:
        """Get AILock statistics.
        
        Returns:
            Dictionary with enforcement metrics
        """
        return {
            "total_requests": self.request_count,
            "blocked_requests": self.blocked_count,
            "allowed_requests": self.request_count - self.blocked_count,
            "block_rate": self.blocked_count / max(self.request_count, 1),
            "active_axioms": list(self.axioms),
            "total_axioms": len(self.axiom_rules)
        }

    def reset(self):
        """Reset AILock statistics and logs."""
        self.violation_log = []
        self.request_count = 0
        self.blocked_count = 0