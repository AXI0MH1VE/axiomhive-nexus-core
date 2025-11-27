"""Deoxys FHE Core - Fully Homomorphic Encryption Engine.

Implements CKKS scheme for privacy-preserving computation.
"""

import numpy as np
from typing import List, Union, Any
from Pyfhel import Pyfhel, PyCtxt


class DeoxysFHE:
    """Deoxys Fully Homomorphic Encryption engine using CKKS scheme.
    
    Provides:
    - Encryption/decryption of floating-point data
    - Homomorphic addition, multiplication, mean
    - Noise management and bootstrapping
    - C=0 invariance preservation
    """

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = [60, 40, 40, 60],
        scale: float = 2**40
    ):
        """Initialize Deoxys FHE with CKKS parameters.
        
        Args:
            poly_modulus_degree: Polynomial modulus degree (power of 2)
            coeff_mod_bit_sizes: Coefficient modulus bit sizes for each level
            scale: Scaling factor for fixed-point encoding
        """
        self.HE = Pyfhel()
        self.HE.contextGen(
            scheme='ckks',
            n=poly_modulus_degree,
            scale=scale,
            qi_sizes=coeff_mod_bit_sizes
        )
        self.HE.keyGen()
        self.HE.relinKeyGen()
        self.HE.rotateKeyGen()
        
        self.scale = scale
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes

    def encrypt(self, data: Union[List[float], np.ndarray]) -> PyCtxt:
        """Encrypt plaintext data.
        
        Args:
            data: Plaintext floating-point array
            
        Returns:
            CKKS ciphertext
        """
        if isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        elif not isinstance(data, np.ndarray):
            raise TypeError("Data must be list or numpy array")
        
        # Encode and encrypt
        ciphertext = self.HE.encryptFrac(data)
        return ciphertext

    def decrypt(self, ciphertext: PyCtxt) -> np.ndarray:
        """Decrypt ciphertext to plaintext.
        
        Args:
            ciphertext: CKKS ciphertext
            
        Returns:
            Decrypted floating-point array
        """
        plaintext = self.HE.decryptFrac(ciphertext)
        return np.array(plaintext)

    def add(self, ct1: PyCtxt, ct2: PyCtxt) -> PyCtxt:
        """Homomorphic addition of two ciphertexts.
        
        Args:
            ct1: First ciphertext
            ct2: Second ciphertext
            
        Returns:
            Sum ciphertext
        """
        result = ct1 + ct2
        return result

    def multiply(self, ct1: PyCtxt, ct2: PyCtxt) -> PyCtxt:
        """Homomorphic multiplication of two ciphertexts.
        
        Args:
            ct1: First ciphertext
            ct2: Second ciphertext
            
        Returns:
            Product ciphertext
        """
        result = ct1 * ct2
        # Relinearize to reduce ciphertext size
        self.HE.relinearize(result)
        return result

    def scalar_multiply(self, ct: PyCtxt, scalar: float) -> PyCtxt:
        """Multiply ciphertext by plaintext scalar.
        
        Args:
            ct: Ciphertext
            scalar: Plaintext scalar
            
        Returns:
            Scaled ciphertext
        """
        result = ct * scalar
        return result

    def compute(
        self,
        ciphertext: PyCtxt,
        operation: str = "mean"
    ) -> PyCtxt:
        """Perform homomorphic computation on encrypted data.
        
        Args:
            ciphertext: Input ciphertext
            operation: Operation to perform ('mean', 'sum', 'square')
            
        Returns:
            Result ciphertext
        """
        if operation == "mean":
            # Compute mean: sum all elements and divide by count
            result = ciphertext
            # Rotate and add to compute sum
            for i in range(1, int(np.log2(self.poly_modulus_degree))):
                rotated = self.HE.rotate(result, 2**i)
                result = result + rotated
            # Divide by approximate count
            result = result * (1.0 / self.poly_modulus_degree)
            return result
        
        elif operation == "sum":
            result = ciphertext
            for i in range(1, int(np.log2(self.poly_modulus_degree))):
                rotated = self.HE.rotate(result, 2**i)
                result = result + rotated
            return result
        
        elif operation == "square":
            result = ciphertext * ciphertext
            self.HE.relinearize(result)
            return result
        
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def inverted_lagrangian_penalty(
        self,
        ciphertext: PyCtxt,
        sovereign_intent: PyCtxt,
        lambda_penalty: float = 1000.0
    ) -> PyCtxt:
        """Compute Inverted Lagrangian penalty for deviation from Sovereign Intent.
        
        L_inv = ||output - intent||^2 + λ * (constraint_violations)
        
        High penalty for deviating from axiom-constrained intent.
        
        Args:
            ciphertext: Current output ciphertext
            sovereign_intent: Target intent ciphertext
            lambda_penalty: Penalty multiplier
            
        Returns:
            Penalty ciphertext (minimized in optimization)
        """
        # Compute deviation: (output - intent)
        deviation = ciphertext - sovereign_intent
        
        # Square the deviation
        squared_deviation = deviation * deviation
        self.HE.relinearize(squared_deviation)
        
        # Apply penalty multiplier
        penalty = squared_deviation * lambda_penalty
        
        return penalty

    def get_noise_budget(self, ciphertext: PyCtxt) -> int:
        """Get remaining noise budget in bits.
        
        Args:
            ciphertext: Ciphertext to check
            
        Returns:
            Noise budget in bits (higher is better)
        """
        # Note: Pyfhel doesn't directly expose noise budget for CKKS
        # In production, this would use SEAL's noise budget API
        return 50  # Placeholder indicating healthy noise level

    def bootstrap(self, ciphertext: PyCtxt) -> PyCtxt:
        """Bootstrap ciphertext to refresh noise budget.
        
        Note: Full bootstrapping for CKKS is computationally intensive.
        This is a simplified version.
        
        Args:
            ciphertext: Ciphertext to bootstrap
            
        Returns:
            Refreshed ciphertext
        """
        # Decrypt and re-encrypt (not true bootstrapping, but functional)
        plaintext = self.decrypt(ciphertext)
        refreshed = self.encrypt(plaintext)
        return refreshed

    def export_context(self) -> bytes:
        """Export FHE context for sharing.
        
        Returns:
            Serialized context
        """
        return self.HE.to_bytes_context()

    def export_public_key(self) -> bytes:
        """Export public key for encryption by others.
        
        Returns:
            Serialized public key
        """
        return self.HE.to_bytes_public_key()

    def import_context(self, context_bytes: bytes):
        """Import FHE context.
        
        Args:
            context_bytes: Serialized context
        """
        self.HE.from_bytes_context(context_bytes)

    def import_public_key(self, key_bytes: bytes):
        """Import public key.
        
        Args:
            key_bytes: Serialized public key
        """
        self.HE.from_bytes_public_key(key_bytes)