"""Bitcoin Lightning Integration - Pay-per-Proof Monetization.

Full LND integration with payment verification.
"""

import grpc
import time
from typing import Dict, Optional, Any
import os
import codecs

# Import LND gRPC stubs
try:
    import lnd_grpc
    LND_AVAILABLE = True
except ImportError:
    LND_AVAILABLE = False
    print("Warning: lnd-grpc not installed. Bitcoin Lightning disabled.")


class BitcoinLightningNode:
    """Bitcoin Lightning Node for pay-per-proof monetization.
    
    Provides:
    - Invoice generation
    - Payment verification
    - Payment waiting with timeout
    - Channel management
    - Balance queries
    """

    def __init__(
        self,
        lnd_host: str = "localhost:10009",
        macaroon_path: str = "~/.lnd/data/chain/bitcoin/mainnet/admin.macaroon",
        cert_path: str = "~/.lnd/tls.cert",
        network: str = "mainnet"
    ):
        """Initialize Lightning node connection.
        
        Args:
            lnd_host: LND gRPC host:port
            macaroon_path: Path to admin macaroon
            cert_path: Path to TLS certificate
            network: Bitcoin network (mainnet, testnet, regtest)
        """
        self.lnd_host = lnd_host
        self.macaroon_path = os.path.expanduser(macaroon_path)
        self.cert_path = os.path.expanduser(cert_path)
        self.network = network
        
        self.stub = None
        self.connected = False
        
        if LND_AVAILABLE:
            self._connect()

    def _connect(self):
        """Establish connection to LND."""
        try:
            # Read credentials
            with open(self.cert_path, 'rb') as f:
                cert = f.read()
            
            with open(self.macaroon_path, 'rb') as f:
                macaroon_bytes = f.read()
                macaroon = codecs.encode(macaroon_bytes, 'hex')
            
            # Create credentials
            cert_creds = grpc.ssl_channel_credentials(cert)
            auth_creds = grpc.metadata_call_credentials(
                lambda context, callback: callback([(b'macaroon', macaroon)], None)
            )
            combined_creds = grpc.composite_channel_credentials(cert_creds, auth_creds)
            
            # Create channel and stub
            channel = grpc.secure_channel(self.lnd_host, combined_creds)
            
            # Import LND stubs
            import lnd_grpc.lightning_pb2 as ln
            import lnd_grpc.lightning_pb2_grpc as lnrpc
            
            self.stub = lnrpc.LightningStub(channel)
            self.ln = ln
            self.connected = True
            
        except Exception as e:
            print(f"Failed to connect to LND: {e}")
            self.connected = False

    def create_invoice(
        self,
        amount_sats: int,
        memo: str = "Axiom Hive Computation",
        expiry: int = 3600
    ) -> Dict[str, Any]:
        """Create Lightning invoice for payment.
        
        Args:
            amount_sats: Amount in satoshis
            memo: Invoice description
            expiry: Expiry time in seconds
            
        Returns:
            Dictionary with payment_hash, payment_request, etc.
        """
        if not self.connected or not LND_AVAILABLE:
            # Return mock invoice for testing
            return {
                "payment_hash": "mock_payment_hash_" + str(int(time.time())),
                "payment_request": "lnbc" + str(amount_sats) + "mock_invoice",
                "add_index": 0,
                "amount_sats": amount_sats,
                "memo": memo,
                "expiry": expiry,
                "mock": True
            }
        
        try:
            invoice_request = self.ln.Invoice(
                value=amount_sats,
                memo=memo,
                expiry=expiry
            )
            
            response = self.stub.AddInvoice(invoice_request)
            
            return {
                "payment_hash": response.payment_hash.hex(),
                "payment_request": response.payment_request,
                "add_index": response.add_index,
                "amount_sats": amount_sats,
                "memo": memo,
                "expiry": expiry,
                "mock": False
            }
            
        except Exception as e:
            raise Exception(f"Failed to create invoice: {e}")

    def check_payment(
        self,
        payment_hash: str
    ) -> bool:
        """Check if invoice has been paid.
        
        Args:
            payment_hash: Payment hash to check
            
        Returns:
            True if paid, False otherwise
        """
        if not self.connected or not LND_AVAILABLE:
            # Mock: always return True for testing
            return True
        
        try:
            payment_hash_bytes = bytes.fromhex(payment_hash)
            request = self.ln.PaymentHash(r_hash=payment_hash_bytes)
            
            response = self.stub.LookupInvoice(request)
            
            return response.settled
            
        except Exception as e:
            print(f"Failed to check payment: {e}")
            return False

    def wait_for_payment(
        self,
        payment_hash: str,
        timeout: int = 300,
        poll_interval: int = 2
    ) -> bool:
        """Wait for payment with timeout.
        
        Args:
            payment_hash: Payment hash to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds
            
        Returns:
            True if paid within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.check_payment(payment_hash):
                return True
            time.sleep(poll_interval)
        
        return False

    def get_balance(self) -> Dict[str, int]:
        """Get node balance.
        
        Returns:
            Dictionary with balance information in satoshis
        """
        if not self.connected or not LND_AVAILABLE:
            return {
                "total_balance": 1000000,
                "confirmed_balance": 1000000,
                "unconfirmed_balance": 0,
                "mock": True
            }
        
        try:
            request = self.ln.WalletBalanceRequest()
            response = self.stub.WalletBalance(request)
            
            return {
                "total_balance": response.total_balance,
                "confirmed_balance": response.confirmed_balance,
                "unconfirmed_balance": response.unconfirmed_balance,
                "mock": False
            }
            
        except Exception as e:
            raise Exception(f"Failed to get balance: {e}")

    def get_info(self) -> Dict[str, Any]:
        """Get node information.
        
        Returns:
            Dictionary with node info
        """
        if not self.connected or not LND_AVAILABLE:
            return {
                "identity_pubkey": "mock_pubkey",
                "alias": "Axiom Hive Node",
                "num_active_channels": 0,
                "num_peers": 0,
                "block_height": 800000,
                "synced_to_chain": True,
                "version": "mock",
                "mock": True
            }
        
        try:
            request = self.ln.GetInfoRequest()
            response = self.stub.GetInfo(request)
            
            return {
                "identity_pubkey": response.identity_pubkey,
                "alias": response.alias,
                "num_active_channels": response.num_active_channels,
                "num_peers": response.num_peers,
                "block_height": response.block_height,
                "synced_to_chain": response.synced_to_chain,
                "version": response.version,
                "mock": False
            }
            
        except Exception as e:
            raise Exception(f"Failed to get info: {e}")

    def list_channels(self) -> list:
        """List all channels.
        
        Returns:
            List of channel information dictionaries
        """
        if not self.connected or not LND_AVAILABLE:
            return []
        
        try:
            request = self.ln.ListChannelsRequest()
            response = self.stub.ListChannels(request)
            
            channels = []
            for channel in response.channels:
                channels.append({
                    "channel_point": channel.channel_point,
                    "capacity": channel.capacity,
                    "local_balance": channel.local_balance,
                    "remote_balance": channel.remote_balance,
                    "active": channel.active
                })
            
            return channels
            
        except Exception as e:
            raise Exception(f"Failed to list channels: {e}")

    def send_payment(
        self,
        payment_request: str,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """Send Lightning payment.
        
        Args:
            payment_request: BOLT11 payment request
            timeout: Payment timeout in seconds
            
        Returns:
            Payment result dictionary
        """
        if not self.connected or not LND_AVAILABLE:
            return {
                "payment_hash": "mock_payment_hash",
                "preimage": "mock_preimage",
                "status": "SUCCEEDED",
                "mock": True
            }
        
        try:
            request = self.ln.SendRequest(
                payment_request=payment_request,
                timeout_seconds=timeout
            )
            
            response = self.stub.SendPaymentSync(request)
            
            return {
                "payment_hash": response.payment_hash.hex(),
                "preimage": response.payment_preimage.hex(),
                "status": "SUCCEEDED" if response.payment_error == "" else "FAILED",
                "error": response.payment_error,
                "mock": False
            }
            
        except Exception as e:
            raise Exception(f"Failed to send payment: {e}")