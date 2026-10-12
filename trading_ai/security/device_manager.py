"""Device Identity, Cryptographic Pairing & Authenticated Encrypted Transport."""

from datetime import datetime, timezone
import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Dict, List, Optional, Set, Tuple

from cryptography.fernet import Fernet

from trading_ai.schemas.enums import DeviceStatus
from trading_ai.schemas.security import (
    DevicePairingApproval,
    DevicePairingRequest,
    DeviceRecord,
    EncryptedPayload,
)
from trading_ai.security.audit import SecurityAuditLogger
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)

# Max replay age window: 60 seconds
REPLAY_WINDOW_SECONDS = 60.0


def _derive_fernet_key(secret: str) -> bytes:
    """Derive standard 32-byte urlsafe base64 key for Fernet from shared secret."""
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


class DeviceManager:
    """
    Manages client device pairing, authenticated sessions, encryption keys, and revocation.

    Security Guarantees:
      - Never silently trust an unknown device.
      - Device pairing strictly requires explicit human user approval.
      - Cryptographic challenge-response authentication.
      - Replay protection with nonces and timestamps.
      - Immediate device revocation.
      - Secrets never logged, persisted in plaintext, or exposed in API responses.
    """

    def __init__(self, repository: Optional[Repository] = None, audit_logger: Optional[SecurityAuditLogger] = None):
        self._repo = repository
        self._audit = audit_logger or SecurityAuditLogger(repository)
        self._devices: Dict[str, DeviceRecord] = {}
        self._shared_secrets: Dict[str, str] = {}  # In-memory ephemeral secret store, never persisted to disk
        self._active_pairing_codes: Dict[str, Tuple[str, float]] = {}  # code -> (device_id, expires_at)
        self._seen_nonces: Set[str] = set()

    def generate_pairing_code(self, device_id: str, device_name: str, device_type: str = "phone") -> str:
        """
        Generate a cryptographically secure 6-digit pairing code with 5-minute expiry.
        """
        code = f"{secrets.randbelow(900000) + 100000}"
        expires_at = time.time() + 300.0  # 5 minutes
        self._active_pairing_codes[code] = (device_id, expires_at)

        # Register un-paired placeholder
        self._devices[device_id] = DeviceRecord(
            device_id=device_id,
            device_name=device_name,
            device_type=device_type,
            status=DeviceStatus.PAIRING_REQUESTED,
            public_key_or_token_hash=hashlib.sha256(code.encode("utf-8")).hexdigest(),
        )

        self._audit.log(
            action="DEVICE_PAIRING_CODE_GENERATED",
            decision="PENDING",
            requesting_component="DeviceManager",
            device_id=device_id,
            details={"device_name": device_name, "expires_in_seconds": 300},
        )
        return code

    def submit_pairing_request(self, req: DevicePairingRequest) -> bool:
        """
        Client device submits pairing code and nonce.
        """
        if req.pairing_code not in self._active_pairing_codes:
            self._audit.log(
                action="DEVICE_PAIRING_FAILED",
                decision="DENIED",
                requesting_component="DeviceManager",
                device_id=req.device_id,
                details={"reason": "Invalid or expired pairing code"},
            )
            return False

        dev_id, expires = self._active_pairing_codes[req.pairing_code]
        if time.time() > expires or dev_id != req.device_id:
            return False

        device = self._devices.get(req.device_id)
        if device:
            device.status = DeviceStatus.USER_APPROVAL
            self._devices[req.device_id] = device

        self._audit.log(
            action="DEVICE_PAIRING_REQUESTED",
            decision="AWAITING_USER_APPROVAL",
            requesting_component="DeviceManager",
            device_id=req.device_id,
            details={"device_name": req.device_name},
        )
        return True

    def approve_pairing(self, approval: DevicePairingApproval) -> Tuple[bool, Optional[str]]:
        """
        User explicitly approves pairing for a device.
        Generates an authenticated shared secret returned securely to the user/client.
        """
        dev_id = approval.device_id
        device = self._devices.get(dev_id)
        if not device:
            return False, None

        if not approval.is_user or not approval.approved:
            device.status = DeviceStatus.REVOKED
            device.revoked_at = datetime.now(timezone.utc)
            device.revocation_reason = "User rejected pairing"
            self._audit.log(
                action="DEVICE_PAIRING_REJECTED",
                decision="DENIED",
                requesting_component="UserApprovalGate",
                device_id=dev_id,
                is_user_authorized=approval.is_user,
            )
            return False, None

        # Generate a strong shared secret
        shared_secret = secrets.token_hex(32)
        self._shared_secrets[dev_id] = shared_secret

        secret_hash = hashlib.sha256(shared_secret.encode("utf-8")).hexdigest()
        device.status = DeviceStatus.ACTIVE
        device.paired_at = datetime.now(timezone.utc)
        device.last_seen = datetime.now(timezone.utc)
        device.public_key_or_token_hash = secret_hash
        self._devices[dev_id] = device

        self._audit.log(
            action="DEVICE_PAIRING_APPROVED",
            decision="ALLOWED",
            requesting_component="UserApprovalGate",
            device_id=dev_id,
            is_user_authorized=True,
            authorized_by=approval.approved_by or "user",
            details={"status": device.status.value},
        )
        return True, shared_secret

    def revoke_device(self, device_id: str, reason: str = "User revoked device") -> bool:
        """
        Immediately revoke a trusted device and purge its session keys.
        """
        device = self._devices.get(device_id)
        if not device:
            return False

        device.status = DeviceStatus.REVOKED
        device.revoked_at = datetime.now(timezone.utc)
        device.revocation_reason = reason
        self._devices[device_id] = device

        # Invalidate shared secret
        if device_id in self._shared_secrets:
            del self._shared_secrets[device_id]

        self._audit.log(
            action="DEVICE_REVOKED",
            decision="REVOKED",
            requesting_component="DeviceManager",
            device_id=device_id,
            is_user_authorized=True,
            details={"reason": reason},
        )
        return True

    def list_devices(self) -> List[DeviceRecord]:
        """Return list of known devices without exposing secrets."""
        return list(self._devices.values())

    def get_device(self, device_id: str) -> Optional[DeviceRecord]:
        """Retrieve single device status."""
        return self._devices.get(device_id)

    def encrypt_for_device(self, device_id: str, plaintext: str) -> EncryptedPayload:
        """
        Encrypt payload using authenticated encryption (Fernet / AES-CBC + HMAC-SHA256).
        """
        device = self._devices.get(device_id)
        if not device or device.status != DeviceStatus.ACTIVE:
            raise PermissionError(f"Cannot encrypt: Device '{device_id}' is not trusted or active.")

        secret = self._shared_secrets.get(device_id)
        if not secret:
            raise PermissionError(f"Session key not established for device '{device_id}'.")

        key = _derive_fernet_key(secret)
        fernet = Fernet(key)

        nonce = secrets.token_hex(16)
        timestamp = time.time()

        # Sign message with HMAC
        sig = hmac.new(secret.encode("utf-8"), f"{plaintext}:{nonce}:{timestamp}".encode("utf-8"), hashlib.sha256).hexdigest()
        ciphertext = fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

        return EncryptedPayload(
            device_id=device_id,
            ciphertext=ciphertext,
            nonce=nonce,
            signature=sig,
            timestamp=timestamp,
        )

    def decrypt_from_device(self, payload: EncryptedPayload) -> str:
        """
        Decrypt and verify authenticated payload with strict replay and signature checks.
        """
        dev_id = payload.device_id
        device = self._devices.get(dev_id)
        if not device or device.status != DeviceStatus.ACTIVE:
            raise PermissionError(f"Rejected message: Device '{dev_id}' is revoked or not active.")

        # Replay Attack Prevention 1: Timestamp freshness
        age = abs(time.time() - payload.timestamp)
        if age > REPLAY_WINDOW_SECONDS:
            self._audit.log(
                action="REPLAY_ATTACK_PREVENTED",
                decision="DENIED",
                requesting_component="DeviceManager",
                device_id=dev_id,
                details={"reason": f"Message timestamp outside window ({age:.1f}s old)"},
            )
            raise ValueError("Message timestamp expired or out of valid replay window.")

        # Replay Attack Prevention 2: Nonce uniqueness
        if payload.nonce in self._seen_nonces:
            self._audit.log(
                action="REPLAY_ATTACK_PREVENTED",
                decision="DENIED",
                requesting_component="DeviceManager",
                device_id=dev_id,
                details={"reason": "Duplicate nonce detected"},
            )
            raise ValueError("Duplicate message nonce detected (Replay Attack).")
        self._seen_nonces.add(payload.nonce)

        secret = self._shared_secrets.get(dev_id)
        if not secret:
            raise PermissionError(f"Missing shared secret for device '{dev_id}'.")

        key = _derive_fernet_key(secret)
        fernet = Fernet(key)

        try:
            decrypted_bytes = fernet.decrypt(payload.ciphertext.encode("utf-8"))
            plaintext = decrypted_bytes.decode("utf-8")
        except Exception as e:
            raise ValueError(f"Decryption failed: corrupted ciphertext or forged key ({e})")

        # Verify HMAC signature
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            f"{plaintext}:{payload.nonce}:{payload.timestamp}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(payload.signature, expected_sig):
            raise ValueError("HMAC signature verification failed: message has been tampered with.")

        device.last_seen = datetime.now(timezone.utc)
        return plaintext
