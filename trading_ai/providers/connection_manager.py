"""Step 18: Universal Connection Manager with Strict Credential Masking."""
import logging
from typing import Any, Dict, List, Optional

from trading_ai.providers.registry import UniversalProviderRegistry
from trading_ai.schemas.provider import ConnectionTestResult, ProviderMetadata
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)

SECRET_KEY_SUBSTRINGS = {"key", "secret", "token", "password", "credential", "auth"}


def mask_sensitive_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mask secret strings so that credentials are NEVER leaked into frontend, logs, or DOM."""
    masked = {}
    for k, v in config.items():
        if any(sec in k.lower() for sec in SECRET_KEY_SUBSTRINGS):
            if isinstance(v, str) and len(v) > 0:
                masked[k] = f"***MASKED_LEN_{len(v)}***"
            else:
                masked[k] = "***MASKED***"
        else:
            masked[k] = v
    return masked


class UniversalConnectionManager:
    """Coordinates provider lifecycle transitions, connection testing, and secure credential handling."""

    def __init__(self, registry: UniversalProviderRegistry, repository: Optional[Repository] = None):
        self._registry = registry
        self._repo = repository

    @property
    def registry(self) -> UniversalProviderRegistry:
        return self._registry

    def test_connection(self, provider_id: str) -> ConnectionTestResult:
        """Execute a safe, non-destructive connection test with zero secret leaks."""
        return self._registry.test_provider(provider_id)

    def configure_provider(self, provider_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Safely save provider configuration and initialize provider."""
        provider = self._registry.get_provider(provider_id)
        if not provider:
            raise ValueError(f"Provider '{provider_id}' not found.")

        # Update in-memory provider
        provider.initialize(config)

        # Mask secrets before returning or storing in public config table
        masked = mask_sensitive_config(config)
        has_key = bool(config.get("api_key"))
        has_secret = bool(config.get("api_secret"))
        provider.metadata.api_key_status = "CONFIGURED" if has_key else "NOT_CONFIGURED"
        provider.metadata.secret_status = "CONFIGURED" if has_secret else "NOT_CONFIGURED"
        if has_key:
            provider.metadata.masked_api_key = masked.get("api_key", "********")
        if has_secret:
            provider.metadata.masked_api_secret = masked.get("api_secret", "********")

        if self._repo:
            self._repo.save_provider_connection_config(provider_id=provider_id, config=masked)
            self._repo.update_universal_provider_status(
                provider_id=provider_id,
                status=provider.metadata.status.value,
                trading_state=provider.metadata.trading_state.value,
            )
            try:
                self._repo.save_universal_provider(provider.metadata)
            except Exception:
                pass

        return {"status": "CONFIGURED", "success": True, "provider_id": provider_id, "masked_config": masked}

    def save_connection_config(self, provider_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for configure_provider."""
        return self.configure_provider(provider_id, config)

    def get_masked_config(self, provider_id: str) -> Dict[str, Any]:
        """Retrieve masked configuration for display in UI."""
        if self._repo:
            cfg = self._repo.get_provider_connection_config(provider_id)
            return mask_sensitive_config(cfg)
        return {}
