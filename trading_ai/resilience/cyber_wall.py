"""Cyber Wall & Directory Sandbox Boundary."""
import os
import re
from pathlib import Path
from typing import List, Optional, Set

from trading_ai.schemas.enums import ThreatLevel
from trading_ai.schemas.recovery import CyberWallValidationResponse


class CyberWallSandbox:
    """Enforces strict path allowlisting, traversal prevention, and filesystem boundaries."""

    DANGEROUS_EXTENSIONS: Set[str] = {
        ".exe", ".bat", ".cmd", ".ps1", ".vbs", ".sh", ".com", ".scr", ".pif",
        ".dll", ".so", ".dylib", ".msi", ".jar", ".app", ".reg"
    }

    SENSITIVE_FILES_OR_DIRS: Set[str] = {
        ".env", ".git", ".ssh", "id_rsa", "id_ed25519", "shadow", "passwd",
        "system32", "syswow64", "windows", "program files"
    }

    def __init__(self, workspace_root: Optional[str] = None, extra_allowed_roots: Optional[List[str]] = None):
        self.workspace_root = os.path.realpath(workspace_root or os.getcwd())
        self.allowed_roots: List[str] = [
            os.path.realpath(os.path.join(self.workspace_root, "data")),
            os.path.realpath(os.path.join(self.workspace_root, "logs")),
            os.path.realpath(os.path.join(self.workspace_root, "backups")),
            os.path.realpath(os.path.join(self.workspace_root, "tests", "fixtures")),
        ]
        if extra_allowed_roots:
            for r in extra_allowed_roots:
                self.allowed_roots.append(os.path.realpath(r))

        # Ensure core sandbox directories exist
        for root in self.allowed_roots:
            os.makedirs(root, exist_ok=True)

    def validate_path(self, target_path: str, action: str = "READ") -> CyberWallValidationResponse:
        """Validate a target path against directory allowlist, traversal, and dangerous extension policies."""
        if not target_path or not isinstance(target_path, str):
            return CyberWallValidationResponse(
                allowed=False,
                canonical_path="",
                threat_level=ThreatLevel.BLOCKED,
                reason="Invalid empty or non-string path.",
            )

        # 1. Path traversal detection: check for raw traversal sequences
        if ".." in target_path or re.search(r"\.\.[/\\]", target_path):
            canonical = os.path.realpath(os.path.abspath(target_path))
            # Even if realpath was resolved, if user sent traversal pattern trying to escape, block!
            if not self._is_within_allowed_roots(canonical):
                return CyberWallValidationResponse(
                    allowed=False,
                    canonical_path=canonical,
                    threat_level=ThreatLevel.CRITICAL,
                    reason=f"Path traversal detected: attempt to escape sandbox boundary via '{target_path}'.",
                )

        try:
            canonical = os.path.realpath(os.path.abspath(target_path))
        except Exception as e:
            return CyberWallValidationResponse(
                allowed=False,
                canonical_path="",
                threat_level=ThreatLevel.BLOCKED,
                reason=f"Failed to canonicalize path: {str(e)}",
            )

        # 2. Check sensitive system files/directories
        lower_canonical = canonical.lower()
        for sensitive in self.SENSITIVE_FILES_OR_DIRS:
            if sensitive in lower_canonical:
                # Allow only if it is genuinely within allowed roots and not an env/private key file
                if sensitive in (".env", ".ssh", "id_rsa", "system32"):
                    return CyberWallValidationResponse(
                        allowed=False,
                        canonical_path=canonical,
                        threat_level=ThreatLevel.CRITICAL,
                        reason=f"Access to sensitive file or system resource '{sensitive}' is strictly prohibited.",
                    )

        # 3. Check dangerous executable file extensions for writes or executions
        _, ext = os.path.splitext(canonical)
        ext_lower = ext.lower()
        if ext_lower in self.DANGEROUS_EXTENSIONS:
            if action.upper() in ("WRITE", "EXECUTE", "CREATE"):
                return CyberWallValidationResponse(
                    allowed=False,
                    canonical_path=canonical,
                    threat_level=ThreatLevel.CRITICAL,
                    reason=f"Dangerous executable script extension '{ext_lower}' is blocked by Cyber Wall.",
                )

        # 4. Enforce directory allowlist
        if not self._is_within_allowed_roots(canonical):
            return CyberWallValidationResponse(
                allowed=False,
                canonical_path=canonical,
                threat_level=ThreatLevel.BLOCKED,
                reason=f"Path '{canonical}' is outside approved sandbox roots.",
            )

        return CyberWallValidationResponse(
            allowed=True,
            canonical_path=canonical,
            threat_level=ThreatLevel.NORMAL,
            reason="Path is authorized within Cyber Wall sandbox.",
        )

    def _is_within_allowed_roots(self, canonical_path: str) -> bool:
        """Check if canonical path starts with any allowed root path."""
        for root in self.allowed_roots:
            # Check exact match or subpath with separator
            if canonical_path == root or canonical_path.startswith(root + os.sep):
                return True
        return False
