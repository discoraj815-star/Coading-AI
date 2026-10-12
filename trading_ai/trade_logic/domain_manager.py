"""Physical Knowledge Source Domain Manager.

Enforces physical isolation of four distinct knowledge domains:
1. 01_YouTube_Lectures/
2. 02_Mentorship_Sessions/
3. 03_Live_Trade_Logs/
4. 04_Trading_Tech_Work/

Guarantees:
- Zero cross-directory file contamination.
- Canonical path checks (symlink / traversal protection).
- Fail-closed path isolation.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from trading_ai.schemas.enums import SourceDomain


class SourceDomainManager:
    """Manages physical directory bounds for the four isolated knowledge domains."""

    DOMAINS = [
        SourceDomain.YOUTUBE_LECTURES,
        SourceDomain.MENTORSHIP_SESSIONS,
        SourceDomain.LIVE_TRADE_LOGS,
        SourceDomain.TRADING_TECH_WORK,
    ]

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or "data/knowledge_domains").resolve()
        self._ensure_domain_directories()

    def _ensure_domain_directories(self) -> None:
        """Create the four physical domain folders if they don't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for domain in self.DOMAINS:
            domain_path = self.base_dir / domain.value
            domain_path.mkdir(parents=True, exist_ok=True)
            readme = domain_path / "README.md"
            if not readme.exists():
                readme.write_text(
                    f"# {domain.value}\n\nIsolated storage for {domain.name}.\n",
                    encoding="utf-8",
                )

    def get_domain_path(self, domain: SourceDomain) -> Path:
        """Get the absolute canonical path of a domain directory."""
        return (self.base_dir / domain.value).resolve()

    def validate_file_in_domain(self, file_path: str, domain: SourceDomain) -> Tuple[bool, str]:
        """Validate that a target file path strictly resides within the specified domain directory.

        Returns (is_valid, canonical_path_or_error_message).
        """
        try:
            target = Path(file_path).resolve()
            domain_root = self.get_domain_path(domain)

            # Check if domain_root is parent of target
            try:
                target.relative_to(domain_root)
            except ValueError:
                return (
                    False,
                    f"Access Denied: Path '{file_path}' attempts to escape isolated domain '{domain.value}'.",
                )

            # Check that it doesn't cross into another domain directory
            for other_domain in self.DOMAINS:
                if other_domain != domain:
                    other_root = self.get_domain_path(other_domain)
                    try:
                        target.relative_to(other_root)
                        return (
                            False,
                            f"Contamination Blocked: Path '{file_path}' belongs to '{other_domain.value}', not '{domain.value}'.",
                        )
                    except ValueError:
                        pass

            return True, str(target)
        except Exception as e:
            return False, f"Path validation error: {str(e)}"

    def list_domain_files(self, domain: SourceDomain) -> List[str]:
        """List all relative file paths inside an isolated domain."""
        domain_root = self.get_domain_path(domain)
        files = []
        for p in domain_root.rglob("*"):
            if p.is_file():
                files.append(str(p.relative_to(domain_root)))
        return files

    def get_domain_stats(self) -> Dict[str, Dict[str, int]]:
        """Return counts and storage stats for each of the four isolated domains."""
        stats = {}
        for domain in self.DOMAINS:
            domain_root = self.get_domain_path(domain)
            file_count = 0
            total_bytes = 0
            for p in domain_root.rglob("*"):
                if p.is_file():
                    file_count += 1
                    total_bytes += p.stat().st_size
            stats[domain.value] = {
                "file_count": file_count,
                "total_bytes": total_bytes,
            }
        return stats

    def resolve_domain_path(self, domain: SourceDomain, relative_path: str) -> Path:
        """Resolve a path within a domain, raising PermissionError if it escapes or contaminates."""
        domain_root = self.get_domain_path(domain)
        target = (domain_root / relative_path).resolve()
        is_valid, msg = self.validate_file_in_domain(str(target), domain)
        if not is_valid:
            raise PermissionError(msg)
        return target

    def save_domain_file(self, domain: SourceDomain, relative_path: str, content: str) -> str:
        """Safely write content to an isolated domain file, preventing traversal."""
        resolved = self.resolve_domain_path(domain, relative_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return str(resolved)

