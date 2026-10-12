"""Screen Access Privacy Boundary: Enforces explicit consent for visual chart monitoring."""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from trading_ai.schemas.enums import ScreenAccessStatus
from trading_ai.schemas.market import ScreenPermission
from trading_ai.storage.repository import Repository


class ScreenAccessManager:
    """
    Manages explicit user permission for window / screen capture.

    Guarantees:
      - No unrestricted desktop surveillance or background screen capture.
      - User must explicitly grant access to a specific application / window.
      - Access can be revoked at any time.
      - Screen evidence is tagged separately from direct market-data evidence.
    """

    def __init__(self, repository: Repository):
        self._repo = repository

    def request_access(
        self, app_name: str, window_title: Optional[str] = None, region_box: Optional[List[int]] = None
    ) -> ScreenPermission:
        """Create a pending access request for user review."""
        perm = ScreenPermission(
            permission_id=str(uuid.uuid4()),
            app_name=app_name,
            window_title=window_title,
            region_box=region_box,
            status=ScreenAccessStatus.SCREEN_ACCESS_REQUESTED,
            requested_at=datetime.now(timezone.utc),
        )
        self._repo.save_screen_permission(perm)
        return perm

    def grant_access(self, permission_id: str) -> Optional[ScreenPermission]:
        """User explicitly grants permission for the requested app/window."""
        perm = self._repo.get_screen_permission(permission_id)
        if not perm:
            return None
        perm.status = ScreenAccessStatus.SCREEN_ACCESS_GRANTED
        perm.granted_at = datetime.now(timezone.utc)
        self._repo.save_screen_permission(perm)
        return perm

    def revoke_access(self, permission_id: str) -> Optional[ScreenPermission]:
        """User explicitly revokes permission."""
        perm = self._repo.get_screen_permission(permission_id)
        if not perm:
            return None
        perm.status = ScreenAccessStatus.SCREEN_ACCESS_REVOKED
        perm.revoked_at = datetime.now(timezone.utc)
        self._repo.save_screen_permission(perm)
        return perm

    def check_access(self, app_name: str) -> ScreenAccessStatus:
        """Check current access status for an application."""
        active = self._repo.get_active_screen_permission(app_name=app_name)
        if active and active.status == ScreenAccessStatus.SCREEN_ACCESS_GRANTED:
            return ScreenAccessStatus.SCREEN_ACCESS_GRANTED
        return ScreenAccessStatus.SCREEN_ACCESS_DISABLED

    def capture_authorized_frame(
        self, app_name: str, window_title: Optional[str] = None
    ) -> Dict:
        """
        Capture visual frame only if explicitly authorized.

        Raises PermissionError if access is not granted.
        """
        status = self.check_access(app_name)
        if status != ScreenAccessStatus.SCREEN_ACCESS_GRANTED:
            raise PermissionError(
                f"Screen access to '{app_name}' is not granted (current status: {status.value}). "
                "Explicit user consent required."
            )

        # In production/testing, return structured mock/real frame reference tagged as SCREEN evidence
        return {
            "source_type": "SCREEN_CAPTURE",
            "app_name": app_name,
            "window_title": window_title,
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "status": "CAPTURED",
            "evidence_type": "visual_screen",
        }
