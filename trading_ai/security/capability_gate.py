"""Capability Policy Gate: Enforces least-privilege capability permissions on host PC."""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Set
import uuid

from trading_ai.schemas.enums import CapabilityState, PCCapability
from trading_ai.schemas.security import CapabilityApproval, CapabilityRequest, PCCapabilityRule
from trading_ai.security.audit import SecurityAuditLogger
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)

# Explicit list of inherently dangerous capabilities that strictly require user approval
DANGEROUS_CAPABILITIES: Set[PCCapability] = {
    PCCapability.FILE_WRITE,
    PCCapability.TERMINAL_EXECUTE,
    PCCapability.SYSTEM_SETTINGS,
}


class CapabilityPolicyGate:
    """
    Deterministic capability-based permission manager.

    Core Principles:
      - Default DENY: all capabilities start in DENIED state.
      - Least-privilege: AI agents and background workers possess no inherent OS privilege.
      - User Approval Gate: Dangerous capabilities can NEVER be self-approved by AI.
      - Audit trail on every permission check, request, grant, and revocation.
    """

    def __init__(self, repository: Optional[Repository] = None, audit_logger: Optional[SecurityAuditLogger] = None):
        self._repo = repository
        self._audit = audit_logger or SecurityAuditLogger(repository)
        self._rules: Dict[PCCapability, PCCapabilityRule] = {}
        self._pending_requests: Dict[str, CapabilityRequest] = {}
        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Initialize all PC capabilities to strict default DENY."""
        for cap in PCCapability:
            is_dangerous = cap in DANGEROUS_CAPABILITIES
            self._rules[cap] = PCCapabilityRule(
                capability=cap,
                state=CapabilityState.DENIED,
                requires_user_approval=True,
                is_dangerous=is_dangerous,
                reason="Default least-privilege security policy",
            )

    def get_rule(self, capability: PCCapability) -> PCCapabilityRule:
        """Return the current rule configuration for a capability."""
        return self._rules.get(
            capability,
            PCCapabilityRule(capability=capability, state=CapabilityState.DENIED, requires_user_approval=True),
        )

    def list_rules(self) -> List[PCCapabilityRule]:
        """Return all active capability policies."""
        return list(self._rules.values())

    def request_capability(
        self,
        capability: PCCapability,
        requesting_component: str,
        context: Optional[Dict] = None,
    ) -> CapabilityRequest:
        """
        AI or subsystem submits a formal request to invoke a PC capability.
        """
        req = CapabilityRequest(
            request_id=f"capreq_{uuid.uuid4().hex[:8]}",
            capability=capability,
            requesting_component=requesting_component,
            context=context or {},
        )
        self._pending_requests[req.request_id] = req

        self._audit.log(
            action="CAPABILITY_REQUESTED",
            decision="PENDING",
            requesting_component=requesting_component,
            capability=capability.value,
            details={"request_id": req.request_id, "is_dangerous": capability in DANGEROUS_CAPABILITIES},
        )
        return req

    def authorize_capability(self, approval: CapabilityApproval) -> PCCapabilityRule:
        """
        User explicitly approves or denies a capability request.
        Strict verification: AI/automated components CANNOT authorize dangerous capabilities.
        """
        cap = approval.capability
        rule = self.get_rule(cap)

        if rule.is_dangerous and not approval.is_user:
            self._audit.log(
                action="PRIVILEGE_ESCALATION_BLOCKED",
                decision="DENIED",
                requesting_component=approval.approved_by or "automated_agent",
                capability=cap.value,
                is_user_authorized=False,
                details={"reason": "Dangerous capability cannot be self-approved by AI"},
            )
            raise PermissionError(f"Capability '{cap.value}' is dangerous and strictly requires human user authorization.")

        if approval.approved:
            rule.state = CapabilityState.APPROVED_ALWAYS
            rule.approved_by = approval.approved_by or "user"
            rule.reason = approval.reason or "Explicit user authorization"
            rule.last_updated = datetime.now(timezone.utc)
            decision = "ALLOWED"
        else:
            rule.state = CapabilityState.DENIED
            rule.reason = approval.reason or "Explicit user denial"
            rule.last_updated = datetime.now(timezone.utc)
            decision = "DENIED"

        self._rules[cap] = rule
        if approval.request_id in self._pending_requests:
            del self._pending_requests[approval.request_id]

        self._audit.log(
            action="CAPABILITY_AUTHORIZED" if approval.approved else "CAPABILITY_DENIED",
            decision=decision,
            requesting_component="UserApprovalGate",
            capability=cap.value,
            is_user_authorized=approval.is_user,
            authorized_by=approval.approved_by,
            details={"approved": approval.approved, "state": rule.state.value},
        )
        return rule

    def revoke_capability(self, capability: PCCapability, revoked_by: str = "user") -> PCCapabilityRule:
        """Immediately revoke a capability, returning it to DENIED state."""
        rule = self.get_rule(capability)
        rule.state = CapabilityState.REVOKED
        rule.approved_by = None
        rule.reason = f"Revoked by {revoked_by}"
        rule.last_updated = datetime.now(timezone.utc)
        self._rules[capability] = rule

        self._audit.log(
            action="CAPABILITY_REVOKED",
            decision="REVOKED",
            requesting_component=revoked_by,
            capability=capability.value,
            is_user_authorized=True,
            authorized_by=revoked_by,
        )
        return rule

    def check_permission(
        self,
        capability: PCCapability,
        requesting_component: str,
    ) -> bool:
        """
        Deterministic capability check:
        AI REQUEST -> POLICY CHECK -> RISK CLASSIFICATION -> USER APPROVAL -> EXECUTION
        """
        rule = self.get_rule(capability)
        is_allowed = rule.state in [CapabilityState.APPROVED_ALWAYS, CapabilityState.APPROVED_ONCE]

        self._audit.log(
            action="CAPABILITY_CHECK",
            decision="ALLOWED" if is_allowed else "DENIED",
            requesting_component=requesting_component,
            capability=capability.value,
            details={"state": rule.state.value, "is_dangerous": rule.is_dangerous},
        )
        return is_allowed

    def execute_with_capability(
        self,
        capability: PCCapability,
        requesting_component: str,
        action_fn: callable,
        *args,
        **kwargs,
    ):
        """
        Execute an action only if capability permission is currently active.
        Raises PermissionError if not authorized.
        """
        if not self.check_permission(capability, requesting_component):
            raise PermissionError(
                f"Permission denied: Capability '{capability.value}' is not authorized for '{requesting_component}'."
            )
        return action_fn(*args, **kwargs)
