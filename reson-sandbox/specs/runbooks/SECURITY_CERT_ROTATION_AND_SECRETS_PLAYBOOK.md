<!-- @dive-file: Security operations runbook for TLS rotation, certificate revocation, and secret handling. -->
<!-- @dive-rel: Implemented by scripts/security/rotate_tls_bundle.sh and scripts/security/revoke_client_cert.sh. -->
<!-- @dive-rel: Validated by scripts/verify_security_ops.sh gate checks. -->
# Reson Sandbox Security Operations Playbook

## Scope

This runbook defines production operations for:

- TLS certificate rotation for control-plane and node endpoints.
- Certificate revocation response for compromised client/server certs.
- Secret-management controls for auth tokens and private keys.

## Ownership

- Primary owner: Reson Sandbox platform team.
- On-call escalation: security incident response rotation.
- Review cadence: quarterly or after any security incident.

## Certificate Rotation (Planned)

1. Generate new TLS bundle using `scripts/security/rotate_tls_bundle.sh`.
2. Store generated keys/certs in managed secret storage (not git, not local disk persistence).
3. Deploy trust-chain update first:
   - add new CA to trust stores while old CA remains trusted.
4. Roll node daemons/control gateways onto new server certs in staged batches.
5. Roll clients onto new client certs.
6. Confirm handshake success and authz checks on all zones/nodes.
7. Remove old CA trust after full migration window closes.

## Certificate Revocation (Incident)

1. Identify compromised cert serial and affected tenant/workload scope.
2. Record revocation using `scripts/security/revoke_client_cert.sh`.
3. Publish revocation list to all control gateways/node daemons.
4. Force reconnect for active sessions authenticated by revoked identities.
5. Rotate any related secrets/tokens and invalidate old credentials.
6. Verify rejected auth for revoked cert and successful auth for healthy certs.
7. Document incident timeline and remediation actions.

## Secret Management Baseline

- All private keys and bearer tokens must be stored in managed secret stores.
- Secrets must be namespace-scoped per environment and least-privilege access controlled.
- No plaintext secrets in repository, CI logs, or local shell history.
- Rotation cadence:
  - auth tokens: at least every 30 days
  - TLS private keys/certs: at least every 90 days
- Emergency rotation must support immediate invalidate-and-reissue.

## Verification and Drills

- Gate 20 (`scripts/verify_security_ops.sh`) validates:
  - rotation/revocation automation scripts
  - authz enforcement unit tests
  - runbook presence and required operational content
- Execute a live rotation+revocation drill at least quarterly and archive evidence.

## Rollback Guidance

- Keep previous cert bundle available for bounded rollback window.
- If rollback is required:
  1. restore previous cert/key secrets
  2. restart affected services in controlled order
  3. verify authz + mTLS connectivity
  4. re-open incident for root-cause and corrected re-roll
