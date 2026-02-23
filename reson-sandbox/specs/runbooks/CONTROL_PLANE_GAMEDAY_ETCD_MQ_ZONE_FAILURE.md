<!-- @dive-file: Operational game-day runbook for etcd quorum, MQ outage, and zone-failure drills. -->
<!-- @dive-rel: Executed by scripts/run_control_plane_gameday_drill.sh and checked by verify_operational_gameday.sh. -->
<!-- @dive-rel: Captures failure-injection expectations for distributed control-plane readiness. -->
# Control Plane Gameday: etcd + MQ + Zone Failure

Status: Locked  
Scope: Section 12 item `12.28` operational runbook + automation contract for distributed control plane failures.

## 1) Objective

Validate operator readiness and automation behavior for three high-impact distributed failure scenarios:

1. etcd quorum loss.
2. MQ (NATS/JetStream) outage.
3. Zone failure represented by loss of all node-registry entries in one zone.

## 2) Preconditions

- Docker and Docker Compose plugin are installed.
- Host ports are available:
  - `${RESON_SANDBOX_GAMEDAY_ETCD_PORT:-32379}`
  - `${RESON_SANDBOX_GAMEDAY_NATS_PORT:-14222}`
  - `${RESON_SANDBOX_GAMEDAY_NATS_MONITOR_PORT:-18222}`
- Compose fixture file exists at `deploy/gameday/docker-compose.control-plane.yml`.

## 3) Automated Drill

Run:

```bash
./scripts/run_control_plane_gameday_drill.sh --strict
```

The drill performs:

1. Boot a 3-member etcd cluster and NATS+JetStream.
2. etcd quorum-loss scenario:
   - write succeeds with healthy quorum,
   - stop two members,
   - verify write rejection on remaining member,
   - restore quorum and verify write success.
3. MQ outage scenario:
   - verify NATS health endpoint reachable,
   - stop NATS,
   - verify endpoint unavailable,
   - restart and verify healthy again.
4. Zone-failure scenario:
   - seed node-registry keys across zones,
   - delete one-zone prefix (`zone-a`) as failure simulation,
   - verify only surviving zone keys remain.

## 4) Pass Criteria

- Script exits `0`.
- Report contains `status=passed`.
- All scenario statuses are `passed`.
- Failure details are empty.

## 5) Failure Handling

- Treat any scenario failure as release-blocking for distributed HA claims.
- Attach generated report to incident/change review.
- Re-run drill after remediation before clearing incident.
