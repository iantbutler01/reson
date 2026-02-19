# Control Gateway HA Topology And Failover Drill

Status: Locked  
Scope: Distributed control endpoint used by `Sandbox::connect(...)` and distributed session routing.

## 1) Topology Contract

- Client-facing endpoint is a single stable URL (DNS/LB) backed by at least 2 control gateway replicas.
- Gateways are stateless request routers that resolve node/session ownership from etcd + MQ-backed control state.
- Gateways are deployed with anti-affinity across failure domains when available.
- Gateway loss must not break active session ownership metadata.

## 2) Traffic Expectations

- New session attach/create requests may land on any healthy gateway replica.
- Gateway selection must be transparent to the facade client.
- On single-gateway failure, requests fail over to surviving replicas without API-shape changes.

## 3) Drill Objective

Validate that the control endpoint remains usable when the primary gateway path is unavailable and the client can continue through secondary candidates.

## 4) Drill Procedure (Automated)

1. Run `scripts/run_control_gateway_failover_drill.sh --strict`.
2. The drill executes facade contract test `control_gateway_failover_prefers_healthy_secondary_endpoint`.
3. On completion, the drill writes a report file with:
   - `status`
   - UTC start/end timestamps
   - selected test selector
   - failure reason (if any)

## 5) Pass Criteria

- Drill script exits `0`.
- Report file contains `status=passed`.
- Facade contract test confirms fallback to healthy secondary endpoint.

## 6) Failure Handling

- If drill fails, block rollout that depends on gateway HA claims.
- Capture report artifact and attach to incident/change review.
- Re-run drill after remediation before marking gateway profile healthy.
