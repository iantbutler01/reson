# Reson Sandbox Envelope Compatibility Policy

## Scope

This policy governs command/event envelopes used by distributed control paths:

- `specs/schemas/control_command_envelope.v1.json`
- `specs/schemas/control_event_envelope.v1.json`

## Versioning

- Every envelope must carry `schema_version`.
- Current production schema is `v1`.
- Producers and consumers must support `N` and `N-1` schema versions during rolling upgrades.

## Compatibility Rules

- Additive, optional fields are backward compatible.
- Removing or renaming required fields is a breaking change and requires new major schema version.
- Required field type changes are breaking changes and require new major schema version.
- Unknown fields must be ignored by consumers unless explicitly reserved for strict validation.

## Deprecation Policy

- A field may be deprecated only after:
  1. replacement field is available in production,
  2. all active consumers support replacement,
  3. deprecation window has elapsed.
- Deprecation must be documented in release notes and schema changelog.

## Validation Contract

- Gate 21 (`scripts/verify_envelope_compat.sh`) enforces:
  - schema files exist and lock required fields,
  - envelope builders emit `schema_version=v1`,
  - unit tests assert required envelope fields.

## Change Control

- Any schema change requires:
  - schema file update,
  - policy/changelog update,
  - gate updates and passing CI evidence.
