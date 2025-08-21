# Implementation Plan

## Overview
Implement support for nested preload keys like "x > y" through proper nested JOIN queries and result parsing.

The current implementation fails with nested paths like "organization > subscription" because it treats the entire string as a single attribute name instead of parsing it as a chain of relationships. We need to parse these nested paths and build appropriate nested JOIN queries to load all related data in a single database query, avoiding N+1 problems.

## Types
Utilize existing type definitions for nested path representation.

Types already defined:
- `PreloadPath = List[str]` - A parsed path like ["organization", "subscription", "plan"]
- Dictionary structures to track JOIN aliases and relationship chains

## Files
Modify models.py to add nested JOIN support in preload methods.

Single file to modify:
- `reson/data/postgres/models.py` - Update all preload methods to handle nested paths
  - `get_with_preload` (line 640) - Add nested JOIN building
  - `list_with_preload` (line 741) - Add nested JOIN building
  - `find_by_with_preload` - Add nested JOIN building
  - `sync_get_with_preload` (line 1131) - Add nested JOIN building
  - `sync_list_with_preload` (line 1212) - Add nested JOIN building
  - `sync_find_by_with_preload` - Add nested JOIN building

## Functions
Update preload functions to parse nested paths and build nested JOINs.

Key changes needed:

1. **Parse nested paths** - Before building JOINs, parse paths like "organization > subscription" into chains:
   ```python
   # Convert ["organization > subscription", "profile"] into:
   # [["organization", "subscription"], ["profile"]]
   ```

2. **Build nested JOINs** - For each chain, build nested JOINs:
   ```sql
   -- For "user > organization > subscription":
   LEFT JOIN organizations t1 ON t0.organization_id = t1.id
   LEFT JOIN subscriptions t2 ON t1.subscription_id = t2.id
   ```

3. **Track relationship chains** - Maintain mapping of:
   - Which alias corresponds to which level in the chain
   - How to parse results back into nested objects

4. **Parse nested results** - After query execution:
   - Parse flat JOIN results into nested object structure
   - Set preloaded attributes at each level of nesting

Functions to modify:
- `get_with_preload` - Parse paths, build nested JOINs, parse nested results
- `list_with_preload` - Parse paths, build nested JOINs, parse nested results
- `find_by_with_preload` - Parse paths, build nested JOINs, parse nested results
- `sync_get_with_preload` - Sync version of above
- `sync_list_with_preload` - Sync version of above
- `sync_find_by_with_preload` - Sync version of above

## Classes
No new classes needed - modify existing DBModel methods.

The implementation will modify the existing `DBModel` class methods to:
1. Parse nested preload paths
2. Build nested JOIN queries
3. Parse results to populate nested relationships

## Dependencies
No new dependencies - utilize existing psycopg and typing.

Current dependencies are sufficient:
- `psycopg` for database operations
- Python's `typing` module for type hints

## Testing
Ensure full backward compatibility while adding nested preload support.

**Critical Requirement: ALL existing tests must continue to pass unchanged**

Existing test suites that must remain passing:
- `test_postgres.py` - Basic ORM functionality tests
- `test_async_postgres.py` - Async database operations (30 tests currently passing)
- `test_sync_postgres.py` - Sync database operations
- `test_instance_preload.py` - Instance-level preloading (currently has 1 failure we need to maintain)
- `test_idempotent_preload.py` - Idempotent preloading behavior
- `test_serialization.py` - JSON serialization (currently has 2 failures we need to maintain)

New nested preload tests that should pass after implementation:
- `test_simple_nested_preload_instance` - Load user > organization > subscription in one query
- `test_deep_nested_preload_instance` - Load user > organization > projects in one query
- `test_sync_nested_preload_instance` - Sync version working
- `test_mixed_nested_and_direct_preload` - Multiple nested paths in one query
- `test_invalid_nested_path` - Graceful handling of invalid nested paths
- `test_nested_with_many_to_many` - Nested paths through many-to-many relationships
- `test_complex_mixed_relationships` - Complex multi-level nesting with mixed relationship types

Backward compatibility verification:
1. Run full test suite before changes to establish baseline
2. After implementation, verify same tests pass/fail as before
3. Simple preloads like `["organization", "profile"]` must work exactly as before
4. Only paths containing ">" should trigger new nested JOIN logic

Performance verification:
- Single SQL query with nested JOINs for nested paths
- No N+1 queries for any preload scenario
- Existing simple preloads should have same query patterns as before
- Monitor query count to ensure no regression

Test execution commands:
```bash
# Run all existing tests to ensure no regression
pytest integration_tests/test_postgres.py test_async_postgres.py test_sync_postgres.py -v

# Run nested preload tests to verify new functionality
pytest integration_tests/test_nested_preload.py -v

# Full test suite to ensure complete compatibility
pytest integration_tests/ -v
```

## Implementation Order
Build nested JOIN support incrementally.

1. **Add path parsing logic** (Priority 1)
   - Parse "x > y > z" strings into path lists
   - Handle both simple ("organization") and nested ("organization > subscription") paths

2. **Build nested JOIN queries** (Priority 2)
   - For each path chain, build appropriate nested JOINs
   - Track aliases for each level (t0, t1, t2, etc.)
   - Include columns from all joined tables

3. **Parse nested results** (Priority 3)
   - Process JOIN results to create nested object structure
   - Set _preloaded_ attributes at each nesting level
   - Handle both single instances and lists

4. **Update async methods** (Priority 4)
   - `get_with_preload` - Full nested JOIN support
   - `list_with_preload` - Full nested JOIN support
   - `find_by_with_preload` - Full nested JOIN support

5. **Update sync methods** (Priority 5)
   - `sync_get_with_preload` - Full nested JOIN support
   - `sync_list_with_preload` - Full nested JOIN support
   - `sync_find_by_with_preload` - Full nested JOIN support

6. **Test and verify** (Priority 6)
   - Run all nested preload tests
   - Verify single queries with nested JOINs
   - Confirm no N+1 queries occur
