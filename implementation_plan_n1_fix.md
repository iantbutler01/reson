# Implementation Plan: Fix N+1 Query Problem in Nested Preloading

## Overview
Refactor the `_apply_nested_preloads` and `_sync_apply_nested_preloads` methods to use batch queries instead of executing individual queries per instance.

The current implementation iterates through instances and calls `instance.preload()` for each one, resulting in N+1 queries. The fix will collect all required IDs/foreign keys and execute batch queries using WHERE IN clauses, similar to the existing `get_many` pattern.

## Types
No new type definitions required.

The existing types will be used:
- `PreloadPath = List[str]` - Already defined for nested paths
- `PreloadTree = Dict[str, Optional["PreloadTree"]]` - Already defined for recursive structure
- Standard async/sync cursors and connection types

## Files
Modified files only - no new files needed.

- **reson/data/postgres/models.py**: Modify the `_apply_nested_preloads` and `_sync_apply_nested_preloads` methods
  - Line ~270-320: Async version needs batching logic
  - Line ~323-370: Sync version needs same batching logic
  - No changes to public API signatures
  - No changes to other preload methods

## Functions
Two functions to be modified with internal implementation changes only.

### Modified Functions:

1. **`_apply_nested_preloads`** (async, classmethod)
   - File: reson/data/postgres/models.py
   - Current: Loops through instances calling instance.preload()
   - Change: Batch load relationships using WHERE IN queries
   - Signature unchanged: `async def _apply_nested_preloads(cls, instances: List[T], tree: PreloadTree, cursor: Optional[psycopg.AsyncCursor[DictRow]] = None) -> None`

2. **`_sync_apply_nested_preloads`** (sync, classmethod)
   - File: reson/data/postgres/models.py  
   - Current: Loops through instances calling instance.sync_preload()
   - Change: Batch load relationships using WHERE IN queries
   - Signature unchanged: `def _sync_apply_nested_preloads(cls, instances: List[T], tree: PreloadTree, cursor: Optional[psycopg.Cursor[DictRow]] = None) -> None`

### Helper Functions (new, private):

3. **`_batch_load_many_to_one`** (async, classmethod)
   - Collect foreign keys from all instances
   - Execute single WHERE id IN (...) query
   - Map results back to instances

4. **`_batch_load_one_to_many`** (async, classmethod)
   - Collect instance IDs
   - Execute single WHERE foreign_key IN (...) query
   - Group results by foreign key value

5. **`_batch_load_many_to_many`** (async, classmethod)
   - Collect instance IDs
   - Execute JOIN query with WHERE IN clause
   - Group results by source ID

6. **Sync versions** of the above three helper functions

## Classes
No class structure changes required.

The DBModel class will have new private helper methods added but no changes to:
- Public API
- Class hierarchy
- Existing method signatures
- PreloadAttribute descriptor class

## Dependencies
No new dependencies required.

Uses existing:
- psycopg for database operations
- Standard library collections for grouping
- Existing execute_query methods from DatabaseManager

## Testing
Enhance existing test coverage to verify N+1 fix.

### Modified Tests:
- **integration_tests/test_nested_preload.py**: Add query counting to verify batch loading
  - Wrap database calls to count executed queries
  - Assert expected query count for nested preloads
  - Test with multiple instances to verify batching

### New Test Cases:
- Test batch loading with 10+ instances
- Test empty result sets in batch queries
- Test mixed null/non-null foreign keys
- Performance comparison test (before/after query counts)
- Edge cases: single instance, no matches, all nulls

## Implementation Order
Sequential steps to minimize risk and ensure backward compatibility.

1. **Create helper methods for batch loading**
   - Implement _batch_load_many_to_one (async)
   - Implement _batch_load_one_to_many (async)
   - Implement _batch_load_many_to_many (async)

2. **Refactor _apply_nested_preloads**
   - Replace instance loop with batch collection
   - Use helper methods for each relationship type
   - Map results back to instances
   - Handle nested recursion with batched results

3. **Create sync versions of helpers**
   - Implement sync versions of batch load helpers
   - Follow same patterns as async versions

4. **Refactor _sync_apply_nested_preloads**
   - Apply same batching logic as async version
   - Use sync helper methods

5. **Add query counting to tests**
   - Create query counter context manager
   - Wrap test database operations
   - Add assertions for expected query counts

6. **Create performance test**
   - Test with 50+ instances
   - Compare query counts before/after
   - Verify sub-linear query growth

7. **Run full test suite**
   - Verify all existing tests pass
   - Check new query count assertions
   - Validate performance improvements
