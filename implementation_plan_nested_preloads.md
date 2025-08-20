# Implementation Plan

Add nested preloading support to allow chained relationship loading with syntax like `preload=["organization > subscription"]`.

The implementation will enable developers to preload deeply nested relationships in a single query or set of queries, reducing the N+1 query problem for complex object graphs. This feature will support both instance-level and class-level preloading methods, maintaining consistency with the existing preloading API while extending its capabilities.

## Overview

Single sentence describing the overall goal.

Multiple paragraphs outlining the scope, context, and high-level approach. The nested preloading feature will extend the existing preloading infrastructure in reson/data/postgres/models.py to support multi-level relationship loading. The syntax `"organization > subscription"` will indicate that when loading an organization, its subscription relationship should also be preloaded. This will work recursively, allowing chains like `"user > organization > subscription > plan"`.

The implementation will need to handle both instance-level preloading (where an instance already exists and we're loading its relationships) and class-level preloading (where we're fetching instances with their relationships in optimized queries). For instance-level preloading, we'll parse the nested syntax and recursively call preload on related objects. For class-level preloading with JOINs, we'll need to build more complex queries that include multiple levels of JOINs.

The feature must maintain backward compatibility with existing preload syntax while adding the new nested capability. It should handle all relationship types (many-to-one, one-to-one, one-to-many, many-to-many) and work with both async and sync versions of all methods.

## Types

Single sentence describing the type system changes.

No new types are required, but we'll add type hints for nested preload parsing functions:

```python
from typing import List, Tuple, Dict, Any, Optional

# Type for parsed preload path
PreloadPath = List[str]  # e.g., ["organization", "subscription", "plan"]

# Type for preload tree structure
PreloadTree = Dict[str, Optional['PreloadTree']]  # Recursive structure for nested preloads
```

## Files

Single sentence describing file modifications.

Detailed breakdown:
- **Modified files:**
  - `reson/data/postgres/models.py`: Add nested preload parsing and implementation logic
  - All methods that accept `preload` parameter will be enhanced to support nested syntax
  
- **New test files:**
  - `integration_tests/test_nested_preload.py`: Comprehensive tests for nested preloading functionality

## Functions

Single sentence describing function modifications.

Detailed breakdown:

**New helper functions in DBModel class:**
- `_parse_preload_paths(preload: List[str]) -> Dict[str, List[PreloadPath]]`: Parse preload strings into structured paths
  - Splits strings like "organization > subscription" into ["organization", "subscription"]
  - Groups paths by their root attribute for efficient processing
  
- `_build_preload_tree(paths: List[PreloadPath]) -> PreloadTree`: Convert paths to tree structure
  - Transforms flat paths into nested dictionary for recursive processing
  
- `_apply_nested_preloads(instances: List[T], tree: PreloadTree, cursor: Optional[...])`: Recursively apply preloads
  - Applies preloading to a list of instances based on the preload tree
  - Handles both single instances and collections

**Modified instance-level methods:**
- `async def preload()`: Enhance to handle nested preload syntax
  - Parse nested paths before processing
  - After loading first level, recursively preload nested relationships
  
- `def sync_preload()`: Synchronous version of enhanced preload
  - Same logic as async version but using sync database operations

**Modified class-level methods:**
- `async def get_with_preload()`: Support nested preloads in single-record fetch
  - Parse nested syntax and handle multi-level JOINs where possible
  - Fall back to recursive loading for complex nested structures
  
- `async def list_with_preload()`: Support nested preloads in list fetch
  - Optimize for batch loading of nested relationships
  
- `async def find_by_with_preload()`: Support nested preloads in find operations
  
- Sync versions of all above methods

**Modified parsing methods:**
- `_parse_joined_results_single()`: Enhanced to handle nested preloaded data
- `_parse_joined_results_list()`: Enhanced to handle nested preloaded data in lists

## Classes

Single sentence describing class modifications.

Detailed breakdown:

**DBModel class enhancements:**
- No structural changes to the class itself
- Add helper methods for nested preload parsing and processing
- Maintain backward compatibility with existing preload interface

**PreloadAttribute class:**
- No modifications needed - existing structure supports nested loading
- The descriptor pattern already provides the metadata needed for traversal

## Dependencies

Single sentence describing dependency modifications.

No new dependencies required - implementation uses existing psycopg and Python standard library features.

## Testing

Single sentence describing testing approach.

Comprehensive test suite covering all nested preload scenarios:

**New test file: `integration_tests/test_nested_preload.py`**
- Test basic nested preloading: `"organization > subscription"`
- Test deep nesting: `"user > organization > subscription > plan"`
- Test multiple nested paths: `["organization > subscription", "organization > users"]`
- Test mixed nested and non-nested: `["organization > subscription", "projects"]`
- Test all relationship types in nested scenarios:
  - many-to-one chains
  - one-to-many with nested many-to-one
  - many-to-many with nested relationships
- Test error handling:
  - Invalid path syntax
  - Non-existent relationships
  - Circular references
- Test performance:
  - Verify query optimization for nested loads
  - Compare with N+1 query approach
- Test both async and sync versions

**Updates to existing tests:**
- Ensure backward compatibility - all existing tests should pass unchanged
- Add nested preload examples to existing test files where appropriate

## Implementation Order

Single sentence describing the implementation sequence.

Numbered steps showing the logical order of changes to minimize conflicts and ensure successful integration:

1. **Add parsing helper functions** (no breaking changes):
   - Implement `_parse_preload_paths()` to parse nested syntax
   - Implement `_build_preload_tree()` to create recursive structure
   - Add unit tests for parsing logic

2. **Enhance instance-level preloading** (backward compatible):
   - Modify `preload()` method to detect and handle nested syntax
   - Implement recursive preloading logic for nested paths
   - Update `sync_preload()` with same logic
   - Test with simple nested relationships

3. **Implement recursive preload application**:
   - Create `_apply_nested_preloads()` helper method
   - Handle different relationship types in nested context
   - Ensure proper error handling for missing relationships

4. **Enhance class-level preloading with basic nesting**:
   - Update `get_with_preload()` to handle one level of nesting
   - Modify JOIN query building to support nested relationships
   - Update result parsing to handle nested data

5. **Extend to list operations**:
   - Enhance `list_with_preload()` for nested preloads
   - Optimize batch loading of nested relationships
   - Handle memory efficiently for large result sets

6. **Complete sync implementations**:
   - Mirror all async enhancements in sync methods
   - Ensure consistent behavior between async and sync

7. **Add comprehensive testing**:
   - Create `test_nested_preload.py` with full test coverage
   - Test edge cases and error conditions
   - Verify performance characteristics

8. **Documentation and examples**:
   - Update docstrings with nested preload examples
   - Add usage examples to test files
   - Document performance considerations

9. **Optimization pass**:
   - Analyze query patterns for optimization opportunities
   - Implement query batching for deeply nested structures
   - Add query count assertions to tests
