# Implementation Plan: Enhanced JSON Serialization with Preloaded Relationships

## Overview
Enhance the JSON serialization methods to recursively serialize preloaded relationships while handling circular references.

The current `to_json_dict()` and `sync_to_json_dict()` methods only serialize column values from the model. This enhancement will detect and include preloaded relationships in the JSON output, omitting any relationships that haven't been loaded to avoid N+1 query issues.

## Types

### Serialization Context Types
```python
# Track visited objects to prevent circular references
VisitedSet = Set[Tuple[Type, int]]  # (model_class, object_id)

# Serialization options
class SerializationOptions:
    include_null: bool = True
    max_depth: Optional[int] = None
    visited: Optional[VisitedSet] = None
```

## Files

### Files to Modify
- **reson/data/postgres/models.py**
  - Update `to_json_dict()` method to handle preloaded relationships
  - Update `sync_to_json_dict()` method to handle preloaded relationships
  - Update `_serialize_value()` to handle circular references
  - Update `_sync_serialize_value()` to handle circular references
  - Add helper method `_get_preloaded_attributes()` to detect loaded relationships
  - Add helper method `_should_serialize_relationship()` for circular reference checking

### Files to Create
- **integration_tests/test_serialization.py**
  - Test recursive serialization with nested preloaded relationships
  - Test circular reference handling
  - Test partial preloading (some relationships loaded, others not)
  - Test both async and sync versions

## Functions

### New Functions in DBModel

#### `_get_preloaded_attributes(self) -> Dict[str, Any]`
- **Purpose**: Detect which PreloadAttribute relationships have been preloaded
- **Returns**: Dictionary mapping attribute names to their preloaded values
- **Implementation**: 
  - Iterate through class attributes to find PreloadAttribute descriptors
  - Check for `_preloaded_{attr_name}` attribute on instance
  - Return only loaded relationships

#### `_should_serialize_relationship(self, visited: Set) -> bool`
- **Purpose**: Check if this object should be serialized to prevent circular references
- **Returns**: True if object hasn't been visited, False if already serialized
- **Implementation**:
  - Create unique identifier (class, id) tuple
  - Check if tuple exists in visited set
  - Add to visited set if not present

### Modified Functions

#### `to_json_dict(self, visited: Optional[Set] = None) -> Dict[str, Any]`
- **Changes**:
  - Add optional `visited` parameter for tracking serialized objects
  - Initialize visited set if None
  - Check for circular references before serializing
  - After serializing columns, get preloaded attributes
  - Recursively serialize each preloaded relationship
  - Pass visited set to nested serializations

#### `sync_to_json_dict(self, visited: Optional[Set] = None) -> Dict[str, Any]`
- **Changes**: Same as async version but synchronous

#### `_serialize_value(self, val, visited: Set)`
- **Changes**:
  - Add visited parameter
  - Pass visited set when recursively serializing DBModel instances
  - Check for circular references before serializing nested objects

#### `_sync_serialize_value(self, val, visited: Set)`
- **Changes**: Same as async version but synchronous

## Classes

### No New Classes
The implementation uses the existing DBModel and PreloadAttribute classes.

## Dependencies

No new external dependencies required. Uses Python's built-in `set` for tracking visited objects.

## Testing

### Test Scenarios

1. **Simple Preloaded Relationship**
   - User with preloaded posts
   - Verify posts are included in JSON

2. **Nested Preloaded Relationships**
   - User > Organization > Subscription
   - Verify all levels are serialized

3. **Partial Preloading**
   - User with posts preloaded but not organization
   - Verify only posts are included, organization key is omitted

4. **Circular References**
   - User > Post > User (back reference)
   - Verify no infinite recursion

5. **Many-to-Many Relationships**
   - Project with preloaded tags
   - Verify collection serialization

6. **One-to-One Relationships**
   - User with preloaded profile
   - Verify single object serialization

7. **Empty Relationships**
   - User with no posts (empty list)
   - Verify empty list is included

8. **Null Relationships**
   - User with null organization_id
   - Verify null is handled correctly

## Implementation Order

1. **Step 1**: Add `_get_preloaded_attributes()` helper method
2. **Step 2**: Add `_should_serialize_relationship()` helper method  
3. **Step 3**: Update `to_json_dict()` to handle preloaded relationships
4. **Step 4**: Update `_serialize_value()` to pass visited set
5. **Step 5**: Update `sync_to_json_dict()` to handle preloaded relationships
6. **Step 6**: Update `_sync_serialize_value()` to pass visited set
7. **Step 7**: Create comprehensive test file
8. **Step 8**: Run tests and verify backward compatibility

## Example Usage

```python
# Load user with nested relationships
user = await User.get_with_preload(
    user_id, 
    preload=["posts", "organization > subscription"]
)

# Serialize to JSON - includes all preloaded relationships
json_data = await user.to_json_dict()
# Result:
# {
#     "id": 1,
#     "name": "Alice",
#     "email": "alice@example.com",
#     "posts": [
#         {"id": 1, "title": "Post 1", "content": "..."},
#         {"id": 2, "title": "Post 2", "content": "..."}
#     ],
#     "organization": {
#         "id": 1,
#         "name": "Tech Corp",
#         "subscription": {
#             "id": 1,
#             "plan_name": "Enterprise",
#             "status": "active"
#         }
#     }
# }

# Without preloading - relationships are omitted
user2 = await User.get(user_id)
json_data2 = await user2.to_json_dict()
# Result:
# {
#     "id": 1,
#     "name": "Alice", 
#     "email": "alice@example.com"
#     # No posts or organization keys
# }
