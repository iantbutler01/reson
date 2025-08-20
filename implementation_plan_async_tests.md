# Implementation Plan

## Overview
Create comprehensive asynchronous tests for PostgreSQL database models that mirror the existing synchronous tests.

This implementation will create a new test file `test_async_postgres.py` that provides complete async test coverage for all database operations, including CRUD operations, batch operations, relationships, preloading, join tables, serialization, and transactions. The async tests will use the same schema isolation approach as the sync tests, creating a temporary test schema for each test run to ensure test isolation and repeatability.

## Types
Define async test model classes with LazyAttribute relationships and proper async method overrides.

The test models will include:
- **AsyncUser**: DBModel with id, name, email, created_at, metadata fields and a lazy-loaded posts relationship
- **AsyncPost**: DBModel with id, user_id, title, content, created_at fields and a lazy-loaded user relationship  
- **AsyncUserRole**: JoinTableDBModel for many-to-many relationship with composite primary key (user_id, role_id)

Each model will override the `db_manager()` class method to use a test DSN with the temporary schema. LazyAttribute decorators will be configured with proper foreign key and reverse foreign key relationships.

## Files
Test file structure and supporting configuration.

### New Files
- `integration_tests/test_async_postgres.py` - Main async test file containing all test functions and model definitions
  - Purpose: Comprehensive async testing of all database operations
  - Structure: Test models at top, schema setup/teardown functions, individual test functions

### Modified Files
None - This is a new test file that doesn't modify existing code

### Configuration Files
No configuration changes needed - pytest-asyncio is already configured in pyproject.toml

## Functions
Async test functions and helper functions for schema management.

### Schema Management Functions
- `get_test_dsn()` - Generate DSN with test schema in search path
- `async def setup_test_schema()` - Create test schema and all tables using async connection
- `async def cleanup_test_schema()` - Drop test schema cascade using async connection
- `async def setup_test_tables()` - Clear and recreate tables within schema

### Test Functions (all async)
- `async def test_async_crud_operations()` - Test insert, get, update, list, find_by, delete operations
- `async def test_async_batch_operations()` - Test insert_many, get_many batch operations
- `async def test_async_relationships()` - Test lazy loading of foreign key relationships
- `async def test_async_preloading()` - Test get_with_preload, list_with_preload, find_by_with_preload
- `async def test_async_join_table()` - Test composite key operations with get_by_pks, delete_by_pks
- `async def test_async_serialization()` - Test to_json_dict async serialization
- `async def test_async_persist_with_id()` - Test persisting with specific ID
- `async def test_async_transaction()` - Test transaction rollback and commit behavior

### Helper Methods
- `get_test_db_manager()` - Return DatabaseManager instance with test DSN

## Classes
Test model classes inheriting from DBModel and JoinTableDBModel.

### AsyncUser(DBModel)
- TABLE_NAME = "users"
- COLUMNS: id (int), name (str), email (str), created_at (datetime), metadata (Jsonb nullable)
- LazyAttribute: posts (one-to-many relationship)
- Override: db_manager() to use test DSN
- Methods: Standard DBModel async methods inherited

### AsyncPost(DBModel)  
- TABLE_NAME = "posts"
- COLUMNS: id (int), user_id (int), title (str), content (str), created_at (datetime)
- LazyAttribute: user (many-to-one relationship)
- Override: db_manager() to use test DSN
- Methods: Standard DBModel async methods inherited

### AsyncUserRole(JoinTableDBModel)
- TABLE_NAME = "user_roles"
- PRIMARY_KEY_DB_NAMES = ["user_id", "role_id"]
- COLUMNS: user_id (int), role_id (int), assigned_at (datetime)
- Override: db_manager() to use test DSN
- Methods: Composite key methods (get_by_pks, delete_by_pks)

## Dependencies
Testing framework and async support requirements.

- **pytest-asyncio**: Already in pyproject.toml for async test support
- **psycopg[binary]**: Already installed for PostgreSQL async support
- **Python 3.7+**: Required for async/await syntax
- **PostgreSQL**: Running instance required (localhost:5432 default)

No new dependencies need to be installed.

## Testing
Test execution and validation strategy.

### Test Isolation
- Each test run creates a unique schema `test_async_postgres_temp`
- Schema is dropped CASCADE at start and end of test suite
- Tables are cleared between individual tests

### Test Coverage
- All async methods in DBModel and JoinTableDBModel
- Edge cases: null values, missing records, duplicate keys
- Transaction behavior with nested async contexts
- Preloading with complex JOINs

### Validation Approach
- Assert return values match expected types
- Verify database state after operations
- Check lazy loading triggers additional queries
- Validate transaction rollback on errors

### Running Tests
```bash
# Run all async tests
pytest integration_tests/test_async_postgres.py -v

# Run specific test
pytest integration_tests/test_async_postgres.py::test_async_crud_operations -v

# Run with coverage
pytest integration_tests/test_async_postgres.py --cov=reson.data.postgres
```

## Implementation Order
Logical sequence to minimize issues and ensure successful integration.

1. Create test file with imports and test model class definitions
2. Implement schema setup/teardown functions
3. Add test_async_crud_operations (basic operations)
4. Add test_async_batch_operations (bulk operations)
5. Add test_async_relationships (lazy loading)
6. Add test_async_preloading (JOIN queries)
7. Add test_async_join_table (composite keys)
8. Add test_async_serialization (JSON conversion)
9. Add test_async_persist_with_id (specific IDs)
10. Add test_async_transaction (transaction handling)
11. Add main block for standalone execution
12. Run full test suite and fix any issues
13. Verify all tests pass consistently
