"""Test asynchronous PostgreSQL interfaces."""

import os
import pytest
import asyncio
from datetime import datetime
from typing import Optional, List
from psycopg.types.json import Jsonb

from reson.data.postgres.models import (
    DBModel,
    JoinTableDBModel,
    Column,
    PreloadAttribute,
    ModelRegistry,
)
from reson.data.postgres.manager import DatabaseManager


# Override db_manager to use test DSN
def get_test_db_manager():
    return DatabaseManager(dsn=get_test_dsn())


# Test models
class AsyncUser(DBModel):
    TABLE_NAME = "users"
    COLUMNS = {
        "id": Column("id", "id", int),
        "name": Column("name", "name", str),
        "email": Column("email", "email", str),
        "created_at": Column("created_at", "createdat", datetime),
        "metadata": Column("metadata", "metadata", Jsonb, nullable=True),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        name: str = "",
        email: str = "",
        created_at: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ):
        self.id = id
        self.name = name
        self.email = email
        self.created_at = created_at or datetime.now()
        self.metadata = metadata

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(reverse_fk="user_id", references="posts", model="AsyncPost")
    async def posts(self) -> List["AsyncPost"]:
        """Lazy load posts asynchronously."""
        return await AsyncPost.list(where="user_id = %s", params=(self.id,))


class AsyncPost(DBModel):
    TABLE_NAME = "posts"
    COLUMNS = {
        "id": Column("id", "id", int),
        "user_id": Column("user_id", "user_id", int),
        "title": Column("title", "title", str),
        "content": Column("content", "content", str),
        "created_at": Column("created_at", "createdat", datetime),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        user_id: Optional[int] = None,
        title: str = "",
        content: str = "",
        created_at: Optional[datetime] = None,
    ):
        self.id = id
        self.user_id = user_id
        self.title = title
        self.content = content
        self.created_at = created_at or datetime.now()

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(foreign_key="user_id", references="users", model="AsyncUser")
    async def user(self) -> Optional[AsyncUser]:
        """Lazy load user asynchronously."""
        if self.user_id:
            return await AsyncUser.get(self.user_id)
        return None


class AsyncUserRole(JoinTableDBModel):
    TABLE_NAME = "user_roles"
    PRIMARY_KEY_DB_NAMES = ["user_id", "role_id"]
    COLUMNS = {
        "user_id": Column("user_id", "user_id", int),
        "role_id": Column("role_id", "role_id", int),
        "assigned_at": Column("assigned_at", "assignedat", datetime),
    }

    def __init__(
        self,
        user_id: Optional[int] = None,
        role_id: Optional[int] = None,
        assigned_at: Optional[datetime] = None,
    ):
        self.user_id = user_id
        self.role_id = role_id
        self.assigned_at = assigned_at or datetime.now()

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()


TEST_SCHEMA = "test_async_postgres_temp"


def get_test_dsn():
    """Get DSN with test schema."""
    base_dsn = os.environ.get(
        "POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    # Parse and add schema option - URL encode the = sign
    from urllib.parse import quote

    options = quote(f"-csearch_path={TEST_SCHEMA}")
    if "?" in base_dsn:
        return f"{base_dsn}&options={options}"
    else:
        return f"{base_dsn}?options={options}"


def register_test_models():
    """Register test models with the registry."""
    registry = ModelRegistry()
    # Register if not already registered
    if "AsyncUser" not in registry.models:
        registry.register_model("AsyncUser", AsyncUser)
    if "AsyncPost" not in registry.models:
        registry.register_model("AsyncPost", AsyncPost)


async def setup_test_schema():
    """Create test schema and tables asynchronously."""
    # Ensure models are registered
    register_test_models()

    # First connect without schema to create it
    base_dsn = os.environ.get(
        "POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    db = DatabaseManager(dsn=base_dsn)

    async with db.get_connection() as conn:
        async with conn.cursor() as cur:
            # Drop schema if exists and create fresh
            await cur.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
            await cur.execute(f"CREATE SCHEMA {TEST_SCHEMA}")
            await conn.commit()

    # Now connect with the schema in search path
    db = DatabaseManager(dsn=get_test_dsn())

    async with db.get_connection() as conn:
        async with conn.cursor() as cur:
            # Create users table
            await cur.execute(
                """
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    createdat TIMESTAMP DEFAULT NOW(),
                    metadata JSONB
                )
            """
            )

            # Create posts table
            await cur.execute(
                """
                CREATE TABLE posts (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    title VARCHAR(255) NOT NULL,
                    content TEXT,
                    createdat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create user_roles join table
            await cur.execute(
                """
                CREATE TABLE user_roles (
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    role_id INTEGER NOT NULL,
                    assignedat TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (user_id, role_id)
                )
            """
            )

            # Create sequences for the tables
            await cur.execute(f"CREATE SEQUENCE IF NOT EXISTS users_seq")
            await cur.execute(f"CREATE SEQUENCE IF NOT EXISTS posts_seq")

            await conn.commit()


async def cleanup_test_schema():
    """Drop the test schema asynchronously."""
    base_dsn = os.environ.get(
        "POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    db = DatabaseManager(dsn=base_dsn)

    async with db.get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
            await conn.commit()


async def setup_test_tables():
    """Set up test tables in the database asynchronously."""
    # Just clear tables, schema already created
    db = DatabaseManager(dsn=get_test_dsn())

    # Clear tables using asynchronous connection
    async with db.get_connection() as conn:
        async with conn.cursor() as cur:
            # Drop tables if they exist
            await cur.execute("DROP TABLE IF EXISTS user_roles CASCADE")
            await cur.execute("DROP TABLE IF EXISTS posts CASCADE")
            await cur.execute("DROP TABLE IF EXISTS users CASCADE")

            # Recreate tables
            await cur.execute(
                """
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    createdat TIMESTAMP DEFAULT NOW(),
                    metadata JSONB
                )
            """
            )

            await cur.execute(
                """
                CREATE TABLE posts (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    title VARCHAR(255) NOT NULL,
                    content TEXT,
                    createdat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            await cur.execute(
                """
                CREATE TABLE user_roles (
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    role_id INTEGER NOT NULL,
                    assignedat TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (user_id, role_id)
                )
            """
            )

            await conn.commit()


@pytest.mark.asyncio
async def test_async_crud_operations():
    """Test asynchronous CRUD operations."""
    await setup_test_schema()  # Create the schema first
    await setup_test_tables()

    # Test insert
    user = AsyncUser(name="John Doe", email="john@example.com", metadata={"age": 30})
    await user.persist()
    assert user.id is not None

    # Test get
    fetched_user = await AsyncUser.get(user.id)
    assert fetched_user is not None
    assert fetched_user.name == "John Doe"
    assert fetched_user.email == "john@example.com"
    assert fetched_user.metadata == {"age": 30}

    # Test update
    fetched_user.name = "Jane Doe"
    fetched_user.metadata = {"age": 31, "city": "NYC"}
    await fetched_user.update()

    updated_user = await AsyncUser.get(user.id)
    assert updated_user is not None
    assert updated_user.name == "Jane Doe"
    assert updated_user.metadata == {"age": 31, "city": "NYC"}

    # Test list
    user2 = AsyncUser(name="Bob Smith", email="bob@example.com")
    await user2.persist()

    users = await AsyncUser.list(order="id")
    assert len(users) >= 2

    # Test find_by
    found_user = await AsyncUser.find_by(email="bob@example.com")
    assert found_user is not None
    assert found_user.name == "Bob Smith"

    # Test delete
    await AsyncUser.delete(user.id)
    try:
        await AsyncUser.get(user.id)
        assert False, "Should have raised ValueError for deleted record"
    except ValueError:
        pass  # Expected

    # Test delete_many
    user3 = AsyncUser(name="Alice", email="alice@example.com")
    await user3.persist()

    ids_to_delete = [user2.id, user3.id]
    await AsyncUser.delete_many([id for id in ids_to_delete if id is not None])
    remaining_users = await AsyncUser.list()
    assert len(remaining_users) == 0


@pytest.mark.asyncio
async def test_async_batch_operations():
    """Test asynchronous batch operations."""
    await setup_test_schema()  # Create the schema first
    await setup_test_tables()

    # Test insert_many
    users = [
        AsyncUser(name="User1", email="user1@example.com"),
        AsyncUser(name="User2", email="user2@example.com"),
        AsyncUser(name="User3", email="user3@example.com"),
    ]
    await AsyncUser.insert_many(users)

    # Verify they were inserted
    all_users = await AsyncUser.list(order="name")
    assert len(all_users) == 3
    assert all_users[0].name == "User1"
    assert all_users[2].name == "User3"

    # Test get_many
    user_ids = [u.id for u in all_users[:2]]
    fetched_users = await AsyncUser.get_many(user_ids)
    assert len(fetched_users) == 2


@pytest.mark.asyncio
async def test_async_relationships():
    """Test asynchronous relationship loading with preloading."""
    await setup_test_schema()  # Create the schema first
    await setup_test_tables()

    # Create user and posts
    user = AsyncUser(name="Author", email="author@example.com")
    await user.persist()

    post1 = AsyncPost(user_id=user.id, title="Post 1", content="Content 1")
    await post1.persist()

    post2 = AsyncPost(user_id=user.id, title="Post 2", content="Content 2")
    await post2.persist()

    # Test relationship loading with preload
    fetched_post = await AsyncPost.get_with_preload(post1.id, preload=["user"])
    assert fetched_post is not None

    # Debug: Check if the preloaded attribute exists
    preloaded_attr = f"_preloaded_user"
    print(f"DEBUG: Has {preloaded_attr}? {hasattr(fetched_post, preloaded_attr)}")
    if hasattr(fetched_post, preloaded_attr):
        print(f"DEBUG: Value: {getattr(fetched_post, preloaded_attr)}")

    author = fetched_post.user  # No await - preloaded data
    print(f"DEBUG: author = {author}")
    assert author is not None
    assert author.name == "Author"

    fetched_user = await AsyncUser.get_with_preload(user.id, preload=["posts"])
    assert fetched_user is not None
    posts = fetched_user.posts  # No await - preloaded data
    assert len(posts) == 2
    assert posts[0].title in ["Post 1", "Post 2"]


@pytest.mark.asyncio
async def test_async_preloading():
    """Test asynchronous preloading."""
    await setup_test_schema()  # Create the schema first
    await setup_test_tables()

    # Create test data
    user = AsyncUser(name="Preload Test", email="preload@example.com")
    await user.persist()

    post1 = AsyncPost(user_id=user.id, title="Preload 1", content="Content")
    await post1.persist()
    post2 = AsyncPost(user_id=user.id, title="Preload 2", content="Content")
    await post2.persist()

    # Test get_with_preload
    preloaded_user = await AsyncUser.get_with_preload(user.id, preload=["posts"])
    assert preloaded_user is not None

    # Access posts without additional queries - preloaded data is returned directly
    posts = preloaded_user.posts  # No await needed for preloaded data
    assert len(posts) == 2

    # Test list_with_preload
    posts = await AsyncPost.list_with_preload(preload=["user"], order="id")
    assert len(posts) == 2
    for post in posts:
        post_user = post.user  # No await needed for preloaded data
        assert post_user is not None
        assert post_user.name == "Preload Test"

    # Test find_by_with_preload
    post = await AsyncPost.find_by_with_preload(preload=["user"], title="Preload 1")
    assert post is not None
    post_user = post.user  # No await needed for preloaded data
    assert post_user.email == "preload@example.com"

    # Test error when accessing non-preloaded attribute
    non_preloaded_user = await AsyncUser.get(user.id)
    try:
        _ = non_preloaded_user.posts  # Should raise AttributeError
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "was not preloaded" in str(e)


@pytest.mark.asyncio
async def test_async_join_table():
    """Test asynchronous join table operations."""
    await setup_test_schema()  # Create the schema first
    await setup_test_tables()

    # Create user
    user = AsyncUser(name="Role User", email="roles@example.com")
    await user.persist()

    # Create user roles
    role1 = AsyncUserRole(user_id=user.id, role_id=1)
    await role1.persist()

    role2 = AsyncUserRole(user_id=user.id, role_id=2)
    await role2.persist()

    # Test get_by_pks
    fetched_role = await AsyncUserRole.get_by_pks(user.id, 1)
    assert fetched_role is not None
    assert fetched_role.role_id == 1

    # Test delete_by_pks
    await AsyncUserRole.delete_by_pks(user.id, 2)
    deleted_role = await AsyncUserRole.get_by_pks(user.id, 2)
    assert deleted_role is None

    # Verify first role still exists
    remaining_role = await AsyncUserRole.get_by_pks(user.id, 1)
    assert remaining_role is not None


@pytest.mark.asyncio
async def test_async_serialization():
    """Test asynchronous JSON serialization."""
    await setup_test_schema()  # Create the schema first
    await setup_test_tables()

    # Create user with metadata
    user = AsyncUser(
        name="JSON Test",
        email="json@example.com",
        metadata={"tags": ["test", "async"], "active": True},
    )
    await user.persist()

    # Test to_json_dict
    json_dict = await user.to_json_dict()
    assert json_dict["name"] == "JSON Test"
    assert json_dict["email"] == "json@example.com"
    assert json_dict["metadata"] == {"tags": ["test", "async"], "active": True}
    assert "createdat" in json_dict  # Column name is createdat, not created_at


@pytest.mark.asyncio
async def test_async_persist_with_id():
    """Test asynchronous persist_with_id."""
    await setup_test_schema()  # Create the schema first
    await setup_test_tables()

    # Create user with specific ID
    user = AsyncUser(id=999, name="Fixed ID", email="fixed@example.com")
    await user.persist_with_id()

    # Verify it was created with that ID
    fetched = await AsyncUser.get(999)
    assert fetched is not None
    assert fetched.id == 999
    assert fetched.name == "Fixed ID"


@pytest.mark.asyncio
async def test_async_transaction():
    """Test asynchronous transactions."""
    await setup_test_schema()  # Create the schema first
    await setup_test_tables()

    db = AsyncUser.db_manager()

    # Test rollback on error
    try:
        async with db.get_cursor() as cur:
            user1 = AsyncUser(name="TX User 1", email="tx1@example.com")
            await user1.persist(cursor=cur)

            # This should fail due to duplicate email
            user2 = AsyncUser(name="TX User 2", email="tx1@example.com")
            await user2.persist(cursor=cur)
    except Exception:
        pass  # Expected

    # Verify rollback worked - no users should exist
    users = await AsyncUser.list()
    assert len(users) == 0

    # Test successful transaction
    async with db.get_cursor() as cur:
        user1 = AsyncUser(name="TX Success 1", email="success1@example.com")
        await user1.persist(cursor=cur)

        user2 = AsyncUser(name="TX Success 2", email="success2@example.com")
        await user2.persist(cursor=cur)

    # Verify commit worked
    users = await AsyncUser.list(order="name")
    assert len(users) == 2
    assert users[0].name == "TX Success 1"
    assert users[1].name == "TX Success 2"


async def main():
    """Run all async tests."""
    try:
        # Setup test schema
        print(f"Creating test schema '{TEST_SCHEMA}'...")
        await setup_test_schema()
        print(f"✓ Test schema created\n")

        # Run tests
        print("Testing async CRUD operations...")
        await test_async_crud_operations()
        print("✓ CRUD operations passed")

        print("Testing async batch operations...")
        await test_async_batch_operations()
        print("✓ Batch operations passed")

        print("Testing async relationships...")
        await test_async_relationships()
        print("✓ Relationships passed")

        print("Testing async preloading...")
        await test_async_preloading()
        print("✓ Preloading passed")

        print("Testing async join table...")
        await test_async_join_table()
        print("✓ Join table passed")

        print("Testing async serialization...")
        await test_async_serialization()
        print("✓ Serialization passed")

        print("Testing async persist with ID...")
        await test_async_persist_with_id()
        print("✓ Persist with ID passed")

        print("Testing async transactions...")
        await test_async_transaction()
        print("✓ Transactions passed")

        print("\n✅ All asynchronous tests passed!")

    finally:
        # Cleanup
        print(f"\nCleaning up test schema '{TEST_SCHEMA}'...")
        await cleanup_test_schema()
        print("✓ Test schema dropped")


if __name__ == "__main__":
    asyncio.run(main())
