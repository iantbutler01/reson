"""Test synchronous PostgreSQL interfaces."""

import os
import pytest
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
class Post(DBModel):
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

    @PreloadAttribute(
        preload=True, foreign_key="user_id", references="users", model="User"
    )
    def user(self):
        """User relationship."""
        ...  # PreloadAttribute handles everything


class User(DBModel):
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

    @PreloadAttribute(
        preload=True, reverse_fk="user_id", references="posts", model="Post"
    )
    def posts(self):
        """Posts relationship."""
        ...  # PreloadAttribute handles everything


# Register models early to avoid conflicts
def register_test_models():
    """Register models for testing with unique names."""
    registry = ModelRegistry()
    # Clear any existing registrations for these models
    if "Post" in registry.models:
        del registry.models["Post"]
    if "User" in registry.models:
        del registry.models["User"]

    # Register with unique names
    registry.register_model("Post", Post)
    registry.register_model("User", User)


class UserRole(JoinTableDBModel):
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


TEST_SCHEMA = "test_sync_postgres_temp"


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


def setup_test_schema():
    """Create test schema and tables."""

    register_test_models()

    # First connect without schema to create it
    base_dsn = os.environ.get(
        "POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    db = DatabaseManager(dsn=base_dsn)

    with db.sync_get_connection() as conn:
        with conn.cursor() as cur:
            # Drop schema if exists and create fresh
            cur.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
            cur.execute(f"CREATE SCHEMA {TEST_SCHEMA}")
            conn.commit()

    # Now connect with the schema in search path
    db = DatabaseManager(dsn=get_test_dsn())

    with db.sync_get_connection() as conn:
        with conn.cursor() as cur:
            # Create users table
            cur.execute(
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
            cur.execute(
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
            cur.execute(
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
            cur.execute(f"CREATE SEQUENCE IF NOT EXISTS users_seq")
            cur.execute(f"CREATE SEQUENCE IF NOT EXISTS posts_seq")

            conn.commit()


def cleanup_test_schema():
    """Drop the test schema."""
    base_dsn = os.environ.get(
        "POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    db = DatabaseManager(dsn=base_dsn)

    with db.sync_get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
            conn.commit()


def setup_test_tables():
    """Set up test tables in the database."""
    # Just clear tables, schema already created
    db = DatabaseManager(dsn=get_test_dsn())

    # Clear tables using synchronous connection
    with db.sync_get_connection() as conn:
        with conn.cursor() as cur:
            # Drop tables if they exist
            cur.execute("DROP TABLE IF EXISTS user_roles CASCADE")
            cur.execute("DROP TABLE IF EXISTS posts CASCADE")
            cur.execute("DROP TABLE IF EXISTS users CASCADE")

            # Recreate tables
            cur.execute(
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

            cur.execute(
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

            cur.execute(
                """
                CREATE TABLE user_roles (
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    role_id INTEGER NOT NULL,
                    assignedat TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (user_id, role_id)
                )
            """
            )

            conn.commit()


def test_sync_crud_operations():
    """Test synchronous CRUD operations."""
    setup_test_schema()  # Create the schema first
    setup_test_tables()

    # Test insert
    user = User(name="John Doe", email="john@example.com", metadata={"age": 30})
    user.sync_persist()
    assert user.id is not None

    # Test get
    fetched_user = User.sync_get(user.id)
    assert fetched_user is not None
    assert fetched_user.name == "John Doe"
    assert fetched_user.email == "john@example.com"
    assert fetched_user.metadata == {"age": 30}

    # Test update
    fetched_user.name = "Jane Doe"
    fetched_user.metadata = {"age": 31, "city": "NYC"}
    fetched_user.sync_update()

    updated_user = User.sync_get(user.id)
    assert updated_user is not None
    assert updated_user.name == "Jane Doe"
    assert updated_user.metadata == {"age": 31, "city": "NYC"}

    # Test list
    user2 = User(name="Bob Smith", email="bob@example.com")
    user2.sync_persist()

    users = User.sync_list(order="id")
    assert len(users) >= 2

    # Test find_by
    found_user = User.sync_find_by(email="bob@example.com")
    assert found_user is not None
    assert found_user.name == "Bob Smith"

    # Test delete
    User.sync_delete(user.id)
    try:
        User.sync_get(user.id)
        assert False, "Should have raised ValueError for deleted record"
    except ValueError:
        pass  # Expected

    # Test delete_many
    user3 = User(name="Alice", email="alice@example.com")
    user3.sync_persist()

    ids_to_delete = [user2.id, user3.id]
    User.sync_delete_many([id for id in ids_to_delete if id is not None])
    remaining_users = User.sync_list()
    assert len(remaining_users) == 0


def test_sync_batch_operations():
    """Test synchronous batch operations."""
    setup_test_schema()  # Create the schema first
    setup_test_tables()

    # Test insert_many
    users = [
        User(name="User1", email="user1@example.com"),
        User(name="User2", email="user2@example.com"),
        User(name="User3", email="user3@example.com"),
    ]
    User.sync_insert_many(users)

    # Verify they were inserted
    all_users = User.sync_list(order="name")
    assert len(all_users) == 3
    assert all_users[0].name == "User1"
    assert all_users[2].name == "User3"

    # Test get_many
    user_ids = [u.id for u in all_users[:2]]
    fetched_users = User.sync_get_many(user_ids)
    assert len(fetched_users) == 2


def test_sync_relationships():
    """Test synchronous relationship loading."""
    setup_test_schema()  # Create the schema first
    setup_test_tables()

    # Create user and posts
    user = User(name="Author", email="author@example.com")
    user.sync_persist()

    post1 = Post(user_id=user.id, title="Post 1", content="Content 1")
    post1.sync_persist()

    post2 = Post(user_id=user.id, title="Post 2", content="Content 2")
    post2.sync_persist()

    # Test lazy loading relationship (sync)
    fetched_post = Post.sync_get_with_preload(post1.id, ["user"])
    assert fetched_post is not None
    author = fetched_post.user
    assert author is not None
    assert author.name == "Author"

    fetched_user = User.sync_get_with_preload(user.id, preload=["posts"])
    assert fetched_user is not None
    posts = fetched_user.posts
    assert len(posts) == 2
    assert posts[0].title in ["Post 1", "Post 2"]


def test_sync_preloading():
    """Test synchronous preloading."""
    setup_test_schema()  # Create the schema first
    setup_test_tables()

    # Create test data
    user = User(name="Preload Test", email="preload@example.com")
    user.sync_persist()

    post1 = Post(user_id=user.id, title="Preload 1", content="Content")
    post1.sync_persist()
    post2 = Post(user_id=user.id, title="Preload 2", content="Content")
    post2.sync_persist()

    # Debug: Check posts exist
    print(f"\nDebug: Created user ID {user.id}")
    print(f"Debug: Created post1 ID {post1.id}")
    print(f"Debug: Created post2 ID {post2.id}")

    # Verify posts exist in DB
    posts_check = Post.sync_list(where="user_id = %s", params=(user.id,))
    print(f"Debug: Posts in DB for user {user.id}: {len(posts_check)}")
    for p in posts_check:
        print(f"  - Post {p.id}: {p.title}")

    # Debug: Run the JOIN query manually to see what we get
    import psycopg
    from psycopg.rows import dict_row

    db = User.db_manager()
    with db.sync_get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            query = """
                SELECT t0.*, t1.*
                FROM users t0
                LEFT JOIN posts t1 ON t1.user_id = t0.id
                WHERE t0.id = %s
            """
            cur.execute(query, (user.id,))
            rows = cur.fetchall()
            print(f"\nDebug: Manual JOIN query returned {len(rows)} rows")
            for i, row in enumerate(rows):
                print(f"Debug: Row {i} columns: {list(row.keys())}")
                if i == 0:  # Show first row values
                    for k, v in row.items():
                        print(f"  {k}: {v}")

    # Test sync_get_with_preload
    preloaded_user = User.sync_get_with_preload(user.id, preload=["posts"])
    assert preloaded_user is not None

    # Debug: Check if preloaded attribute exists
    if hasattr(preloaded_user, "_preloaded_posts"):
        print(
            f"\nDebug: Has _preloaded_posts: {getattr(preloaded_user, '_preloaded_posts')}"
        )
    else:
        print("\nDebug: Does NOT have _preloaded_posts attribute")

    # Debug: Check what LazyAttribute returns
    posts_result = preloaded_user.posts
    print(f"Debug: Posts from preloaded user: {posts_result}")
    print(f"Debug: Type of posts result: {type(posts_result)}")
    if posts_result:
        print(f"Debug: Number of posts: {len(posts_result)}")

    # Access posts without additional queries
    assert len(preloaded_user.posts) == 2

    # Test sync_list_with_preload
    print("\nDebug: Testing sync_list_with_preload...")

    # Debug: Run the query manually to see columns
    db = Post.db_manager()
    with db.sync_get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            query = """
                SELECT t0.id AS "t0.id", t0.user_id AS "t0.user_id", t0.title AS "t0.title",
                       t1.id AS "t1.id", t1.name AS "t1.name", t1.email AS "t1.email"
                FROM posts t0
                LEFT JOIN users t1 ON t0.user_id = t1.id
                ORDER BY t0.id
            """
            cur.execute(query)
            rows = cur.fetchall()
            print(f"Debug: Manual query returned {len(rows)} rows")
            if rows:
                print(f"Debug: First row columns: {list(rows[0].keys())}")

    posts = Post.sync_list_with_preload(preload=["user"], order="id")
    print(f"Debug: sync_list_with_preload returned {len(posts)} posts")
    for i, post in enumerate(posts):
        print(
            f"Debug: Post {i}: id={post.id}, title={post.title}, user_id={post.user_id}"
        )
        if hasattr(post, "_preloaded_user"):
            print(f"  Has _preloaded_user: {getattr(post, '_preloaded_user')}")
        else:
            print("  Does NOT have _preloaded_user")

    assert len(posts) == 2
    for post in posts:
        assert post.user is not None
        assert post.user.name == "Preload Test"

    # Test sync_find_by_with_preload
    post = Post.sync_find_by_with_preload(preload=["user"], title="Preload 1")
    assert post is not None
    assert post.user.email == "preload@example.com"


def test_sync_join_table():
    """Test synchronous join table operations."""
    setup_test_schema()  # Create the schema first
    setup_test_tables()

    # Create user
    user = User(name="Role User", email="roles@example.com")
    user.sync_persist()

    # Create user roles
    role1 = UserRole(user_id=user.id, role_id=1)
    role1.sync_persist()

    role2 = UserRole(user_id=user.id, role_id=2)
    role2.sync_persist()

    # Test sync_get_by_pks
    fetched_role = UserRole.sync_get_by_pks(user.id, 1)
    assert fetched_role is not None
    assert fetched_role.role_id == 1

    # Test sync_delete_by_pks
    UserRole.sync_delete_by_pks(user.id, 2)
    deleted_role = UserRole.sync_get_by_pks(user.id, 2)
    assert deleted_role is None

    # Verify first role still exists
    remaining_role = UserRole.sync_get_by_pks(user.id, 1)
    assert remaining_role is not None


def test_sync_serialization():
    """Test synchronous JSON serialization."""
    setup_test_schema()  # Create the schema first
    setup_test_tables()

    # Create user with metadata
    user = User(
        name="JSON Test",
        email="json@example.com",
        metadata={"tags": ["test", "sync"], "active": True},
    )
    user.sync_persist()

    # Test sync_to_json_dict
    json_dict = user.sync_to_json_dict()
    assert json_dict["name"] == "JSON Test"
    assert json_dict["email"] == "json@example.com"
    assert json_dict["metadata"] == {"tags": ["test", "sync"], "active": True}
    assert "createdat" in json_dict  # Column name is createdat, not created_at


def test_sync_persist_with_id():
    """Test synchronous persist_with_id."""
    setup_test_schema()  # Create the schema first
    setup_test_tables()

    # Create user with specific ID
    user = User(id=999, name="Fixed ID", email="fixed@example.com")
    user.sync_persist_with_id()

    # Verify it was created with that ID
    fetched = User.sync_get(999)
    assert fetched is not None
    assert fetched.id == 999
    assert fetched.name == "Fixed ID"


def test_sync_transaction():
    """Test synchronous transactions."""
    setup_test_schema()  # Create the schema first
    setup_test_tables()

    db = User.db_manager()

    # Test rollback on error
    try:
        with db.sync_get_cursor() as cur:
            user1 = User(name="TX User 1", email="tx1@example.com")
            user1.sync_persist(cursor=cur)

            # This should fail due to duplicate email
            user2 = User(name="TX User 2", email="tx1@example.com")
            user2.sync_persist(cursor=cur)
    except Exception:
        pass  # Expected

    # Verify rollback worked - no users should exist
    users = User.sync_list()
    assert len(users) == 0

    # Test successful transaction
    with db.sync_get_cursor() as cur:
        user1 = User(name="TX Success 1", email="success1@example.com")
        user1.sync_persist(cursor=cur)

        user2 = User(name="TX Success 2", email="success2@example.com")
        user2.sync_persist(cursor=cur)

    # Verify commit worked
    users = User.sync_list(order="name")
    assert len(users) == 2
    assert users[0].name == "TX Success 1"
    assert users[1].name == "TX Success 2"


if __name__ == "__main__":
    try:
        # Setup test schema
        print(f"Creating test schema '{TEST_SCHEMA}'...")
        setup_test_schema()
        print(f"✓ Test schema created\n")

        # Run tests
        print("Testing sync CRUD operations...")
        test_sync_crud_operations()
        print("✓ CRUD operations passed")

        print("Testing sync batch operations...")
        test_sync_batch_operations()
        print("✓ Batch operations passed")

        print("Testing sync relationships...")
        test_sync_relationships()
        print("✓ Relationships passed")

        print("Testing sync preloading...")
        test_sync_preloading()
        print("✓ Preloading passed")

        print("Testing sync join table...")
        test_sync_join_table()
        print("✓ Join table passed")

        print("Testing sync serialization...")
        test_sync_serialization()
        print("✓ Serialization passed")

        print("Testing sync persist with ID...")
        test_sync_persist_with_id()
        print("✓ Persist with ID passed")

        print("Testing sync transactions...")
        test_sync_transaction()
        print("✓ Transactions passed")

        print("\n✅ All synchronous tests passed!")

    finally:
        # Cleanup
        print(f"\nCleaning up test schema '{TEST_SCHEMA}'...")
        cleanup_test_schema()
        print("✓ Test schema dropped")
