"""
Tests for JSON serialization with preloaded relationships.

Tests the to_json_dict and sync_to_json_dict methods with:
- Basic column serialization
- Preloaded relationship serialization
- Nested preloaded relationships
- Circular reference handling
- Backward compatibility
"""

import asyncio
import os
import pytest
from datetime import datetime
from decimal import Decimal
from typing import Optional, List

import psycopg
from psycopg.types.json import Jsonb

from reson.data.postgres.models import DBModel, Column, PreloadAttribute, ModelRegistry
from reson.data.postgres.manager import DatabaseManager


# Test schema configuration
TEST_SCHEMA = "test_serialization_temp"


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


def get_test_db_manager():
    """Get database manager with test DSN."""
    return DatabaseManager(dsn=get_test_dsn())


# Test Models
class Organization(DBModel):
    TABLE_NAME = "organizations"
    COLUMNS = {
        "id": Column("id", "id", int),
        "name": Column("name", "name", str),
        "created_at": Column("created_at", "createdat", datetime),
        "metadata": Column("metadata", "metadata", Jsonb),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ):
        self.id = id
        self.name = name
        self.created_at = created_at or datetime.now()
        self.metadata = metadata or {}

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(
        preload=True,
        reverse_fk="organization_id",
        references="users",
        relationship_type="one_to_many",
        model="User",
    )
    def users(self) -> List["User"]:
        """Relationship to users."""
        return []  # Handled by PreloadAttribute

    @PreloadAttribute(
        preload=True,
        reverse_fk="organization_id",
        references="subscriptions",
        relationship_type="one_to_one",
        model="Subscription",
    )
    def subscription(self) -> Optional["Subscription"]:
        """Relationship to subscription."""
        return None  # Handled by PreloadAttribute


class User(DBModel):
    TABLE_NAME = "users"
    COLUMNS = {
        "id": Column("id", "id", int),
        "name": Column("name", "name", str),
        "email": Column("email", "email", str),
        "organization_id": Column("organization_id", "organization_id", int),
        "created_at": Column("created_at", "createdat", datetime),
        "balance": Column("balance", "balance", Decimal, nullable=True),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        email: Optional[str] = None,
        organization_id: Optional[int] = None,
        created_at: Optional[datetime] = None,
        balance: Optional[Decimal] = None,
    ):
        self.id = id
        self.name = name
        self.email = email
        self.organization_id = organization_id
        self.created_at = created_at or datetime.now()
        self.balance = balance

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(
        preload=True,
        foreign_key="organization_id",
        references="organizations",
        ref_column="id",
        model="Organization",
    )
    def organization(self) -> Optional[Organization]:
        """Relationship to organization."""
        return None  # Handled by PreloadAttribute

    @PreloadAttribute(
        preload=True,
        reverse_fk="user_id",
        references="posts",
        relationship_type="one_to_many",
        model="Post",
    )
    def posts(self) -> List["Post"]:
        """Relationship to posts."""
        return []  # Handled by PreloadAttribute

    @PreloadAttribute(
        preload=True,
        join_table="user_tags",
        references="tags",
        join_fk="user_id",
        join_ref_fk="tag_id",
        model="Tag",
    )
    def tags(self) -> List["Tag"]:
        """Relationship to tags."""
        return []  # Handled by PreloadAttribute


class Post(DBModel):
    TABLE_NAME = "posts"
    COLUMNS = {
        "id": Column("id", "id", int),
        "title": Column("title", "title", str),
        "content": Column("content", "content", str),
        "user_id": Column("user_id", "user_id", int),
        "created_at": Column("created_at", "createdat", datetime),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        title: Optional[str] = None,
        content: Optional[str] = None,
        user_id: Optional[int] = None,
        created_at: Optional[datetime] = None,
    ):
        self.id = id
        self.title = title
        self.content = content
        self.user_id = user_id
        self.created_at = created_at or datetime.now()

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(
        preload=True,
        foreign_key="user_id",
        references="users",
        ref_column="id",
        model="User",
    )
    def user(self) -> Optional[User]:
        """Relationship to user."""
        return None  # Handled by PreloadAttribute


class Subscription(DBModel):
    TABLE_NAME = "subscriptions"
    COLUMNS = {
        "id": Column("id", "id", int),
        "plan": Column("plan", "plan", str),
        "organization_id": Column("organization_id", "organization_id", int),
        "price": Column("price", "price", Decimal),
        "created_at": Column("created_at", "createdat", datetime),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        plan: Optional[str] = None,
        organization_id: Optional[int] = None,
        price: Optional[Decimal] = None,
        created_at: Optional[datetime] = None,
    ):
        self.id = id
        self.plan = plan
        self.organization_id = organization_id
        self.price = price or Decimal("0.00")
        self.created_at = created_at or datetime.now()

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(
        preload=True,
        foreign_key="organization_id",
        references="organizations",
        ref_column="id",
        model="Organization",
    )
    def organization(self) -> Optional[Organization]:
        """Relationship to organization."""
        return None  # Handled by PreloadAttribute


class Tag(DBModel):
    TABLE_NAME = "tags"
    COLUMNS = {
        "id": Column("id", "id", int),
        "name": Column("name", "name", str),
        "created_at": Column("created_at", "createdat", datetime),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ):
        self.id = id
        self.name = name
        self.created_at = created_at or datetime.now()

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(
        preload=True,
        join_table="user_tags",
        references="users",
        join_fk="tag_id",
        join_ref_fk="user_id",
        model="User",
    )
    def users(self) -> List[User]:
        """Relationship to users."""
        return []  # Handled by PreloadAttribute


# Register models
registry = ModelRegistry()
registry.register_model("Organization", Organization)
registry.register_model("User", User)
registry.register_model("Post", Post)
registry.register_model("Subscription", Subscription)
registry.register_model("Tag", Tag)


async def setup_test_schema():
    """Create test schema and tables."""
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
    db = get_test_db_manager()

    async with db.get_connection() as conn:
        async with conn.cursor() as cur:
            # Create organizations table
            await cur.execute(
                """
                CREATE TABLE organizations (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255),
                    createdat TIMESTAMP DEFAULT NOW(),
                    metadata JSONB
                )
            """
            )

            # Create users table
            await cur.execute(
                """
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255),
                    email VARCHAR(255),
                    organization_id INTEGER REFERENCES organizations(id),
                    createdat TIMESTAMP DEFAULT NOW(),
                    balance DECIMAL(10, 2)
                )
            """
            )

            # Create posts table
            await cur.execute(
                """
                CREATE TABLE posts (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(255),
                    content TEXT,
                    user_id INTEGER REFERENCES users(id),
                    createdat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create subscriptions table
            await cur.execute(
                """
                CREATE TABLE subscriptions (
                    id SERIAL PRIMARY KEY,
                    plan VARCHAR(100),
                    organization_id INTEGER REFERENCES organizations(id),
                    price DECIMAL(10, 2),
                    createdat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create tags table
            await cur.execute(
                """
                CREATE TABLE tags (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    createdat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create user_tags join table
            await cur.execute(
                """
                CREATE TABLE user_tags (
                    user_id INTEGER REFERENCES users(id),
                    tag_id INTEGER REFERENCES tags(id),
                    PRIMARY KEY (user_id, tag_id)
                )
            """
            )

            # Create sequences
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS organizations_seq")
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS users_seq")
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS posts_seq")
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS subscriptions_seq")
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS tags_seq")

            await conn.commit()


async def cleanup_test_schema():
    """Drop the test schema."""
    base_dsn = os.environ.get(
        "POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    db = DatabaseManager(dsn=base_dsn)

    async with db.get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
            await conn.commit()


# Helper function to create test data
async def create_test_data():
    """Create test data."""
    async with Organization.db_manager().get_cursor() as cursor:
        # Create organization
        org = Organization(
            name="Tech Corp", metadata={"industry": "technology", "size": "large"}
        )
        await org.persist(cursor)

        # Create subscription
        sub = Subscription(
            plan="Enterprise", organization_id=org.id, price=Decimal("999.99")
        )
        await sub.persist(cursor)

        # Create users
        user1 = User(
            name="Alice",
            email="alice@tech.com",
            organization_id=org.id,
            balance=Decimal("1500.50"),
        )
        await user1.persist(cursor)

        user2 = User(
            name="Bob",
            email="bob@tech.com",
            organization_id=org.id,
            balance=Decimal("2000.00"),
        )
        await user2.persist(cursor)

        # Create posts
        post1 = Post(title="First Post", content="Hello World", user_id=user1.id)
        await post1.persist(cursor)

        post2 = Post(title="Second Post", content="Python is great", user_id=user1.id)
        await post2.persist(cursor)

        post3 = Post(title="Bob's Post", content="Testing 123", user_id=user2.id)
        await post3.persist(cursor)

        # Create tags
        tag1 = Tag(name="python")
        await tag1.persist(cursor)

        tag2 = Tag(name="testing")
        await tag2.persist(cursor)

        tag3 = Tag(name="development")
        await tag3.persist(cursor)

        # Associate tags with users
        await cursor.execute(
            "INSERT INTO user_tags (user_id, tag_id) VALUES (%s, %s)",
            (user1.id, tag1.id),
        )
        await cursor.execute(
            "INSERT INTO user_tags (user_id, tag_id) VALUES (%s, %s)",
            (user1.id, tag2.id),
        )
        await cursor.execute(
            "INSERT INTO user_tags (user_id, tag_id) VALUES (%s, %s)",
            (user2.id, tag2.id),
        )
        await cursor.execute(
            "INSERT INTO user_tags (user_id, tag_id) VALUES (%s, %s)",
            (user2.id, tag3.id),
        )

    return {
        "org": org,
        "users": [user1, user2],
        "posts": [post1, post2, post3],
        "tags": [tag1, tag2, tag3],
        "subscription": sub,
    }


# Async tests
@pytest.mark.asyncio
async def test_basic_serialization_without_preload():
    """Test that serialization without preloading only includes columns."""
    await setup_test_schema()
    data = await create_test_data()
    user = await User.get(data["users"][0].id)

    result = await user.to_json_dict()

    # Should have column values
    assert "id" in result
    assert result["name"] == "Alice"
    assert result["email"] == "alice@tech.com"
    assert result["organization_id"] == data["org"].id
    assert "createdat" in result
    assert result["balance"] == 1500.50  # Decimal converted to float

    # Should NOT have relationships
    assert "organization" not in result
    assert "posts" not in result
    assert "tags" not in result


@pytest.mark.asyncio
async def test_serialization_with_preloaded_many_to_one():
    """Test serialization with preloaded many-to-one relationship."""
    await setup_test_schema()
    data = await create_test_data()
    user = await User.get_with_preload(data["users"][0].id, preload=["organization"])

    result = await user.to_json_dict()

    # Should have columns
    assert result["name"] == "Alice"

    # Should have preloaded organization
    assert "organization" in result
    assert result["organization"]["id"] == data["org"].id
    assert result["organization"]["name"] == "Tech Corp"
    assert result["organization"]["metadata"] == {
        "industry": "technology",
        "size": "large",
    }

    # Organization shouldn't have its relationships since they weren't preloaded
    assert "users" not in result["organization"]
    assert "subscription" not in result["organization"]


@pytest.mark.asyncio
async def test_serialization_with_preloaded_one_to_many():
    """Test serialization with preloaded one-to-many relationship."""
    await setup_test_schema()
    data = await create_test_data()
    org = await Organization.get_with_preload(data["org"].id, preload=["users"])

    result = await org.to_json_dict()

    # Should have columns
    assert result["name"] == "Tech Corp"

    # Should have preloaded users
    assert "users" in result
    assert len(result["users"]) == 2

    # Check users are properly serialized
    user_names = {u["name"] for u in result["users"]}
    assert user_names == {"Alice", "Bob"}

    # Users shouldn't have their relationships since they weren't preloaded
    for user in result["users"]:
        assert "organization" not in user
        assert "posts" not in user
        assert "tags" not in user


@pytest.mark.asyncio
async def test_serialization_with_preloaded_one_to_one():
    """Test serialization with preloaded one-to-one relationship."""
    await setup_test_schema()
    data = await create_test_data()
    org = await Organization.get_with_preload(data["org"].id, preload=["subscription"])

    result = await org.to_json_dict()

    # Should have columns
    assert result["name"] == "Tech Corp"

    # Should have preloaded subscription
    assert "subscription" in result
    assert result["subscription"]["id"] == data["subscription"].id
    assert result["subscription"]["plan"] == "Enterprise"
    assert result["subscription"]["price"] == 999.99  # Decimal converted to float

    # Subscription shouldn't have its organization since it wasn't preloaded
    assert "organization" not in result["subscription"]


@pytest.mark.asyncio
async def test_serialization_with_preloaded_many_to_many():
    """Test serialization with preloaded many-to-many relationship."""
    await setup_test_schema()
    data = await create_test_data()
    user = await User.get_with_preload(data["users"][0].id, preload=["tags"])

    result = await user.to_json_dict()

    # Should have columns
    assert result["name"] == "Alice"

    # Should have preloaded tags
    assert "tags" in result
    assert len(result["tags"]) == 2

    # Check tags are properly serialized
    tag_names = {t["name"] for t in result["tags"]}
    assert tag_names == {"python", "testing"}

    # Tags shouldn't have their users since they weren't preloaded
    for tag in result["tags"]:
        assert "users" not in tag


@pytest.mark.asyncio
async def test_nested_serialization():
    """Test serialization with nested preloaded relationships."""
    await setup_test_schema()
    data = await create_test_data()

    # Preload user with organization and posts
    user = await User.get(data["users"][0].id)
    await user.preload(["organization > subscription", "posts"])

    result = await user.to_json_dict()

    # Should have direct preloaded relationships
    assert "organization" in result
    assert "posts" in result
    assert len(result["posts"]) == 2

    # Organization should have its nested subscription
    assert "subscription" in result["organization"]
    assert result["organization"]["subscription"]["plan"] == "Enterprise"

    # Posts shouldn't have their user since it wasn't preloaded at that level
    for post in result["posts"]:
        assert "user" not in post


@pytest.mark.asyncio
async def test_circular_reference_handling():
    """Test that circular references are handled properly."""
    await setup_test_schema()
    data = await create_test_data()

    # Create a circular reference: user -> organization -> users (which includes original user)
    user = await User.get(data["users"][0].id)
    await user.preload(["organization"])
    await user.organization.preload(["users"])

    # Each user in the organization should also preload their organization
    for org_user in user.organization.users:
        await org_user.preload(["organization"])

    result = await user.to_json_dict()

    # Should have organization
    assert "organization" in result
    assert result["organization"]["id"] == data["org"].id

    # Organization should have users
    assert "users" in result["organization"]
    assert len(result["organization"]["users"]) == 2

    # Users in organization should have the full cached organization (circular reference is handled by caching)
    for org_user in result["organization"]["users"]:
        # Both users should have the full cached organization object
        assert "organization" in org_user
        assert org_user["organization"]["id"] == data["org"].id
        assert org_user["organization"]["name"] == "Tech Corp"
        # The cached org should have metadata and created_at fields
        assert "metadata" in org_user["organization"]
        assert "createdat" in org_user["organization"]


@pytest.mark.asyncio
async def test_multiple_preloaded_relationships():
    """Test serialization with multiple preloaded relationships."""
    await setup_test_schema()
    data = await create_test_data()
    user = await User.get_with_preload(
        data["users"][0].id, preload=["organization", "posts", "tags"]
    )

    result = await user.to_json_dict()

    # Should have all preloaded relationships
    assert "organization" in result
    assert "posts" in result
    assert "tags" in result

    # Verify each relationship
    assert result["organization"]["name"] == "Tech Corp"
    assert len(result["posts"]) == 2
    assert len(result["tags"]) == 2


@pytest.mark.asyncio
async def test_null_relationship_serialization():
    """Test serialization when a preloaded relationship is null."""
    await setup_test_schema()
    # Create a user without an organization
    user = User(name="Charlie", email="charlie@test.com", organization_id=None)
    await user.persist()

    # Preload the null organization relationship
    await user.preload(["organization"])

    result = await user.to_json_dict()

    # Should have organization key with None value
    assert "organization" in result
    assert result["organization"] is None

    # Cleanup
    if user.id is not None:
        await User.delete(user.id)


@pytest.mark.asyncio
async def test_empty_list_relationship_serialization():
    """Test serialization when a preloaded list relationship is empty."""
    await setup_test_schema()
    # Create a user without posts
    user = User(name="David", email="david@test.com", organization_id=None)
    await user.persist()

    # Preload the empty posts relationship
    await user.preload(["posts"])

    result = await user.to_json_dict()

    # Should have posts key with empty list
    assert "posts" in result
    assert result["posts"] == []

    # Cleanup
    if user.id is not None:
        await User.delete(user.id)


# Sync tests
@pytest.mark.asyncio
async def test_sync_basic_serialization_without_preload():
    """Test sync serialization without preloading only includes columns."""
    await setup_test_schema()
    data = await create_test_data()
    user = User.sync_get(data["users"][0].id)

    result = user.sync_to_json_dict()

    # Should have column values
    assert "id" in result
    assert result["name"] == "Alice"
    assert result["email"] == "alice@tech.com"
    assert result["organization_id"] == data["org"].id
    assert "createdat" in result
    assert result["balance"] == 1500.50

    # Should NOT have relationships
    assert "organization" not in result
    assert "posts" not in result
    assert "tags" not in result


@pytest.mark.asyncio
async def test_sync_serialization_with_preloaded_relationships():
    """Test sync serialization with preloaded relationships."""
    await setup_test_schema()
    data = await create_test_data()
    user = User.sync_get_with_preload(
        data["users"][0].id, preload=["organization", "posts"]
    )

    result = user.sync_to_json_dict()

    # Should have columns and preloaded relationships
    assert result["name"] == "Alice"
    assert "organization" in result
    assert result["organization"]["name"] == "Tech Corp"
    assert "posts" in result
    assert len(result["posts"]) == 2

    # Should NOT have non-preloaded relationships
    assert "tags" not in result


@pytest.mark.asyncio
async def test_sync_circular_reference_handling():
    """Test sync circular reference handling."""
    await setup_test_schema()
    data = await create_test_data()

    # Create a circular reference
    user = User.sync_get(data["users"][0].id)
    user.sync_preload(["organization"])
    user.organization.sync_preload(["users"])

    for org_user in user.organization.users:
        org_user.sync_preload(["organization"])

    result = user.sync_to_json_dict()

    # Should handle circular reference properly
    assert "organization" in result
    assert "users" in result["organization"]

    # Users in organization should have the full cached organization (circular reference is handled by caching)
    for org_user in result["organization"]["users"]:
        # Both users should have the full cached organization object
        assert "organization" in org_user
        assert org_user["organization"]["id"] == data["org"].id
        assert org_user["organization"]["name"] == "Tech Corp"
        # The cached org should have metadata and created_at fields
        assert "metadata" in org_user["organization"]
        assert "createdat" in org_user["organization"]


@pytest.mark.asyncio
async def test_sync_nested_serialization():
    """Test sync serialization with nested preloaded relationships."""
    await setup_test_schema()
    data = await create_test_data()

    # Preload nested relationships
    user = User.sync_get(data["users"][0].id)
    user.sync_preload(["organization > subscription", "posts"])

    result = user.sync_to_json_dict()

    # Should have nested relationships
    assert "organization" in result
    assert "subscription" in result["organization"]
    assert result["organization"]["subscription"]["plan"] == "Enterprise"
    assert "posts" in result
    assert len(result["posts"]) == 2


# Performance test
@pytest.mark.asyncio
async def test_serialization_performance_with_large_dataset():
    """Test serialization performance with a larger dataset."""
    await setup_test_schema()
    # Create a larger dataset
    async with Organization.db_manager().get_cursor() as cursor:
        org = Organization(name="Big Corp", metadata={"size": "huge"})
        await org.persist(cursor)

        # Create 50 users
        users = []
        for i in range(50):
            user = User(
                name=f"User {i}",
                email=f"user{i}@bigcorp.com",
                organization_id=org.id,
                balance=Decimal(f"{1000 + i * 10}.00"),
            )
            await user.persist(cursor)
            users.append(user)

            # Create 3 posts per user
            for j in range(3):
                post = Post(
                    title=f"Post {j} by User {i}",
                    content=f"Content {j}",
                    user_id=user.id,
                )
                await post.persist(cursor)

    # Test serialization with preloaded relationships
    import time

    start = time.time()

    org = await Organization.get_with_preload(org.id, preload=["users"])
    result = await org.to_json_dict()

    elapsed = time.time() - start

    # Should complete reasonably quickly
    assert elapsed < 2.0  # Should take less than 2 seconds

    # Verify result
    assert len(result["users"]) == 50
    assert all("posts" not in user for user in result["users"])  # Posts not preloaded


# Test datetime and special type serialization
@pytest.mark.asyncio
async def test_special_type_serialization():
    """Test serialization of special types like datetime, Decimal, JSON."""
    await setup_test_schema()
    data = await create_test_data()
    org = await Organization.get(data["org"].id)

    result = await org.to_json_dict()

    # Datetime should be ISO format string
    assert isinstance(result["createdat"], str)
    assert "T" in result["createdat"]  # ISO format has T separator

    # JSON/JSONB should be preserved as dict
    assert isinstance(result["metadata"], dict)
    assert result["metadata"]["industry"] == "technology"

    # Test Decimal conversion
    sub = await Subscription.get(data["subscription"].id)
    sub_result = await sub.to_json_dict()

    # Decimal should be converted to float
    assert isinstance(sub_result["price"], float)
    assert sub_result["price"] == 999.99


if __name__ == "__main__":
    # Run async tests
    asyncio.run(test_basic_serialization_without_preload())
