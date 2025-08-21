"""Test many-to-many nested preloading functionality to reproduce the bug."""

import os
import asyncio
import pytest
from typing import List, Optional
from datetime import datetime
from enum import Enum

from reson.data.postgres.models import (
    DBModel,
    Column,
    PreloadAttribute,
    ModelRegistry,
)
from reson.data.postgres.manager import DatabaseManager


TEST_SCHEMA = "test_many_to_many_nested_preload_temp"


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
    return DatabaseManager(dsn=get_test_dsn())


async def setup_test_schema():
    """Create test schema and tables asynchronously."""
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
            # Create subscriptions table
            await cur.execute(
                """
                CREATE TABLE subscriptions (
                    id SERIAL PRIMARY KEY,
                    createdat TIMESTAMP DEFAULT NOW(),
                    updatedat TIMESTAMP DEFAULT NOW(),
                    customerid TEXT,
                    subscriptionid TEXT,
                    type TEXT NOT NULL,
                    expiresat TIMESTAMP
                )
            """
            )

            # Create organizations table
            await cur.execute(
                """
                CREATE TABLE organizations (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    createdat TIMESTAMP DEFAULT NOW(),
                    updatedat TIMESTAMP DEFAULT NOW(),
                    subscriptionid INTEGER REFERENCES subscriptions(id),
                    llmconfig JSONB,
                    linearinstallid INTEGER
                )
            """
            )

            # Create users table
            await cur.execute(
                """
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    updatedat TIMESTAMP DEFAULT NOW(),
                    createdat TIMESTAMP DEFAULT NOW(),
                    email TEXT NOT NULL,
                    username TEXT NOT NULL,
                    name TEXT NOT NULL,
                    pending BOOLEAN DEFAULT FALSE
                )
            """
            )

            # Create join table matching the exact user schema
            await cur.execute(
                """
                CREATE TABLE organization_users (
                    orgid INTEGER REFERENCES organizations(id) ON DELETE CASCADE,
                    userid INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    PRIMARY KEY (orgid, userid)
                )
            """
            )

            # Create sequences for the tables (needed by persist method)
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS subscriptions_seq")
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS organizations_seq")
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS users_seq")

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


# Test entities matching the user's exact setup
class SubscriptionType(Enum):
    NONE = "NONE"
    STARTER = "STARTER"
    GROWTH = "GROWTH"
    TEAM = "TEAM"


class SubscriptionEntity(DBModel):
    TABLE_NAME = "subscriptions"
    COLUMNS = {
        "id": Column("id", "id", int),
        "created_at": Column("created_at", "createdat", datetime),
        "updated_at": Column("updated_at", "updatedat", datetime),
        "customer_id": Column("customer_id", "customerid", str, nullable=True),
        "subscription_id": Column(
            "subscription_id", "subscriptionid", str, nullable=True
        ),
        "type": Column("type", "type", SubscriptionType),
        "expires_at": Column("expires_at", "expiresat", datetime, nullable=True),
    }

    def __init__(
        self,
        type: SubscriptionType = SubscriptionType.NONE,
        id: Optional[int] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        customer_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ):
        self.id = id
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.customer_id = customer_id
        self.subscription_id = subscription_id
        self.type = type
        self.expires_at = expires_at

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()


class OrganizationEntity(DBModel):
    TABLE_NAME = "organizations"
    COLUMNS = {
        "id": Column("id", "id", int),
        "name": Column("name", "name", str),
        "created_at": Column("created_at", "createdat", datetime),
        "updated_at": Column("updated_at", "updatedat", datetime),
        "subscription_id": Column(
            "subscription_id", "subscriptionid", int, nullable=True
        ),
        "llm_config": Column("llm_config", "llmconfig", dict, nullable=True),
        "linear_install_id": Column(
            "linear_install_id", "linearinstallid", int, nullable=True
        ),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        updated_at: Optional[datetime] = None,
        created_at: Optional[datetime] = None,
        name: str = "",
        subscription_id: Optional[int] = None,
        llm_config: Optional[dict] = None,
        linear_install_id: Optional[int] = None,
    ):
        self.id = id
        self.updated_at = updated_at or datetime.now()
        self.created_at = created_at or datetime.now()
        self.name = name
        self.subscription_id = subscription_id
        self.llm_config = llm_config
        self.linear_install_id = linear_install_id

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(
        foreign_key="subscriptionid",
        references="subscriptions",
        model="SubscriptionEntity",
    )
    async def subscription(self) -> "SubscriptionEntity":  # type: ignore
        pass

    @PreloadAttribute(
        join_table="organization_users",
        join_fk="orgid",
        join_ref_fk="userid",
        references="users",
        model="UserEntity",
    )
    async def users(self) -> List["UserEntity"]:  # type: ignore
        pass


class UserEntity(DBModel):
    TABLE_NAME = "users"
    COLUMNS = {
        "id": Column("id", "id", int),
        "updated_at": Column("updated_at", "updatedat", datetime),
        "created_at": Column("created_at", "createdat", datetime),
        "email": Column("email", "email", str),
        "username": Column("username", "username", str),
        "name": Column("name", "name", str),
        "pending": Column("pending", "pending", bool),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        updated_at: Optional[datetime] = None,
        created_at: Optional[datetime] = None,
        email: str = "",
        username: str = "",
        name: str = "",
        pending: bool = False,
    ):
        self.id = id
        self.updated_at = updated_at or datetime.now()
        self.created_at = created_at or datetime.now()
        self.email = email
        self.username = username
        self.name = name
        self.pending = pending

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(
        join_table="organization_users",
        join_fk="userid",
        join_ref_fk="orgid",
        references="organizations",
        model="OrganizationEntity",
    )
    async def organizations(self) -> List["OrganizationEntity"]:  # type: ignore
        pass


@pytest.fixture(scope="module", autouse=True)
def cleanup_model_registry():
    """Clear ModelRegistry before and after test module to ensure test isolation."""
    # Clear before tests run
    registry = ModelRegistry()
    registry.models.clear()

    # Now register the models for this test module
    registry.register_model("SubscriptionEntity", SubscriptionEntity)
    registry.register_model("OrganizationEntity", OrganizationEntity)
    registry.register_model("UserEntity", UserEntity)

    yield  # Run tests

    # Clear after tests complete
    registry.models.clear()


async def setup_test_data():
    """Set up test data with relationships - matching user's exact scenario."""
    # Create the schema and tables first
    await setup_test_schema()

    async with UserEntity.db_manager().get_cursor() as cursor:
        # Create test data - matching user's exact scenario (1 user, 1 org, 1 subscription)
        subscription = SubscriptionEntity(type=SubscriptionType.STARTER)
        subscription = await subscription.persist(cursor=cursor)

        org = OrganizationEntity(
            name="Test Organization", subscription_id=subscription.id
        )
        org = await org.persist(cursor=cursor)

        user = UserEntity(
            email="test@example.com", username="testuser", name="Test User"
        )
        user = await user.persist(cursor=cursor)

        # Create many-to-many relationship
        await cursor.execute(
            "INSERT INTO organization_users (orgid, userid) VALUES (%s, %s)",
            (org.id, user.id),
        )

        await cursor.execute("COMMIT")

        return {
            "user": user,
            "organization": org,
            "subscription": subscription,
        }


class TestManyToManyNestedPreload:
    """Test many-to-many nested preloading functionality to reproduce the bug."""

    def test_sync_many_to_many_nested_preload(self):
        """Test nested preload through many-to-many relationship (sync) - reproduces the bug."""
        # Run async setup synchronously
        data = asyncio.run(setup_test_data())
        user = data["user"]

        # Reload user and test nested preload
        user = UserEntity.sync_get(user.id)
        user.sync_preload(["organizations > subscription"], refresh=True)

        # Access the organizations
        user_orgs: List[OrganizationEntity] = user.organizations

        # Debug output to check what we get
        print(f"[TEST] user_orgs type: {type(user_orgs)}")
        print(
            f"[TEST] user_orgs length: {len(user_orgs) if isinstance(user_orgs, list) else 'not a list'}"
        )

        assert isinstance(user_orgs, list)

        if isinstance(user_orgs, list) and len(user_orgs) > 0:
            org = user_orgs[0]
            print(f"[TEST] Organization: {org.name}")
            print(
                f"[TEST] Has _preloaded_subscription: {hasattr(org, '_preloaded_subscription')}"
            )

            # Try to access subscription - this should work but will fail
            try:
                subscription = org.subscription
                print(f"[TEST] Subscription accessed successfully: {subscription.type}")
            except AttributeError as e:
                print(f"[TEST] Failed to access subscription: {e}")

        # Cleanup
        asyncio.run(cleanup_test_schema())

    @pytest.mark.asyncio
    async def test_async_many_to_many_nested_preload(self):
        """Test nested preload through many-to-many relationship (async) - reproduces the bug."""
        data = await setup_test_data()
        user = data["user"]

        # Reload user and test nested preload
        user = await UserEntity.get(user.id)
        await user.preload(["organizations > subscription"], refresh=True)

        # Access the organizations
        user_orgs: List[OrganizationEntity] = user.organizations

        # Debug output to check what we get
        print(f"[TEST] user_orgs type: {type(user_orgs)}")
        print(
            f"[TEST] user_orgs length: {len(user_orgs) if isinstance(user_orgs, list) else 'not a list'}"
        )

        assert isinstance(user_orgs, list)

        if isinstance(user_orgs, list) and len(user_orgs) > 0:
            org = user_orgs[0]
            print(f"[TEST] Organization: {org.name}")
            print(
                f"[TEST] Has _preloaded_subscription: {hasattr(org, '_preloaded_subscription')}"
            )

            # Try to access subscription - this should work but will fail
            try:
                subscription = org.subscription
                print(f"[TEST] Subscription accessed successfully: {subscription.type}")
            except AttributeError as e:
                print(f"[TEST] Failed to access subscription: {e}")

        # Cleanup
        await cleanup_test_schema()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
