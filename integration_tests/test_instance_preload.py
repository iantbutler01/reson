"""Test instance-level preload functionality"""

import asyncio
import os
import pytest
from typing import List, Optional
from datetime import datetime
from urllib.parse import quote

from reson.data.postgres.models import DBModel, PreloadAttribute, Column, ModelRegistry
from reson.data.postgres.manager import DatabaseManager
from psycopg.types.json import Jsonb


TEST_SCHEMA = "test_instance_preload_temp"


def get_test_dsn():
    """Get DSN with test schema."""
    base_dsn = os.environ.get(
        "POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    # Parse and add schema option - URL encode the = sign
    options = quote(f"-csearch_path={TEST_SCHEMA}")
    if "?" in base_dsn:
        return f"{base_dsn}&options={options}"
    else:
        return f"{base_dsn}?options={options}"


def get_test_db_manager():
    return DatabaseManager(dsn=get_test_dsn())


# Define test models
class Organization(DBModel):
    TABLE_NAME = "organizations"
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
        reverse_fk="organization_id",
        references="users",
        relationship_type="one_to_many",
        model="User",
    )
    def users(self) -> List["User"]:
        return []  # Descriptor - never actually called


class Project(DBModel):
    TABLE_NAME = "projects"
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


class User(DBModel):
    TABLE_NAME = "users"
    COLUMNS = {
        "id": Column("id", "id", int),
        "name": Column("name", "name", str),
        "email": Column("email", "email", str),
        "organization_id": Column(
            "organization_id", "organization_id", int, nullable=True
        ),
        "created_at": Column("created_at", "createdat", datetime),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        email: Optional[str] = None,
        organization_id: Optional[int] = None,
        created_at: Optional[datetime] = None,
    ):
        self.id = id
        self.name = name
        self.email = email
        self.organization_id = organization_id
        self.created_at = created_at or datetime.now()

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(
        foreign_key="organization_id",
        references="organizations",
        ref_column="id",
        relationship_type="many_to_one",
        model="Organization",
    )
    def organization(self) -> Optional[Organization]:
        return None  # Descriptor - never actually called

    @PreloadAttribute(
        join_table="user_projects",
        join_fk="user_id",
        join_ref_fk="project_id",
        references="projects",
        ref_column="id",
        relationship_type="many_to_many",
        model="Project",
    )
    def projects(self) -> List[Project]:
        return []  # Descriptor - never actually called


def register_test_models():
    """Register test models with the registry."""
    registry = ModelRegistry()
    # Clear any existing registrations
    if "User" in registry.models:
        registry.delete_model("User")
    if "Organization" in registry.models:
        registry.delete_model("Organization")
    if "Project" in registry.models:
        registry.delete_model("Project")

    # Register models
    registry.register_model("User", User)
    registry.register_model("Organization", Organization)
    registry.register_model("Project", Project)


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
            # Create organizations table
            await cur.execute(
                """
                CREATE TABLE organizations (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    createdat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create projects table
            await cur.execute(
                """
                CREATE TABLE projects (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    createdat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create users table with organization_id
            await cur.execute(
                """
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    organization_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL,
                    createdat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create user_projects join table
            await cur.execute(
                """
                CREATE TABLE user_projects (
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                    PRIMARY KEY (user_id, project_id)
                )
            """
            )

            # Create sequences
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS organizations_seq")
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS projects_seq")
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


def setup_test_schema_sync():
    """Create test schema and tables synchronously."""
    # Ensure models are registered
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
            # Create organizations table
            cur.execute(
                """
                CREATE TABLE organizations (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    createdat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create projects table
            cur.execute(
                """
                CREATE TABLE projects (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    createdat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create users table with organization_id
            cur.execute(
                """
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    organization_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL,
                    createdat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create user_projects join table
            cur.execute(
                """
                CREATE TABLE user_projects (
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                    PRIMARY KEY (user_id, project_id)
                )
            """
            )

            # Create sequences
            cur.execute("CREATE SEQUENCE IF NOT EXISTS organizations_seq")
            cur.execute("CREATE SEQUENCE IF NOT EXISTS projects_seq")
            cur.execute("CREATE SEQUENCE IF NOT EXISTS users_seq")

            conn.commit()


def cleanup_test_schema_sync():
    """Drop the test schema synchronously."""
    base_dsn = os.environ.get(
        "POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    db = DatabaseManager(dsn=base_dsn)

    with db.sync_get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
            conn.commit()


async def setup_test_data():
    """Set up test data for instance preload tests."""
    # Create organizations
    org1 = Organization(name="Acme Corp")
    await org1.persist()

    org2 = Organization(name="Tech Co")
    await org2.persist()

    # Create projects
    project1 = Project(name="Project Alpha")
    await project1.persist()

    project2 = Project(name="Project Beta")
    await project2.persist()

    # Create users
    user1 = User(
        name="Bismuth User", email="free@bismuth.cloud", organization_id=org1.id
    )
    await user1.persist()

    user2 = User(name="John Doe", email="john@example.com", organization_id=org2.id)
    await user2.persist()

    # Create user-project relationships
    db = get_test_db_manager()
    async with db.get_cursor() as cur:
        await cur.execute(
            "INSERT INTO user_projects (user_id, project_id) VALUES (%s, %s)",
            (user1.id, project1.id),
        )
        await cur.execute(
            "INSERT INTO user_projects (user_id, project_id) VALUES (%s, %s)",
            (user1.id, project2.id),
        )

    return user1, user2, org1, org2, project1, project2


def setup_test_data_sync():
    """Set up test data for sync instance preload tests."""
    # Create organizations
    org1 = Organization(name="Acme Corp")
    org1.sync_persist()

    org2 = Organization(name="Tech Co")
    org2.sync_persist()

    # Create projects
    project1 = Project(name="Project Alpha")
    project1.sync_persist()

    project2 = Project(name="Project Beta")
    project2.sync_persist()

    # Create users
    user1 = User(
        name="Bismuth User", email="free@bismuth.cloud", organization_id=org1.id
    )
    user1.sync_persist()

    user2 = User(name="John Doe", email="john@example.com", organization_id=org2.id)
    user2.sync_persist()

    # Create user-project relationships
    db = get_test_db_manager()
    with db.sync_get_cursor() as cur:
        cur.execute(
            "INSERT INTO user_projects (user_id, project_id) VALUES (%s, %s)",
            (user1.id, project1.id),
        )
        cur.execute(
            "INSERT INTO user_projects (user_id, project_id) VALUES (%s, %s)",
            (user1.id, project2.id),
        )

    return user1, user2, org1, org2, project1, project2


@pytest.mark.asyncio
async def test_instance_preload():
    """Test the instance-level preload functionality"""
    await setup_test_schema()
    user1, user2, org1, org2, project1, project2 = await setup_test_data()

    print("Testing instance-level preload...")

    # Test 1: Get user without preloading
    print("\n1. Getting user without preload...")
    user = await User.get(user1.id)
    print(f"   User: {user.name} ({user.email})")

    # This would raise AttributeError without preloading
    try:
        org = user.organization
        print(f"   Organization accessed: {org}")
    except AttributeError as e:
        print(f"   Expected error: {e}")

    # Test 2: Preload specific attributes
    print("\n2. Preloading specific attributes...")
    await user.preload(["organization"])
    org = user.organization  # Now this works
    print(f"   Organization after preload: {org.name if org else 'None'}")

    # Test 3: Preload all attributes
    print("\n3. Preloading all attributes...")
    user2_obj = await User.get(user1.id)
    await user2_obj.preload()  # Preloads all PreloadAttributes

    # All relationships should now be accessible
    org = user2_obj.organization
    projects = user2_obj.projects
    print(f"   Organization: {org.name if org else 'None'}")
    print(f"   Projects: {[p.name for p in projects] if projects else 'None'}")

    # Test 4: Test with unpersisted entity (should raise error)
    print("\n4. Testing with unpersisted entity...")
    new_user = User(name="New User", email="new@example.com")
    try:
        await new_user.preload()
        print("   ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"   Expected error: {e}")

    # Test 5: Test sync version
    print("\n5. Testing sync preload...")
    user3 = User.sync_get(user1.id)
    user3.sync_preload(["organization", "projects"])
    org = user3.organization
    projects = user3.projects
    print(f"   Organization: {org.name if org else 'None'}")
    print(f"   Projects: {[p.name for p in projects] if projects else 'None'}")

    print("\nAll tests completed successfully!")

    # Cleanup
    await cleanup_test_schema()


def test_sync_instance_preload():
    """Test the synchronous instance-level preload functionality"""
    setup_test_schema_sync()
    user1, user2, org1, org2, project1, project2 = setup_test_data_sync()

    print("Testing synchronous instance-level preload...")

    # Test 1: Get user without preloading
    print("\n1. Getting user without preload...")
    user = User.sync_get(user1.id)
    print(f"   User: {user.name} ({user.email})")

    # This would raise AttributeError without preloading
    try:
        org = user.organization
        print(f"   Organization accessed: {org}")
    except AttributeError as e:
        print(f"   Expected error: {e}")

    # Test 2: Preload specific attributes
    print("\n2. Preloading specific attributes...")
    user.sync_preload(["organization"])
    org = user.organization  # Now this works
    print(f"   Organization after preload: {org.name if org else 'None'}")

    print("\nSync tests completed successfully!")

    # Cleanup
    cleanup_test_schema_sync()


async def main():
    """Run all async tests."""
    try:
        print("Running instance preload tests...")
        await test_instance_preload()
        print("✓ Async instance preload tests passed")

        test_sync_instance_preload()
        print("✓ Sync instance preload tests passed")

        print("\n✅ All instance preload tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
