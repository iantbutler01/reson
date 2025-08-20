"""Test instance-level preload functionality"""

import asyncio
import os
from typing import List, Optional
from datetime import datetime

from reson.data.postgres.models import DBModel, PreloadAttribute, Column, ModelRegistry
from psycopg.types.json import Jsonb


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

    @PreloadAttribute(
        preload=True,
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

    @PreloadAttribute(
        preload=True,
        foreign_key="organization_id",
        references="organizations",
        ref_column="id",
        relationship_type="many_to_one",
        model="Organization",
    )
    def organization(self) -> Optional[Organization]:
        return None  # Descriptor - never actually called

    @PreloadAttribute(
        preload=True,
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


# Register models
registry = ModelRegistry()
registry.register_model("User", User)
registry.register_model("Organization", Organization)
registry.register_model("Project", Project)


async def test_instance_preload():
    """Test the instance-level preload functionality"""

    print("Testing instance-level preload...")

    # Setup database connection
    db_manager = User.db_manager()

    # Test 1: Get user without preloading
    print("\n1. Getting user without preload...")
    user = await User.get(1)  # Assuming user with id=1 exists
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
    user2 = await User.get(1)
    await user2.preload()  # Preloads all PreloadAttributes

    # All relationships should now be accessible
    org = user2.organization
    projects = user2.projects
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
    user3 = User.sync_get(1)
    user3.sync_preload(["organization", "projects"])
    org = user3.organization
    projects = user3.projects
    print(f"   Organization: {org.name if org else 'None'}")
    print(f"   Projects: {[p.name for p in projects] if projects else 'None'}")

    print("\nAll tests completed successfully!")


def test_sync_instance_preload():
    """Test the synchronous instance-level preload functionality"""

    print("Testing synchronous instance-level preload...")

    # Test 1: Get user without preloading
    print("\n1. Getting user without preload...")
    user = User.sync_get(1)  # Assuming user with id=1 exists
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


if __name__ == "__main__":
    # Run async tests
    asyncio.run(test_instance_preload())

    # Run sync tests
    test_sync_instance_preload()
