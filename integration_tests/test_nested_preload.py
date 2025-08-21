"""Test nested preloading functionality."""

import os
import asyncio
import pytest
from typing import List, Optional
from datetime import datetime

from reson.data.postgres.models import (
    DBModel,
    JoinTableDBModel,
    Column,
    PreloadAttribute,
    ModelRegistry,
)
from reson.data.postgres.manager import DatabaseManager
from psycopg.types.json import Jsonb


TEST_SCHEMA = "test_nested_preload_temp"


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
                    plan_name VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    createdat TIMESTAMP DEFAULT NOW(),
                    updatedat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create organizations table
            await cur.execute(
                """
                CREATE TABLE organizations (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    subscription_id INTEGER REFERENCES subscriptions(id) ON DELETE SET NULL,
                    createdat TIMESTAMP DEFAULT NOW(),
                    updatedat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create users table
            await cur.execute(
                """
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    organization_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL,
                    createdat TIMESTAMP DEFAULT NOW(),
                    updatedat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create projects table
            await cur.execute(
                """
                CREATE TABLE projects (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    organization_id INTEGER REFERENCES organizations(id) ON DELETE CASCADE,
                    createdat TIMESTAMP DEFAULT NOW(),
                    updatedat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create profiles table (one-to-one with users)
            await cur.execute(
                """
                CREATE TABLE profiles (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                    bio TEXT,
                    avatar_url VARCHAR(500),
                    createdat TIMESTAMP DEFAULT NOW(),
                    updatedat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create tags table (many-to-many with projects)
            await cur.execute(
                """
                CREATE TABLE tags (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    color VARCHAR(7),
                    createdat TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create project_tags join table (many-to-many)
            await cur.execute(
                """
                CREATE TABLE project_tags (
                    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                    tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
                    assigned_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (project_id, tag_id)
                )
            """
            )

            # Create user_projects join table (many-to-many - users working on projects)
            await cur.execute(
                """
                CREATE TABLE user_projects (
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                    role VARCHAR(50),
                    joined_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (user_id, project_id)
                )
            """
            )

            # Create sequences for the tables
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS subscriptions_seq")
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS organizations_seq")
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS users_seq")
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS projects_seq")
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS profiles_seq")
            await cur.execute("CREATE SEQUENCE IF NOT EXISTS tags_seq")

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


# Don't register models at module level - will do it in fixture


@pytest.fixture(scope="module", autouse=True)
def cleanup_model_registry():
    """Clear ModelRegistry before and after test module to ensure test isolation."""
    # Clear before tests run
    registry = ModelRegistry()
    registry.models.clear()

    # Now register the models for this test module
    registry.register_model("User", User)
    registry.register_model("Organization", Organization)
    registry.register_model("Subscription", Subscription)
    registry.register_model("Project", Project)
    registry.register_model("Profile", Profile)
    registry.register_model("Tag", Tag)
    registry.register_model("UserProject", UserProject)
    registry.register_model("ProjectTag", ProjectTag)

    yield  # Run tests

    # Clear after tests complete
    registry.models.clear()


class User(DBModel):
    TABLE_NAME = "users"
    COLUMNS = {
        "id": Column("id", "id", int),
        "name": Column("name", "name", str),
        "email": Column("email", "email", str),
        "created_at": Column("created_at", "createdat", datetime),
        "updated_at": Column("updated_at", "updatedat", datetime),
        "organization_id": Column(
            "organization_id", "organization_id", int, nullable=True
        ),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        email: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        organization_id: Optional[int] = None,
    ):
        self.id = id
        self.name = name
        self.email = email
        self.created_at = created_at
        self.updated_at = updated_at
        self.organization_id = organization_id

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(
        preload=True,
        foreign_key="organization_id",
        references="organizations",
        ref_column="id",
        relationship_type="many_to_one",
        model="Organization",
    )
    def organization(self):
        """Lazy-loaded organization relationship."""
        pass

    @PreloadAttribute(
        preload=True,
        reverse_fk="user_id",
        references="profiles",
        relationship_type="one_to_one",
        model="Profile",
    )
    def profile(self):
        """Lazy-loaded profile relationship (one-to-one)."""
        pass

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
    def assigned_projects(self):
        """Lazy-loaded assigned projects (many-to-many)."""
        pass


class Organization(DBModel):
    TABLE_NAME = "organizations"
    COLUMNS = {
        "id": Column("id", "id", int),
        "name": Column("name", "name", str),
        "created_at": Column("created_at", "createdat", datetime),
        "updated_at": Column("updated_at", "updatedat", datetime),
        "subscription_id": Column(
            "subscription_id", "subscription_id", int, nullable=True
        ),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        subscription_id: Optional[int] = None,
    ):
        self.id = id
        self.name = name
        self.created_at = created_at
        self.updated_at = updated_at
        self.subscription_id = subscription_id

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
    def users(self):
        """Lazy-loaded users relationship."""
        pass

    @PreloadAttribute(
        preload=True,
        foreign_key="subscription_id",
        references="subscriptions",
        ref_column="id",
        relationship_type="many_to_one",
        model="Subscription",
    )
    def subscription(self):
        """Lazy-loaded subscription relationship."""
        pass

    @PreloadAttribute(
        preload=True,
        reverse_fk="organization_id",
        references="projects",
        relationship_type="one_to_many",
        model="Project",
    )
    def projects(self):
        """Lazy-loaded projects relationship."""
        pass


class Subscription(DBModel):
    TABLE_NAME = "subscriptions"
    COLUMNS = {
        "id": Column("id", "id", int),
        "plan_name": Column("plan_name", "plan_name", str),
        "status": Column("status", "status", str),
        "created_at": Column("created_at", "createdat", datetime),
        "updated_at": Column("updated_at", "updatedat", datetime),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        plan_name: Optional[str] = None,
        status: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        self.id = id
        self.plan_name = plan_name
        self.status = status
        self.created_at = created_at
        self.updated_at = updated_at

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(
        preload=True,
        reverse_fk="subscription_id",
        references="organizations",
        relationship_type="one_to_many",
        model="Organization",
    )
    def organizations(self):
        """Lazy-loaded organizations relationship."""
        pass


class Project(DBModel):
    TABLE_NAME = "projects"
    COLUMNS = {
        "id": Column("id", "id", int),
        "name": Column("name", "name", str),
        "description": Column("description", "description", str, nullable=True),
        "organization_id": Column("organization_id", "organization_id", int),
        "created_at": Column("created_at", "createdat", datetime),
        "updated_at": Column("updated_at", "updatedat", datetime),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        organization_id: Optional[int] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.organization_id = organization_id
        self.created_at = created_at
        self.updated_at = updated_at

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(
        preload=True,
        foreign_key="organization_id",
        references="organizations",
        ref_column="id",
        relationship_type="many_to_one",
        model="Organization",
    )
    def organization(self):
        """Lazy-loaded organization relationship."""
        pass

    @PreloadAttribute(
        preload=True,
        join_table="project_tags",
        join_fk="project_id",
        join_ref_fk="tag_id",
        references="tags",
        ref_column="id",
        relationship_type="many_to_many",
        model="Tag",
    )
    def tags(self):
        """Lazy-loaded tags (many-to-many)."""
        pass

    @PreloadAttribute(
        preload=True,
        join_table="user_projects",
        join_fk="project_id",
        join_ref_fk="user_id",
        references="users",
        ref_column="id",
        relationship_type="many_to_many",
        model="User",
    )
    def team_members(self):
        """Lazy-loaded team members (many-to-many)."""
        pass


class Profile(DBModel):
    TABLE_NAME = "profiles"
    COLUMNS = {
        "id": Column("id", "id", int),
        "user_id": Column("user_id", "user_id", int),
        "bio": Column("bio", "bio", str, nullable=True),
        "avatar_url": Column("avatar_url", "avatar_url", str, nullable=True),
        "created_at": Column("created_at", "createdat", datetime),
        "updated_at": Column("updated_at", "updatedat", datetime),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        user_id: Optional[int] = None,
        bio: Optional[str] = None,
        avatar_url: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        self.id = id
        self.user_id = user_id
        self.bio = bio
        self.avatar_url = avatar_url
        self.created_at = created_at
        self.updated_at = updated_at

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(
        preload=True,
        foreign_key="user_id",
        references="users",
        ref_column="id",
        relationship_type="one_to_one",
        model="User",
    )
    def user(self):
        """Lazy-loaded user relationship (one-to-one)."""
        pass


class Tag(DBModel):
    TABLE_NAME = "tags"
    COLUMNS = {
        "id": Column("id", "id", int),
        "name": Column("name", "name", str),
        "color": Column("color", "color", str, nullable=True),
        "created_at": Column("created_at", "createdat", datetime),
    }

    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        color: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ):
        self.id = id
        self.name = name
        self.color = color
        self.created_at = created_at

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()

    @PreloadAttribute(
        preload=True,
        join_table="project_tags",
        join_fk="tag_id",
        join_ref_fk="project_id",
        references="projects",
        ref_column="id",
        relationship_type="many_to_many",
        model="Project",
    )
    def projects(self):
        """Lazy-loaded projects (many-to-many)."""
        pass


class UserProject(JoinTableDBModel):
    TABLE_NAME = "user_projects"
    PRIMARY_KEY_DB_NAMES = ["user_id", "project_id"]
    COLUMNS = {
        "user_id": Column("user_id", "user_id", int),
        "project_id": Column("project_id", "project_id", int),
        "role": Column("role", "role", str, nullable=True),
        "joined_at": Column("joined_at", "joined_at", datetime),
    }

    def __init__(
        self,
        user_id: Optional[int] = None,
        project_id: Optional[int] = None,
        role: Optional[str] = None,
        joined_at: Optional[datetime] = None,
    ):
        self.user_id = user_id
        self.project_id = project_id
        self.role = role
        self.joined_at = joined_at or datetime.now()

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()


class ProjectTag(JoinTableDBModel):
    TABLE_NAME = "project_tags"
    PRIMARY_KEY_DB_NAMES = ["project_id", "tag_id"]
    COLUMNS = {
        "project_id": Column("project_id", "project_id", int),
        "tag_id": Column("tag_id", "tag_id", int),
        "assigned_at": Column("assigned_at", "assigned_at", datetime),
    }

    def __init__(
        self,
        project_id: Optional[int] = None,
        tag_id: Optional[int] = None,
        assigned_at: Optional[datetime] = None,
    ):
        self.project_id = project_id
        self.tag_id = tag_id
        self.assigned_at = assigned_at or datetime.now()

    @classmethod
    def db_manager(cls):
        return get_test_db_manager()


# Models will be registered in the cleanup_model_registry fixture


@pytest.fixture
async def setup_nested_test_data():
    """Set up test data with nested relationships."""
    # Create the schema and tables first
    await setup_test_schema()

    async with User.db_manager().get_cursor() as cursor:
        # Tables are already created, just need to insert data
        # No need to delete since schema was freshly created

        # Create subscriptions
        subscription1 = Subscription(plan_name="Enterprise", status="active")
        subscription1 = await subscription1.persist(cursor=cursor)

        subscription2 = Subscription(plan_name="Basic", status="active")
        subscription2 = await subscription2.persist(cursor=cursor)

        # Create organizations with subscriptions
        org1 = Organization(name="Tech Corp", subscription_id=subscription1.id)
        org1 = await org1.persist(cursor=cursor)

        org2 = Organization(name="Small Startup", subscription_id=subscription2.id)
        org2 = await org2.persist(cursor=cursor)

        # Create users
        user1 = User(name="Alice", email="alice@techcorp.com", organization_id=org1.id)
        user1 = await user1.persist(cursor=cursor)

        user2 = User(name="Bob", email="bob@techcorp.com", organization_id=org1.id)
        user2 = await user2.persist(cursor=cursor)

        user3 = User(
            name="Charlie", email="charlie@startup.com", organization_id=org2.id
        )
        user3 = await user3.persist(cursor=cursor)

        # Create projects
        project1 = Project(
            name="Main Website", description="Company website", organization_id=org1.id
        )
        project1 = await project1.persist(cursor=cursor)

        project2 = Project(
            name="Mobile App",
            description="iOS and Android app",
            organization_id=org1.id,
        )
        project2 = await project2.persist(cursor=cursor)

        project3 = Project(
            name="MVP", description="Minimum viable product", organization_id=org2.id
        )
        project3 = await project3.persist(cursor=cursor)

        # Create profiles (one-to-one with users)
        profile1 = Profile(
            user_id=user1.id,
            bio="Senior Developer at Tech Corp",
            avatar_url="https://example.com/alice.jpg",
        )
        profile1 = await profile1.persist(cursor=cursor)

        profile2 = Profile(
            user_id=user2.id,
            bio="Product Manager",
            avatar_url="https://example.com/bob.jpg",
        )
        profile2 = await profile2.persist(cursor=cursor)

        # Create tags
        tag1 = Tag(name="backend", color="#0000FF")
        tag1 = await tag1.persist(cursor=cursor)

        tag2 = Tag(name="frontend", color="#00FF00")
        tag2 = await tag2.persist(cursor=cursor)

        tag3 = Tag(name="mobile", color="#FF0000")
        tag3 = await tag3.persist(cursor=cursor)

        # Associate tags with projects (many-to-many)
        project_tag1 = ProjectTag(project_id=project1.id, tag_id=tag1.id)
        await project_tag1.persist(cursor=cursor)

        project_tag2 = ProjectTag(project_id=project1.id, tag_id=tag2.id)
        await project_tag2.persist(cursor=cursor)

        project_tag3 = ProjectTag(project_id=project2.id, tag_id=tag3.id)
        await project_tag3.persist(cursor=cursor)

        project_tag4 = ProjectTag(project_id=project2.id, tag_id=tag2.id)
        await project_tag4.persist(cursor=cursor)

        # Associate users with projects (many-to-many)
        user_project1 = UserProject(
            user_id=user1.id, project_id=project1.id, role="Lead Developer"
        )
        await user_project1.persist(cursor=cursor)

        user_project2 = UserProject(
            user_id=user2.id, project_id=project1.id, role="Manager"
        )
        await user_project2.persist(cursor=cursor)

        user_project3 = UserProject(
            user_id=user1.id, project_id=project2.id, role="Consultant"
        )
        await user_project3.persist(cursor=cursor)

        await cursor.execute("COMMIT")

        return {
            "users": [user1, user2, user3],
            "organizations": [org1, org2],
            "subscriptions": [subscription1, subscription2],
            "projects": [project1, project2, project3],
            "profiles": [profile1, profile2],
            "tags": [tag1, tag2, tag3],
            "project_tags": [project_tag1, project_tag2, project_tag3, project_tag4],
            "user_projects": [user_project1, user_project2, user_project3],
        }


class TestNestedPreload:
    """Test nested preloading functionality."""

    @pytest.mark.asyncio
    async def test_simple_nested_preload_instance(self, setup_nested_test_data):
        """Test simple nested preloading on instance: user > organization > subscription."""
        data = await setup_nested_test_data
        user = data["users"][0]  # Alice

        # Reload user and preload nested relationships
        user = await User.get(user.id)
        await user.preload(["organization > subscription"])

        # Check that organization is loaded
        assert hasattr(user, "_preloaded_organization")
        org = user.organization
        assert org is not None
        assert org.name == "Tech Corp"

        # Check that subscription is loaded on the organization
        assert hasattr(org, "_preloaded_subscription")
        subscription = org.subscription
        print("SUB", subscription)
        assert subscription is not None
        assert subscription.plan_name == "Enterprise"

    @pytest.mark.asyncio
    async def test_multiple_nested_paths_instance(self, setup_nested_test_data):
        """Test multiple nested paths on instance."""
        data = await setup_nested_test_data
        org = data["organizations"][0]  # Tech Corp

        # Reload org and preload multiple nested paths
        org = await Organization.get(org.id)
        await org.preload(["subscription", "users", "projects"])

        # Check all relationships are loaded
        assert hasattr(org, "_preloaded_subscription")
        assert hasattr(org, "_preloaded_users")
        assert hasattr(org, "_preloaded_projects")

        assert org.subscription.plan_name == "Enterprise"
        assert len(org.users) == 2
        assert len(org.projects) == 2

    @pytest.mark.asyncio
    async def test_deep_nested_preload_instance(self, setup_nested_test_data):
        """Test deeper nested preloading: user > organization > projects."""
        data = await setup_nested_test_data
        user = data["users"][0]  # Alice

        # Reload user and preload deeply nested relationships
        user = await User.get(user.id)
        await user.preload(["organization > projects"])

        # Check that organization is loaded
        org = user.organization
        assert org is not None

        # Check that projects are loaded on the organization
        assert hasattr(org, "_preloaded_projects")
        projects = org.projects
        assert len(projects) == 2
        assert any(p.name == "Main Website" for p in projects)
        assert any(p.name == "Mobile App" for p in projects)

    def test_sync_nested_preload_instance(self, setup_nested_test_data):
        """Test synchronous nested preloading."""
        # Run async setup synchronously
        data = asyncio.run(setup_nested_test_data)
        user = data["users"][0]  # Alice

        # Reload user and preload nested relationships synchronously
        user = User.sync_get(user.id)
        user.sync_preload(["organization > subscription"])

        # Check that organization is loaded
        assert hasattr(user, "_preloaded_organization")
        org = user.organization
        assert org is not None
        assert org.name == "Tech Corp"

        # Check that subscription is loaded on the organization
        assert hasattr(org, "_preloaded_subscription")
        subscription = org.subscription
        assert subscription is not None
        assert subscription.plan_name == "Enterprise"

    @pytest.mark.asyncio
    async def test_nested_preload_with_collection(self, setup_nested_test_data):
        """Test nested preloading with collection relationships."""
        data = await setup_nested_test_data
        org = data["organizations"][0]  # Tech Corp

        # Reload org and preload users with their nested relationships
        org = await Organization.get(org.id)
        # This should load all users, but nested preloading on collections
        # requires special handling that we need to implement
        await org.preload(["users"])

        # Check that users are loaded
        assert hasattr(org, "_preloaded_users")
        users = org.users
        assert len(users) == 2

        # For now, nested preloading on collections would require manual iteration
        # This is a limitation we can address in future improvements

    @pytest.mark.asyncio
    async def test_mixed_nested_and_direct_preload(self, setup_nested_test_data):
        """Test mixing nested and direct preload paths."""
        data = await setup_nested_test_data
        user = data["users"][0]  # Alice

        # Reload user with mixed preload paths
        user = await User.get(user.id)
        await user.preload(["organization > subscription", "organization > projects"])

        # Check organization is loaded
        org = user.organization
        assert org is not None

        # Check both nested relationships are loaded
        assert hasattr(org, "_preloaded_subscription")
        assert hasattr(org, "_preloaded_projects")
        assert org.subscription.plan_name == "Enterprise"
        assert len(org.projects) == 2

    @pytest.mark.asyncio
    async def test_invalid_nested_path(self, setup_nested_test_data):
        """Test that invalid nested paths raise appropriate errors."""
        data = await setup_nested_test_data
        user = data["users"][0]

        # Reload user
        user = await User.get(user.id)

        # Try to preload an invalid nested path
        # This should not raise an error but simply skip invalid attributes
        await user.preload(["organization > invalid_attribute"])

        # Organization should still be loaded
        assert hasattr(user, "_preloaded_organization")
        org = user.organization
        assert org is not None

        # But the invalid attribute should not exist
        assert not hasattr(org, "_preloaded_invalid_attribute")

    @pytest.mark.asyncio
    async def test_one_to_one_nested_preload(self, setup_nested_test_data):
        """Test nested preloading with one-to-one relationships."""
        data = await setup_nested_test_data
        user = data["users"][0]  # Alice

        # Test user > profile (one-to-one)
        user = await User.get(user.id)
        await user.preload(["profile"])

        assert hasattr(user, "_preloaded_profile")
        profile = user.profile
        assert profile is not None
        assert profile.bio == "Senior Developer at Tech Corp"
        assert profile.avatar_url == "https://example.com/alice.jpg"

        # Test reverse one-to-one: profile > user
        profile = await Profile.get(profile.id)
        await profile.preload(["user"])

        assert hasattr(profile, "_preloaded_user")
        loaded_user = profile.user
        assert loaded_user is not None
        assert loaded_user.name == "Alice"
        assert loaded_user.email == "alice@techcorp.com"

    @pytest.mark.asyncio
    async def test_many_to_many_nested_preload(self, setup_nested_test_data):
        """Test nested preloading with many-to-many relationships."""
        data = await setup_nested_test_data
        project = data["projects"][0]  # Main Website

        # Test project > tags (many-to-many)
        project = await Project.get(project.id)
        await project.preload(["tags"])

        assert hasattr(project, "_preloaded_tags")
        tags = project.tags
        assert len(tags) == 2
        tag_names = [t.name for t in tags]
        assert "backend" in tag_names
        assert "frontend" in tag_names

        # Test reverse many-to-many: tag > projects
        tag = data["tags"][0]  # backend tag
        tag = await Tag.get(tag.id)
        await tag.preload(["projects"])

        assert hasattr(tag, "_preloaded_projects")
        projects = tag.projects
        assert len(projects) == 1
        assert projects[0].name == "Main Website"

        # Test user > assigned_projects (many-to-many with extra data)
        user = data["users"][0]  # Alice
        user = await User.get(user.id)
        await user.preload(["assigned_projects"])

        assert hasattr(user, "_preloaded_assigned_projects")
        assigned_projects = user.assigned_projects
        assert len(assigned_projects) == 2
        project_names = [p.name for p in assigned_projects]
        assert "Main Website" in project_names
        assert "Mobile App" in project_names

        # Test project > team_members (reverse many-to-many)
        project = await Project.get(data["projects"][0].id)
        await project.preload(["team_members"])

        assert hasattr(project, "_preloaded_team_members")
        team_members = project.team_members
        assert len(team_members) == 2
        member_names = [m.name for m in team_members]
        assert "Alice" in member_names
        assert "Bob" in member_names

    @pytest.mark.asyncio
    async def test_nested_with_one_to_one(self, setup_nested_test_data):
        """Test nested preloading through one-to-one relationships."""
        data = await setup_nested_test_data
        user = data["users"][0]  # Alice

        # Test user > profile in nested path
        user = await User.get(user.id)
        await user.preload(["profile", "organization"])

        # Verify both relationships are loaded
        assert hasattr(user, "_preloaded_profile")
        assert hasattr(user, "_preloaded_organization")

        profile = user.profile
        org = user.organization
        assert profile.bio == "Senior Developer at Tech Corp"
        assert org.name == "Tech Corp"

    @pytest.mark.asyncio
    async def test_nested_with_many_to_many(self, setup_nested_test_data):
        """Test nested preloading through many-to-many relationships."""
        data = await setup_nested_test_data
        user = data["users"][0]  # Alice

        # Test user > assigned_projects > tags (through many-to-many)
        user = await User.get(user.id)
        await user.preload(["assigned_projects > tags"])

        assert hasattr(user, "_preloaded_assigned_projects")
        projects = user.assigned_projects
        assert len(projects) == 2

        # Check that tags are loaded on each project
        for project in projects:
            assert hasattr(project, "_preloaded_tags")
            tags = project.tags
            assert len(tags) > 0

    @pytest.mark.asyncio
    async def test_complex_mixed_relationships(self, setup_nested_test_data):
        """Test complex nested preloading with mixed relationship types."""
        data = await setup_nested_test_data
        user = data["users"][0]  # Alice

        # Test: user > organization > projects > tags
        # This involves: many-to-one > one-to-many > many-to-many
        user = await User.get(user.id)
        await user.preload(
            [
                "profile",  # one-to-one
                "organization > subscription",  # many-to-one > many-to-one
                "organization > projects > tags",  # many-to-one > one-to-many > many-to-many
                "assigned_projects",  # many-to-many
            ]
        )

        # Verify all relationships are loaded
        assert hasattr(user, "_preloaded_profile")
        assert hasattr(user, "_preloaded_organization")
        assert hasattr(user, "_preloaded_assigned_projects")

        # Check nested preloading
        org = user.organization
        assert hasattr(org, "_preloaded_subscription")
        assert hasattr(org, "_preloaded_projects")

        # Check deeply nested preloading
        projects = org.projects
        for project in projects:
            assert hasattr(project, "_preloaded_tags")
            tags = project.tags
            assert len(tags) > 0

    @pytest.mark.asyncio
    async def test_many_to_many_parent_with_nested_preload(
        self, setup_nested_test_data
    ):
        """Test nested preloading when parent is many-to-many (user.assigned_projects > organization > subscription)."""
        data = await setup_nested_test_data
        user = data["users"][0]  # Alice

        # Test: user > assigned_projects > organization > subscription
        # This involves: many-to-many > many-to-one > many-to-one
        user = await User.get(user.id)
        await user.preload(["assigned_projects > organization > subscription"])

        # Verify projects are loaded
        assert hasattr(user, "_preloaded_assigned_projects")
        projects = user.assigned_projects
        assert len(projects) == 2

        # Check that organization is loaded on each project
        for project in projects:
            assert hasattr(project, "_preloaded_organization")
            org = project.organization
            assert org is not None
            assert org.name == "Tech Corp"

            # Check that subscription is loaded on the organization
            assert hasattr(org, "_preloaded_subscription")
            subscription = org.subscription
            assert subscription is not None
            assert subscription.plan_name == "Enterprise"

    def test_sync_many_to_many_parent_with_nested_preload(self, setup_nested_test_data):
        """Test synchronous nested preloading when parent is many-to-many."""
        # Run async setup synchronously
        data = asyncio.run(setup_nested_test_data)
        user = data["users"][0]  # Alice

        # Test: user > assigned_projects > organization > subscription (synchronous)
        user = User.sync_get(user.id)
        user.sync_preload(["assigned_projects > organization > subscription"])

        # Verify projects are loaded
        assert hasattr(user, "_preloaded_assigned_projects")
        projects = user.assigned_projects
        assert len(projects) == 2

        # Check that organization is loaded on each project
        for project in projects:
            assert hasattr(project, "_preloaded_organization")
            org = project.organization
            assert org is not None
            assert org.name == "Tech Corp"

            # Check that subscription is loaded on the organization
            assert hasattr(org, "_preloaded_subscription")
            subscription = org.subscription
            assert subscription is not None
            assert subscription.plan_name == "Enterprise"
