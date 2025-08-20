from __future__ import annotations

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Self,
    Type,
    TypeVar,
    cast,
    Tuple,
)
import json
import os
import threading
import asyncio
import inspect

import psycopg
from psycopg.rows import DictRow
from psycopg.types.json import Json, Jsonb

from reson.data.postgres.manager import DatabaseManager

T = TypeVar("T", bound="DBModel")
JT = TypeVar("JT", bound="JoinTableDBModel")
R = TypeVar("R")

# Type aliases for nested preload parsing
PreloadPath = List[str]  # e.g., ["organization", "subscription", "plan"]
PreloadTree = Dict[
    str, Optional["PreloadTree"]
]  # Recursive structure for nested preloads


class ModelRegistry:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not ModelRegistry._initialized:
            self.models = {}
            ModelRegistry._initialized = True

    def register_model(self, name: str, model: Type[DBModel]):
        self.models[name] = model

    def delete_model(self, name: str):
        del self.models[name]

    def get_model(self, name: str) -> Type[DBModel]:
        return self.models[name]


class PreloadAttribute(Generic[R]):
    def __init__(
        self,
        func: Optional[Callable[[Any], R]] = None,
        *,
        preload: bool = False,
        foreign_key: Optional[str] = None,  # FK column in this table
        references: Optional[str] = None,  # Referenced table name
        ref_column: Optional[str] = "id",  # Column in referenced table
        reverse_fk: Optional[str] = None,  # FK column in other table pointing to us
        relationship_type: Optional[
            str
        ] = None,  # 'one_to_one', 'one_to_many', 'many_to_one', 'many_to_many'
        join_table: Optional[str] = None,  # Join table for many-to-many
        join_fk: Optional[str] = None,  # FK in join table pointing to us
        model: Optional[str] = None,
        join_ref_fk: Optional[
            str
        ] = None,  # FK in join table pointing to referenced table
    ):
        self.preload = preload
        self.lock = threading.Lock()  # Always initialize lock
        self.foreign_key = foreign_key
        self.references = references
        self.ref_column = ref_column
        self.reverse_fk = reverse_fk
        self.relationship_type = relationship_type
        self.join_table = join_table
        self.join_fk = join_fk
        self.join_ref_fk = join_ref_fk
        self.model_name = model
        self.registry = ModelRegistry()

        if func is not None:
            # Used as @LazyAttribute (without parentheses)
            self.func = func
            self.is_async = inspect.iscoroutinefunction(func)
        else:
            # Used as @LazyAttribute(preload=True) (with parentheses) - decorator factory
            self.func = None
            self.is_async = False

    @property
    def model(self) -> Type[DBModel]:
        if self.model_name:
            return self.registry.get_model(self.model_name)
        else:
            raise ValueError("No model set.")

    def __call__(self, func: Callable[[Any], R]) -> "PreloadAttribute[R]":
        """Support for decorator factory pattern @LazyAttribute(preload=True)"""
        if self.func is not None:
            raise RuntimeError("LazyAttribute already has a function")

        return PreloadAttribute(
            func,
            preload=self.preload,
            foreign_key=self.foreign_key,
            references=self.references,
            ref_column=self.ref_column,
            reverse_fk=self.reverse_fk,
            relationship_type=self.relationship_type,
            join_table=self.join_table,
            join_fk=self.join_fk,
            join_ref_fk=self.join_ref_fk,
            model=self.model_name,
        )

    def __get__(self, instance: Any, cls: Any) -> R:
        if instance is None:
            return self  # type: ignore[return-value]

        if self.func is None:
            raise RuntimeError("LazyAttribute not properly initialized with a function")

        preloaded_attr_name = f"_preloaded_{self.func.__name__}"

        # Check if we have preloaded data
        if hasattr(instance, preloaded_attr_name):
            return cast(R, getattr(instance, preloaded_attr_name))

        # Raise error if not preloaded - no lazy loading
        raise AttributeError(
            f"Attribute '{self.func.__name__}' was not preloaded. "
            f"Use get_with_preload/list_with_preload/find_by_with_preload with preload=['{self.func.__name__}'] "
            f"to load this relationship."
        )

    def set_preloaded_value(self, instance: Any, value: R) -> None:
        """Set a preloaded value that bypasses lazy loading"""
        if self.func is None:
            raise RuntimeError("LazyAttribute not properly initialized")
        preloaded_attr_name = f"_preloaded_{self.func.__name__}"
        setattr(instance, preloaded_attr_name, value)

    def invalidate(self, instance: Any) -> None:
        """Invalidate both lazy and preloaded caches"""
        if self.func is None:
            raise RuntimeError("LazyAttribute not properly initialized")
        attr_name = f"_{self.func.__name__}"
        preloaded_attr_name = f"_preloaded_{self.func.__name__}"

        if hasattr(instance, attr_name):
            delattr(instance, attr_name)
        if hasattr(instance, preloaded_attr_name):
            delattr(instance, preloaded_attr_name)

    @classmethod
    def get_preloadable_attributes(
        cls, model_cls: Type[Any]
    ) -> Dict[str, "PreloadAttribute"]:
        """Get all LazyAttribute descriptors marked for preloading"""
        preloadable = {}
        for attr_name in dir(model_cls):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(model_cls, attr_name)
            if isinstance(attr_value, PreloadAttribute) and attr_value.preload:
                preloadable[attr_name] = attr_value
        return preloadable


class Column:
    def __init__(
        self,
        python_name: str,
        db_name: str,
        type: Any,
        nullable: bool = False,
    ):
        self.python_name = python_name
        self.db_name = db_name
        self.type = type
        self.nullable = nullable


# Singleton DB manager so you don’t create a new pool per call.
_DBM_SINGLETON: Optional[DatabaseManager] = None


class DBModel:
    TABLE_NAME: str
    COLUMNS: Dict[str, Column] = {}

    @classmethod
    def db_manager(cls) -> DatabaseManager:
        global _DBM_SINGLETON
        if _DBM_SINGLETON is None:
            _DBM_SINGLETON = DatabaseManager(
                dsn=os.environ.get(
                    "POSTGRES_DSN",
                    "postgresql://postgres:postgres@localhost:5432/quarkus",
                )
            )
        return _DBM_SINGLETON

    # ---------- nested preload helpers ----------

    @classmethod
    def _parse_preload_paths(cls, preload: List[str]) -> Dict[str, List[PreloadPath]]:
        """Parse preload strings into structured paths.

        Args:
            preload: List of preload strings like ["organization > subscription", "projects"]

        Returns:
            Dictionary mapping root attributes to their nested paths
            e.g., {"organization": [["organization", "subscription"]], "projects": [["projects"]]}
        """
        paths_by_root: Dict[str, List[PreloadPath]] = {}

        for preload_str in preload:
            # Split by ">" and strip whitespace
            parts = [part.strip() for part in preload_str.split(">")]

            if not parts or not parts[0]:
                continue

            root = parts[0]
            if root not in paths_by_root:
                paths_by_root[root] = []

            paths_by_root[root].append(parts)

        return paths_by_root

    @classmethod
    def _build_preload_tree(cls, paths: List[PreloadPath]) -> PreloadTree:
        """Convert list of paths to a tree structure for recursive processing.

        Args:
            paths: List of paths like [["organization", "subscription"], ["organization", "users"]]

        Returns:
            Tree structure like {"organization": {"subscription": {}, "users": {}}}
        """
        tree: PreloadTree = {}

        for path in paths:
            current: PreloadTree = tree
            for part in path:
                if current is not None and part not in current:
                    current[part] = {}
                if current is not None:
                    current = current.get(part)  # type: ignore

        return tree

    @classmethod
    async def _apply_nested_preloads(
        cls,
        instances: List[T],
        tree: PreloadTree,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> None:
        """Recursively apply nested preloads to a list of instances.

        Args:
            instances: List of instances to preload relationships on
            tree: Nested preload tree structure
            cursor: Optional database cursor
        """
        if not instances or not tree:
            return

        # Process each root-level attribute in the tree
        for attr_name, subtree in tree.items():
            if not hasattr(cls, attr_name):
                continue

            # Preload the current level attribute for all instances
            for instance in instances:
                await instance.preload([attr_name], cursor=cursor, refresh=False)

            # If there are nested preloads, process them recursively
            if subtree:
                # Collect all related objects from this level
                related_objects = []
                for instance in instances:
                    preloaded_attr_name = f"_preloaded_{attr_name}"
                    if hasattr(instance, preloaded_attr_name):
                        value = getattr(instance, preloaded_attr_name)
                        if value is not None:
                            if isinstance(value, list):
                                related_objects.extend(value)
                            else:
                                related_objects.append(value)

                # If we have related objects, recursively preload their relationships
                if related_objects:
                    # Get the model class for the related objects
                    lazy_attr = getattr(cls, attr_name)
                    if isinstance(lazy_attr, PreloadAttribute) and lazy_attr.model:
                        await lazy_attr.model._apply_nested_preloads(
                            related_objects, subtree, cursor
                        )

    @classmethod
    def _sync_apply_nested_preloads(
        cls,
        instances: List[T],
        tree: PreloadTree,
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> None:
        """Synchronous version of _apply_nested_preloads."""
        if not instances or not tree:
            return

        # Process each root-level attribute in the tree
        for attr_name, subtree in tree.items():
            if not hasattr(cls, attr_name):
                continue

            # Preload the current level attribute for all instances
            for instance in instances:
                instance.sync_preload([attr_name], cursor=cursor, refresh=False)

            # If there are nested preloads, process them recursively
            if subtree:
                # Collect all related objects from this level
                related_objects = []
                for instance in instances:
                    preloaded_attr_name = f"_preloaded_{attr_name}"
                    if hasattr(instance, preloaded_attr_name):
                        value = getattr(instance, preloaded_attr_name)
                        if value is not None:
                            if isinstance(value, list):
                                related_objects.extend(value)
                            else:
                                related_objects.append(value)

                # If we have related objects, recursively preload their relationships
                if related_objects:
                    # Get the model class for the related objects
                    lazy_attr = getattr(cls, attr_name)
                    if isinstance(lazy_attr, PreloadAttribute) and lazy_attr.model:
                        lazy_attr.model._sync_apply_nested_preloads(
                            related_objects, subtree, cursor
                        )

    # ---------- row helpers ----------

    @staticmethod
    def _row_get_case_insensitive(row: Mapping[str, Any], key: str) -> Any:
        if key in row:
            return row[key]
        lk = key.lower()
        if lk in row:
            return row[lk]
        uk = key.upper()
        return row.get(uk)

    @classmethod
    def from_db_row(cls: Type[T], row: Mapping[str, Any]) -> T:
        python_dict: Dict[str, Any] = {}
        for col in cls.COLUMNS.values():
            raw_val = cls._row_get_case_insensitive(row, col.db_name)
            if raw_val is None:
                python_dict[col.python_name] = None
                continue

            # JSON columns are already native Python on read
            if col.type in (Json, Jsonb):
                python_dict[col.python_name] = raw_val
                continue

            if not isinstance(raw_val, col.type):
                try:
                    python_dict[col.python_name] = col.type(raw_val)
                except Exception:
                    python_dict[col.python_name] = raw_val
            else:
                python_dict[col.python_name] = raw_val

        return cast(T, cls(**python_dict))  # type: ignore[arg-type]

    # ---------- read API (async) ----------

    @classmethod
    async def get(
        cls: Type[T],
        id: Any,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> T:
        query = f"SELECT * FROM {cls.TABLE_NAME} WHERE id = %s"
        row = await cls.db_manager().execute_and_fetch_one(
            query, params=(id,), cursor=cursor
        )
        if row is None:
            raise ValueError(f"No {cls.__name__} found with id {id}")
        return cls.from_db_row(row)

    # ---------- read API (sync) ----------

    @classmethod
    def sync_get(
        cls: Type[T],
        id: Any,
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> T:
        """Synchronous version of get()"""
        query = f"SELECT * FROM {cls.TABLE_NAME} WHERE id = %s"
        row = cls.db_manager().sync_execute_and_fetch_one(
            query, params=(id,), cursor=cursor
        )
        if row is None:
            raise ValueError(f"No {cls.__name__} found with id {id}")
        return cls.from_db_row(row)

    @classmethod
    async def get_many(
        cls: Type[T],
        ids: List[Any],
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> List[T]:
        if not ids:
            return []
        placeholders = ",".join(["%s"] * len(ids))
        query = f"SELECT * FROM {cls.TABLE_NAME} WHERE id IN ({placeholders})"
        rows = await cls.db_manager().execute_query(
            query, params=tuple(ids), cursor=cursor
        )
        return [cls.from_db_row(r) for r in rows] if rows else []

    @classmethod
    def sync_get_many(
        cls: Type[T],
        ids: List[Any],
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> List[T]:
        """Synchronous version of get_many()"""
        if not ids:
            return []
        placeholders = ",".join(["%s"] * len(ids))
        query = f"SELECT * FROM {cls.TABLE_NAME} WHERE id IN ({placeholders})"
        rows = cls.db_manager().sync_execute_query(
            query, params=tuple(ids), cursor=cursor
        )
        return [cls.from_db_row(r) for r in rows] if rows else []

    @classmethod
    async def list(
        cls: Type[T],
        where: Optional[str] = None,
        order: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> List[T]:
        query = f"SELECT * FROM {cls.TABLE_NAME}"
        if where:
            query += f" WHERE {where}"
        if order:
            query += f" ORDER BY {order}"
        rows = await cls.db_manager().execute_query(query, params=params, cursor=cursor)
        return [cls.from_db_row(r) for r in rows] if rows else []

    @classmethod
    def sync_list(
        cls: Type[T],
        where: Optional[str] = None,
        order: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> List[T]:
        """Synchronous version of list()"""
        query = f"SELECT * FROM {cls.TABLE_NAME}"
        if where:
            query += f" WHERE {where}"
        if order:
            query += f" ORDER BY {order}"
        rows = cls.db_manager().sync_execute_query(query, params=params, cursor=cursor)
        return [cls.from_db_row(r) for r in rows] if rows else []

    @classmethod
    async def find_by(
        cls: Type[T],
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        where: List[str] = []
        values: List[Any] = []
        for column, value in kwargs.items():
            db_column = next(
                (
                    col.db_name
                    for col in cls.COLUMNS.values()
                    if col.python_name == column
                ),
                column,
            )
            where.append(f"{db_column} = %s")
            if isinstance(value, Enum):
                value = value.value
            values.append(value)

        query = (
            f"SELECT * FROM {cls.TABLE_NAME} "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY createdat DESC LIMIT 1"
        )
        row = await cls.db_manager().execute_and_fetch_one(
            query, tuple(values), cursor=cursor
        )
        return None if row is None else cls.from_db_row(row)

    @classmethod
    def sync_find_by(
        cls: Type[T],
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        """Synchronous version of find_by()"""
        where: List[str] = []
        values: List[Any] = []
        for column, value in kwargs.items():
            db_column = next(
                (
                    col.db_name
                    for col in cls.COLUMNS.values()
                    if col.python_name == column
                ),
                column,
            )
            where.append(f"{db_column} = %s")
            if isinstance(value, Enum):
                value = value.value
            values.append(value)

        query = (
            f"SELECT * FROM {cls.TABLE_NAME} "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY createdat DESC LIMIT 1"
        )
        row = cls.db_manager().sync_execute_and_fetch_one(
            query, tuple(values), cursor=cursor
        )
        return None if row is None else cls.from_db_row(row)

    # ---------- instance-level preloading (async) ----------

    async def preload(
        self,
        attributes: Optional[List[str]] = None,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
        refresh: bool = False,
    ) -> Self:
        """Force preload specified attributes or all PreloadAttributes on this instance.

        Supports nested preloading with syntax like "organization > subscription".

        Args:
            attributes: List of attribute names to preload. Supports nested syntax.
                       If None, preloads all PreloadAttributes.
            cursor: Optional database cursor to use for queries.
            refresh: If True, always fetch from database. If False (default), skip already loaded attributes.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If the entity hasn't been persisted (no id).
            AttributeError: If an attribute name doesn't exist or isn't a PreloadAttribute.
        """
        # Check if entity has been persisted
        if not hasattr(self, "id") or self.id is None:
            raise ValueError(
                f"Cannot preload attributes on unpersisted {self.__class__.__name__} instance. "
                "Entity must be persisted (have an id) before preloading."
            )

        # Get attributes to preload
        if attributes is None:
            # Get all PreloadAttributes
            preloadable = PreloadAttribute.get_preloadable_attributes(self.__class__)
            attributes = list(preloadable.keys())

        if not attributes:
            return self

        # Check if any attributes contain nested syntax
        has_nested = any(">" in attr for attr in attributes)

        if has_nested:
            # Parse nested paths and organize by root
            paths_by_root = self.__class__._parse_preload_paths(attributes)

            # Process each root attribute
            for root, paths in paths_by_root.items():
                # Build tree for this root's paths
                tree = self.__class__._build_preload_tree(paths)

                # Apply nested preloads starting from this instance
                await self.__class__._apply_nested_preloads([self], tree, cursor)

            return self

        # Original non-nested preloading logic
        for attr_name in attributes:
            if not hasattr(self.__class__, attr_name):
                raise AttributeError(
                    f"Attribute '{attr_name}' does not exist on {self.__class__.__name__}"
                )

            lazy_attr = getattr(self.__class__, attr_name)
            if not isinstance(lazy_attr, PreloadAttribute):
                raise AttributeError(
                    f"Attribute '{attr_name}' is not a PreloadAttribute on {self.__class__.__name__}"
                )

            # Skip if already loaded and refresh not requested
            if not refresh:
                preloaded_attr_name = f"_preloaded_{attr_name}"
                if hasattr(self, preloaded_attr_name):
                    continue

            # Load based on relationship type
            if lazy_attr.foreign_key and lazy_attr.references:
                # Many-to-one: this table has FK to other table
                # Map DB column name to Python attribute name
                python_attr = None
                for col in self.COLUMNS.values():
                    if col.db_name == lazy_attr.foreign_key:
                        python_attr = col.python_name
                        break

                if python_attr:
                    fk_value = getattr(self, python_attr, None)
                    if fk_value is not None:
                        query = f"SELECT * FROM {lazy_attr.references} WHERE {lazy_attr.ref_column} = %s"
                        row = await self.__class__.db_manager().execute_and_fetch_one(
                            query, params=(fk_value,), cursor=cursor
                        )
                        if row:
                            related_obj = lazy_attr.model.from_db_row(row)
                            lazy_attr.set_preloaded_value(self, related_obj)
                        else:
                            lazy_attr.set_preloaded_value(self, None)
                    else:
                        lazy_attr.set_preloaded_value(self, None)
                else:
                    # FK column not found in COLUMNS - set to None
                    lazy_attr.set_preloaded_value(self, None)

            elif lazy_attr.reverse_fk and lazy_attr.references:
                # One-to-many or One-to-one: other table has FK to this table
                query = f"SELECT * FROM {lazy_attr.references} WHERE {lazy_attr.reverse_fk} = %s"
                rows = await self.__class__.db_manager().execute_query(
                    query, params=(self.id,), cursor=cursor
                )
                if lazy_attr.relationship_type == "one_to_one":
                    # For one-to-one, return single entity or None
                    related_obj = lazy_attr.model.from_db_row(rows[0]) if rows else None
                    lazy_attr.set_preloaded_value(self, related_obj)
                else:
                    # For one-to-many, return list
                    related_objects = (
                        [lazy_attr.model.from_db_row(row) for row in rows]
                        if rows
                        else []
                    )
                    lazy_attr.set_preloaded_value(self, related_objects)

            elif lazy_attr.join_table and lazy_attr.references:
                # Many-to-many: relationship through join table
                join_fk = lazy_attr.join_fk or "source_id"
                join_ref_fk = lazy_attr.join_ref_fk or "target_id"

                query = (
                    f"SELECT r.* FROM {lazy_attr.references} r "
                    f"JOIN {lazy_attr.join_table} j ON j.{join_ref_fk} = r.{lazy_attr.ref_column} "
                    f"WHERE j.{join_fk} = %s"
                )
                rows = await self.__class__.db_manager().execute_query(
                    query, params=(self.id,), cursor=cursor
                )
                related_objects = (
                    [lazy_attr.model.from_db_row(row) for row in rows] if rows else []
                )
                lazy_attr.set_preloaded_value(self, related_objects)

        return self

    # ---------- preloading API (async) ----------

    @classmethod
    async def get_with_preload(
        cls: Type[T],
        id: Any,
        preload: Optional[List[str]] = None,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> T:
        """Get an instance with preloaded relationships using JOINs"""
        if not preload:
            return await cls.get(id, cursor=cursor)

        # Build JOIN query with all requested preloads using column aliases
        select_parts = []
        from_part = f"{cls.TABLE_NAME} t0"
        join_parts = []
        join_info = []

        # Add main table columns with t0. prefix
        for col in cls.COLUMNS.values():
            select_parts.append(f't0.{col.db_name} AS "t0.{col.db_name}"')

        for i, attr_name in enumerate(preload):
            if not hasattr(cls, attr_name):
                continue

            lazy_attr = getattr(cls, attr_name)
            if not isinstance(lazy_attr, PreloadAttribute):
                continue

            alias = f"t{i+1}"

            # Build JOIN based on relationship metadata
            if lazy_attr.foreign_key and lazy_attr.references:
                # Many-to-one: this table has FK to other table
                # Get columns for the referenced table by finding its model class

                for col in lazy_attr.model.COLUMNS.values():
                    select_parts.append(
                        f'{alias}.{col.db_name} AS "{alias}.{col.db_name}"'
                    )

                join_parts.append(
                    f"LEFT JOIN {lazy_attr.references} {alias} "
                    f"ON t0.{lazy_attr.foreign_key} = {alias}.{lazy_attr.ref_column}"
                )
                join_info.append((attr_name, alias, "many_to_one"))

            elif lazy_attr.reverse_fk and lazy_attr.references:
                # One-to-many or One-to-one: other table has FK to this table
                # Get columns for the referenced table
                for col in lazy_attr.model.COLUMNS.values():
                    select_parts.append(
                        f'{alias}.{col.db_name} AS "{alias}.{col.db_name}"'
                    )

                join_parts.append(
                    f"LEFT JOIN {lazy_attr.references} {alias} "
                    f"ON {alias}.{lazy_attr.reverse_fk} = t0.id"
                )
                # Check if this is a one-to-one relationship
                if lazy_attr.relationship_type == "one_to_one":
                    join_info.append((attr_name, alias, "one_to_one"))
                else:
                    join_info.append((attr_name, alias, "one_to_many"))

            elif lazy_attr.join_table and lazy_attr.references:
                # Many-to-many: relationship through join table
                # Get columns for the referenced table
                for col in lazy_attr.model.COLUMNS.values():
                    select_parts.append(
                        f'{alias}.{col.db_name} AS "{alias}.{col.db_name}"'
                    )

                join_alias = f"j{i+1}"
                join_parts.append(
                    f"LEFT JOIN {lazy_attr.join_table} {join_alias} "
                    f"ON {join_alias}.{lazy_attr.join_fk or 'source_id'} = t0.id"
                )
                join_parts.append(
                    f"LEFT JOIN {lazy_attr.references} {alias} "
                    f"ON {alias}.{lazy_attr.ref_column} = {join_alias}.{lazy_attr.join_ref_fk or 'target_id'}"
                )
                join_info.append((attr_name, alias, "many_to_many"))

        # Build final query
        query = f"SELECT {', '.join(select_parts)} FROM {from_part}"
        if join_parts:
            query += " " + " ".join(join_parts)
        query += " WHERE t0.id = %s"

        rows = await cls.db_manager().execute_query(query, params=(id,), cursor=cursor)

        if not rows:
            raise ValueError(f"No {cls.__name__} found with id {id}")

        # Parse results
        result = cls._parse_joined_results_single(rows, join_info)
        if result is None:
            raise ValueError(f"No {cls.__name__} found with id {id}")
        return result

    @classmethod
    async def list_with_preload(
        cls: Type[T],
        preload: Optional[List[str]] = None,
        where: Optional[str] = None,
        order: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> List[T]:
        """Get a list of instances with preloaded relationships using JOINs"""
        if not preload:
            return await cls.list(
                where=where, order=order, params=params, cursor=cursor
            )

        # Build JOIN query with all requested preloads using column aliases
        select_parts = []
        from_part = f"{cls.TABLE_NAME} t0"
        join_parts = []
        join_info = []

        # Add main table columns with t0. prefix
        for col in cls.COLUMNS.values():
            select_parts.append(f't0.{col.db_name} AS "t0.{col.db_name}"')

        for i, attr_name in enumerate(preload):
            if not hasattr(cls, attr_name):
                continue

            lazy_attr = getattr(cls, attr_name)
            if not isinstance(lazy_attr, PreloadAttribute):
                continue

            alias = f"t{i+1}"

            # Build JOIN based on relationship metadata
            if lazy_attr.foreign_key and lazy_attr.references:
                # Many-to-one: this table has FK to other table
                # Get columns for the referenced table
                for col in lazy_attr.model.COLUMNS.values():
                    select_parts.append(
                        f'{alias}.{col.db_name} AS "{alias}.{col.db_name}"'
                    )

                join_parts.append(
                    f"LEFT JOIN {lazy_attr.references} {alias} "
                    f"ON t0.{lazy_attr.foreign_key} = {alias}.{lazy_attr.ref_column}"
                )
                join_info.append((attr_name, alias, "many_to_one"))

            elif lazy_attr.reverse_fk and lazy_attr.references:
                # One-to-many: other table has FK to this table
                for col in lazy_attr.model.COLUMNS.values():
                    select_parts.append(
                        f'{alias}.{col.db_name} AS "{alias}.{col.db_name}"'
                    )

                join_parts.append(
                    f"LEFT JOIN {lazy_attr.references} {alias} "
                    f"ON {alias}.{lazy_attr.reverse_fk} = t0.id"
                )

                if lazy_attr.relationship_type == "one_to_many":
                    join_info.append((attr_name, alias, "one_to_many"))
                elif lazy_attr.relationship_type == "one_to_one":
                    join_info.append((attr_name, alias, "one_to_one"))
                else:
                    raise ValueError(
                        f"Invalid relationship type: {lazy_attr.relationship_type}"
                    )

            elif lazy_attr.join_table and lazy_attr.references:
                # Many-to-many: relationship through join table
                # Get columns for the referenced table
                for col in lazy_attr.model.COLUMNS.values():
                    select_parts.append(
                        f'{alias}.{col.db_name} AS "{alias}.{col.db_name}"'
                    )

                join_alias = f"j{i+1}"
                join_parts.append(
                    f"LEFT JOIN {lazy_attr.join_table} {join_alias} "
                    f"ON {join_alias}.{lazy_attr.join_fk or 'source_id'} = t0.id"
                )
                join_parts.append(
                    f"LEFT JOIN {lazy_attr.references} {alias} "
                    f"ON {alias}.{lazy_attr.ref_column} = {join_alias}.{lazy_attr.join_ref_fk or 'target_id'}"
                )
                join_info.append((attr_name, alias, "many_to_many"))

        # Build final query
        query = f"SELECT {', '.join(select_parts)} FROM {from_part}"
        if join_parts:
            query += " " + " ".join(join_parts)
        if where:
            # Simply prepend WHERE clause - it should already have proper formatting
            query += f" WHERE {where}"
        if order:
            # Add t0. prefix to column names in ORDER BY clause if not already prefixed
            order_parts = []
            for part in order.split(","):
                part = part.strip()
                # Check if it has DESC or ASC
                col_part = part.split()[0]
                if not col_part.startswith("t") and "." not in col_part:
                    part = part.replace(col_part, f"t0.{col_part}", 1)
                order_parts.append(part)
            order = ", ".join(order_parts)
            query += f" ORDER BY {order}"

        rows = await cls.db_manager().execute_query(query, params=params, cursor=cursor)

        if not rows:
            return []

        # Parse results
        return cls._parse_joined_results_list(rows, join_info)

    @classmethod
    def _parse_joined_results_single(
        cls: Type[T],
        rows: Sequence[Mapping[str, Any]],
        join_info: List[tuple[str, str, str]],
    ) -> Optional[T]:
        """Parse JOINed query results into instance with preloaded relationships"""
        if not rows:
            return None

        # Create main instance from first row's t0 columns
        # Handle both aliased (t0.id) and non-aliased column names
        main_cols = {}
        for k, v in rows[0].items():
            if k.startswith("t0."):
                # Strip the t0. prefix for aliased columns
                main_cols[k[3:]] = v
            elif not any(k.startswith(f"{alias}.") for _, alias, _ in join_info):
                # Non-aliased columns that don't belong to joined tables
                main_cols[k] = v
        instance = cls.from_db_row(main_cols)

        # Process each relationship
        for attr_name, alias, rel_type in join_info:
            if not hasattr(cls, attr_name):
                continue

            lazy_attr = getattr(cls, attr_name)
            if not isinstance(lazy_attr, PreloadAttribute):
                continue

            if rel_type == "many_to_one" or rel_type == "one_to_one":
                # Single related object
                related_cols = {
                    k.replace(f"{alias}.", ""): v
                    for k, v in rows[0].items()
                    if k.startswith(f"{alias}.")
                }
                if related_cols and any(v is not None for v in related_cols.values()):
                    related_obj = lazy_attr.model.from_db_row(related_cols)
                    lazy_attr.set_preloaded_value(instance, related_obj)
                else:
                    lazy_attr.set_preloaded_value(instance, None)

            elif rel_type in ("one_to_many", "many_to_many"):
                # Multiple related objects
                related_objects = []
                seen_ids = set()

                for row in rows:
                    related_cols = {
                        k.replace(f"{alias}.", ""): v
                        for k, v in row.items()
                        if k.startswith(f"{alias}.")
                    }
                    if related_cols and any(
                        v is not None for v in related_cols.values()
                    ):
                        # Check if we've seen this ID already (avoid duplicates)
                        obj_id = related_cols.get("id")
                        if obj_id and obj_id not in seen_ids:
                            seen_ids.add(obj_id)
                            # Get the related model class from references
                            related_objects.append(
                                lazy_attr.model.from_db_row(related_cols)
                            )

                lazy_attr.set_preloaded_value(instance, related_objects)

        return instance

    @classmethod
    def _parse_joined_results_list(
        cls: Type[T],
        rows: Sequence[Mapping[str, Any]],
        join_info: List[tuple[str, str, str]],
    ) -> List[T]:
        """Parse JOINed query results into instances with preloaded relationships"""
        if not rows:
            return []

        instances_map: Dict[Any, T] = {}
        relationships_map: Dict[Any, Dict[str, Any]] = {}

        for row in rows:
            # Extract main instance columns
            # Handle both aliased (t0.id) and non-aliased column names
            main_cols = {}
            for k, v in row.items():
                if k.startswith("t0."):
                    # Strip the t0. prefix for aliased columns
                    main_cols[k[3:]] = v
                elif not any(k.startswith(f"{alias}.") for _, alias, _ in join_info):
                    # Non-aliased columns that don't belong to joined tables
                    main_cols[k] = v
            main_id = main_cols.get("id")

            if main_id not in instances_map:
                instances_map[main_id] = cls.from_db_row(main_cols)
                relationships_map[main_id] = {
                    attr_name: (
                        [] if rel_type in ("one_to_many", "many_to_many") else None
                    )
                    for attr_name, _, rel_type in join_info
                }

            instance = instances_map[main_id]

            # Process each relationship
            for attr_name, alias, rel_type in join_info:
                if not hasattr(cls, attr_name):
                    continue

                lazy_attr = getattr(cls, attr_name)
                if not isinstance(lazy_attr, PreloadAttribute):
                    continue

                related_cols = {
                    k.replace(f"{alias}.", ""): v
                    for k, v in row.items()
                    if k.startswith(f"{alias}.")
                }

                if related_cols and any(v is not None for v in related_cols.values()):
                    if rel_type == "many_to_one" or rel_type == "one_to_one":
                        # Single related object
                        if relationships_map[main_id][attr_name] is None:
                            relationships_map[main_id][attr_name] = (
                                lazy_attr.model.from_db_row(related_cols)
                            )

                    elif rel_type in ("one_to_many", "many_to_many"):
                        # Multiple related objects
                        obj_id = related_cols.get("id")
                        existing_ids = {
                            getattr(o, "id", None)
                            for o in relationships_map[main_id][attr_name]
                        }
                        if obj_id and obj_id not in existing_ids:
                            relationships_map[main_id][attr_name].append(
                                lazy_attr.model.from_db_row(related_cols)
                            )

        # Set all preloaded values
        for main_id, instance in instances_map.items():
            for attr_name, _, _ in join_info:
                if hasattr(cls, attr_name):
                    lazy_attr = getattr(cls, attr_name)
                    if isinstance(lazy_attr, PreloadAttribute):
                        value = relationships_map[main_id].get(attr_name)
                        if value is not None or attr_name in relationships_map[main_id]:
                            lazy_attr.set_preloaded_value(instance, value)

        return list(instances_map.values())

    @classmethod
    async def find_by_with_preload(
        cls: Type[T],
        preload: Optional[List[str]] = None,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        """Find by with preloaded relationships using JOINs"""
        where: List[str] = []
        values: List[Any] = []
        for column, value in kwargs.items():
            db_column = next(
                (
                    col.db_name
                    for col in cls.COLUMNS.values()
                    if col.python_name == column
                ),
                column,
            )
            where.append(f"t0.{db_column} = %s")
            if isinstance(value, Enum):
                value = value.value
            values.append(value)

        where_clause = " AND ".join(where)
        order_clause = "t0.createdat DESC"

        instances = await cls.list_with_preload(
            preload=preload,
            where=where_clause,
            order=order_clause,
            params=tuple(values),
            cursor=cursor,
        )

        return instances[0] if instances else None

    # ---------- instance-level preloading (sync) ----------

    def sync_preload(
        self,
        attributes: Optional[List[str]] = None,
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
        refresh: bool = False,
    ) -> Self:
        """Synchronous version of preload(). Force preload specified attributes or all PreloadAttributes on this instance.

        Supports nested preloading with syntax like "organization > subscription".

        Args:
            attributes: List of attribute names to preload. Supports nested syntax.
                       If None, preloads all PreloadAttributes.
            cursor: Optional database cursor to use for queries.
            refresh: If True, always fetch from database. If False (default), skip already loaded attributes.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If the entity hasn't been persisted (no id).
            AttributeError: If an attribute name doesn't exist or isn't a PreloadAttribute.
        """
        # Check if entity has been persisted
        if not hasattr(self, "id") or self.id is None:
            raise ValueError(
                f"Cannot preload attributes on unpersisted {self.__class__.__name__} instance. "
                "Entity must be persisted (have an id) before preloading."
            )

        # Get attributes to preload
        if attributes is None:
            # Get all PreloadAttributes
            preloadable = PreloadAttribute.get_preloadable_attributes(self.__class__)
            attributes = list(preloadable.keys())

        if not attributes:
            return self

        # Check if any attributes contain nested syntax
        has_nested = any(">" in attr for attr in attributes)

        if has_nested:
            # Parse nested paths and organize by root
            paths_by_root = self.__class__._parse_preload_paths(attributes)

            # Process each root attribute
            for root, paths in paths_by_root.items():
                # Build tree for this root's paths
                tree = self.__class__._build_preload_tree(paths)

                # Apply nested preloads starting from this instance
                self.__class__._sync_apply_nested_preloads([self], tree, cursor)

            return self

        # Original non-nested preloading logic
        for attr_name in attributes:
            if not hasattr(self.__class__, attr_name):
                raise AttributeError(
                    f"Attribute '{attr_name}' does not exist on {self.__class__.__name__}"
                )

            lazy_attr = getattr(self.__class__, attr_name)
            if not isinstance(lazy_attr, PreloadAttribute):
                raise AttributeError(
                    f"Attribute '{attr_name}' is not a PreloadAttribute on {self.__class__.__name__}"
                )

            # Skip if already loaded and refresh not requested
            if not refresh:
                preloaded_attr_name = f"_preloaded_{attr_name}"
                if hasattr(self, preloaded_attr_name):
                    continue

            # Load based on relationship type
            if lazy_attr.foreign_key and lazy_attr.references:
                # Many-to-one: this table has FK to other table
                # Map DB column name to Python attribute name
                python_attr = None
                for col in self.COLUMNS.values():
                    if col.db_name == lazy_attr.foreign_key:
                        python_attr = col.python_name
                        break

                if python_attr:
                    fk_value = getattr(self, python_attr, None)
                    if fk_value is not None:
                        query = f"SELECT * FROM {lazy_attr.references} WHERE {lazy_attr.ref_column} = %s"
                        row = self.__class__.db_manager().sync_execute_and_fetch_one(
                            query, params=(fk_value,), cursor=cursor
                        )
                        if row:
                            related_obj = lazy_attr.model.from_db_row(row)
                            lazy_attr.set_preloaded_value(self, related_obj)
                        else:
                            lazy_attr.set_preloaded_value(self, None)
                    else:
                        lazy_attr.set_preloaded_value(self, None)
                else:
                    # FK column not found in COLUMNS - set to None
                    lazy_attr.set_preloaded_value(self, None)

            elif lazy_attr.reverse_fk and lazy_attr.references:
                # One-to-many or One-to-one: other table has FK to this table
                query = f"SELECT * FROM {lazy_attr.references} WHERE {lazy_attr.reverse_fk} = %s"
                rows = self.__class__.db_manager().sync_execute_query(
                    query, params=(self.id,), cursor=cursor
                )
                if lazy_attr.relationship_type == "one_to_one":
                    # For one-to-one, return single entity or None
                    related_obj = lazy_attr.model.from_db_row(rows[0]) if rows else None
                    lazy_attr.set_preloaded_value(self, related_obj)
                else:
                    # For one-to-many, return list
                    related_objects = (
                        [lazy_attr.model.from_db_row(row) for row in rows]
                        if rows
                        else []
                    )
                    lazy_attr.set_preloaded_value(self, related_objects)

            elif lazy_attr.join_table and lazy_attr.references:
                # Many-to-many: relationship through join table
                join_fk = lazy_attr.join_fk or "source_id"
                join_ref_fk = lazy_attr.join_ref_fk or "target_id"

                query = (
                    f"SELECT r.* FROM {lazy_attr.references} r "
                    f"JOIN {lazy_attr.join_table} j ON j.{join_ref_fk} = r.{lazy_attr.ref_column} "
                    f"WHERE j.{join_fk} = %s"
                )
                rows = self.__class__.db_manager().sync_execute_query(
                    query, params=(self.id,), cursor=cursor
                )
                related_objects = (
                    [lazy_attr.model.from_db_row(row) for row in rows] if rows else []
                )
                lazy_attr.set_preloaded_value(self, related_objects)

        return self

    # ---------- preloading API (sync) ----------

    @classmethod
    def sync_get_with_preload(
        cls: Type[T],
        id: Any,
        preload: Optional[List[str]] = None,
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> T:
        """Synchronous version of get_with_preload()"""
        if not preload:
            return cls.sync_get(id, cursor=cursor)

        # Build JOIN query with all requested preloads using column aliases
        select_parts = []
        from_part = f"{cls.TABLE_NAME} t0"
        join_parts = []
        join_info = []

        # Add main table columns with t0. prefix
        for col in cls.COLUMNS.values():
            select_parts.append(f't0.{col.db_name} AS "t0.{col.db_name}"')

        for i, attr_name in enumerate(preload):
            if not hasattr(cls, attr_name):
                continue

            lazy_attr = getattr(cls, attr_name)
            if not isinstance(lazy_attr, PreloadAttribute):
                continue

            alias = f"t{i+1}"

            # Build JOIN based on relationship metadata
            if lazy_attr.foreign_key and lazy_attr.references:
                # Many-to-one: this table has FK to other table
                # Get columns for the referenced table by finding its model class
                for col in lazy_attr.model.COLUMNS.values():
                    select_parts.append(
                        f'{alias}.{col.db_name} AS "{alias}.{col.db_name}"'
                    )

                join_parts.append(
                    f"LEFT JOIN {lazy_attr.references} {alias} "
                    f"ON t0.{lazy_attr.foreign_key} = {alias}.{lazy_attr.ref_column}"
                )
                join_info.append((attr_name, alias, "many_to_one"))

            elif lazy_attr.reverse_fk and lazy_attr.references:
                # One-to-many or One-to-one: other table has FK to this table
                # Get columns for the referenced table
                for col in lazy_attr.model.COLUMNS.values():
                    select_parts.append(
                        f'{alias}.{col.db_name} AS "{alias}.{col.db_name}"'
                    )

                join_parts.append(
                    f"LEFT JOIN {lazy_attr.references} {alias} "
                    f"ON {alias}.{lazy_attr.reverse_fk} = t0.id"
                )
                # Check if this is a one-to-one relationship
                if lazy_attr.relationship_type == "one_to_one":
                    join_info.append((attr_name, alias, "one_to_one"))
                else:
                    join_info.append((attr_name, alias, "one_to_many"))

        # Build final query
        query = f"SELECT {', '.join(select_parts)} FROM {from_part}"
        if join_parts:
            query += " " + " ".join(join_parts)
        query += " WHERE t0.id = %s"

        rows = cls.db_manager().sync_execute_query(query, params=(id,), cursor=cursor)

        if not rows:
            raise ValueError(f"No {cls.__name__} found with id {id}")

        # Parse results
        result = cls._parse_joined_results_single(rows, join_info)
        if result is None:
            raise ValueError(f"No {cls.__name__} found with id {id}")
        return result

    @classmethod
    def sync_list_with_preload(
        cls: Type[T],
        preload: Optional[List[str]] = None,
        where: Optional[str] = None,
        order: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> List[T]:
        """Synchronous version of list_with_preload()"""
        if not preload:
            return cls.sync_list(where=where, order=order, params=params, cursor=cursor)

        # Build JOIN query with all requested preloads using column aliases
        select_parts = []
        from_part = f"{cls.TABLE_NAME} t0"
        join_parts = []
        join_info = []

        # Add main table columns with t0. prefix
        for col in cls.COLUMNS.values():
            select_parts.append(f't0.{col.db_name} AS "t0.{col.db_name}"')

        for i, attr_name in enumerate(preload):
            if not hasattr(cls, attr_name):
                continue

            lazy_attr = getattr(cls, attr_name)
            if not isinstance(lazy_attr, PreloadAttribute):
                continue

            alias = f"t{i+1}"

            # Build JOIN based on relationship metadata
            if lazy_attr.foreign_key and lazy_attr.references:
                # Many-to-one: this table has FK to other table
                # Get columns for the referenced table
                for col in lazy_attr.model.COLUMNS.values():
                    select_parts.append(
                        f'{alias}.{col.db_name} AS "{alias}.{col.db_name}"'
                    )

                join_parts.append(
                    f"LEFT JOIN {lazy_attr.references} {alias} "
                    f"ON t0.{lazy_attr.foreign_key} = {alias}.{lazy_attr.ref_column}"
                )
                join_info.append((attr_name, alias, "many_to_one"))

            elif lazy_attr.reverse_fk and lazy_attr.references:
                # One-to-many or One-to-one: other table has FK to this table
                # Get columns for the referenced table
                for col in lazy_attr.model.COLUMNS.values():
                    select_parts.append(
                        f'{alias}.{col.db_name} AS "{alias}.{col.db_name}"'
                    )

                join_parts.append(
                    f"LEFT JOIN {lazy_attr.references} {alias} "
                    f"ON {alias}.{lazy_attr.reverse_fk} = t0.id"
                )
                # Check if this is a one-to-one relationship
                if lazy_attr.relationship_type == "one_to_one":
                    join_info.append((attr_name, alias, "one_to_one"))
                else:
                    join_info.append((attr_name, alias, "one_to_many"))

        # Build final query
        query = f"SELECT {', '.join(select_parts)} FROM {from_part}"
        if join_parts:
            query += " " + " ".join(join_parts)
        if where:
            # Simply prepend WHERE clause - it should already have proper formatting
            query += f" WHERE {where}"
        if order:
            # Add t0. prefix to column names in ORDER BY clause if not already prefixed
            order_parts = []
            for part in order.split(","):
                part = part.strip()
                # Check if it has DESC or ASC
                col_part = part.split()[0]
                if not col_part.startswith("t") and "." not in col_part:
                    part = part.replace(col_part, f"t0.{col_part}", 1)
                order_parts.append(part)
            order = ", ".join(order_parts)
            query += f" ORDER BY {order}"

        rows = cls.db_manager().sync_execute_query(query, params=params, cursor=cursor)

        if not rows:
            return []

        # Parse results
        return cls._parse_joined_results_list(rows, join_info)

    @classmethod
    def sync_find_by_with_preload(
        cls: Type[T],
        preload: Optional[List[str]] = None,
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        """Synchronous version of find_by_with_preload()"""
        where: List[str] = []
        values: List[Any] = []
        for column, value in kwargs.items():
            db_column = next(
                (
                    col.db_name
                    for col in cls.COLUMNS.values()
                    if col.python_name == column
                ),
                column,
            )
            where.append(f"t0.{db_column} = %s")
            if isinstance(value, Enum):
                value = value.value
            values.append(value)

        where_clause = " AND ".join(where)
        order_clause = "t0.createdat DESC"

        instances = cls.sync_list_with_preload(
            preload=preload,
            where=where_clause,
            order=order_clause,
            params=tuple(values),
            cursor=cursor,
        )

        return instances[0] if instances else None

    # ---------- serialization helpers ----------

    def _get_preloaded_attributes(self) -> Dict[str, Any]:
        """Get all preloaded attributes on this instance.

        Returns:
            Dictionary mapping attribute names to their preloaded values.
        """
        preloaded = {}
        for attr_name in dir(self.__class__):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self.__class__, attr_name)
            if isinstance(attr_value, PreloadAttribute):
                preloaded_attr_name = f"_preloaded_{attr_name}"
                if hasattr(self, preloaded_attr_name):
                    preloaded[attr_name] = getattr(self, preloaded_attr_name)
        return preloaded

    def _should_serialize_relationship(self, visited: set) -> bool:
        """Check if this object should be serialized to prevent circular references.

        Args:
            visited: Set of (class, id) tuples that have been visited.

        Returns:
            True if object hasn't been visited, False if already serialized.
        """
        if not hasattr(self, "id") or self.id is None:
            # Objects without IDs can't be tracked for circular references
            return True

        key = (self.__class__, self.id)
        if key in visited:
            return False
        visited.add(key)
        return True

    def to_db_dict(self) -> Dict[str, Any]:
        db_dict = {
            col.db_name: getattr(self, col.python_name)
            for col in self.COLUMNS.values()
            if hasattr(self, col.python_name)
        }

        for col in self.COLUMNS.values():
            val = db_dict.get(col.db_name)
            if col.type in (Json, Jsonb) and val is not None:
                # Wrap for sending (prefer jsonb on write)
                if not isinstance(val, (Json, Jsonb)):
                    db_dict[col.db_name] = Jsonb(val)
            elif isinstance(val, Enum):
                db_dict[col.db_name] = val.value

        return db_dict

    async def to_json_dict(
        self, cache: Optional[Dict[Tuple, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Convert model instance to JSON-serializable dictionary.

        Args:
            cache: Dictionary mapping (class, id) tuples to their serialized form to prevent circular references.

        Returns:
            Dictionary with all columns and preloaded relationships serialized.
        """
        if cache is None:
            cache = {}

        # Check if we've already serialized this object
        my_key = (
            (self.__class__, self.id)
            if hasattr(self, "id") and self.id is not None
            else None
        )

        if my_key and my_key in cache:
            # Return the cached serialized version
            return cache[my_key]

        new_dict = {}

        # Add to cache immediately with a placeholder to prevent infinite recursion
        if my_key:
            cache[my_key] = new_dict

        # Serialize columns
        for col in self.COLUMNS.values():
            if hasattr(self, col.python_name):
                try:
                    val = await getattr(self, col.python_name)
                    new_dict[col.db_name] = await self._serialize_value(val, cache)
                except Exception as e:
                    print(f"Error awaiting {col.python_name}: {e}")
                    try:
                        val = getattr(self, col.python_name)
                        new_dict[col.db_name] = await self._serialize_value(val, cache)
                    except Exception as e2:
                        print(f"Error serializing {col.python_name}: {e2}")
                        new_dict[col.db_name] = None

        # Add preloaded relationships
        preloaded = self._get_preloaded_attributes()
        for attr_name, value in preloaded.items():
            if value is None:
                # Include null relationships when preloaded
                new_dict[attr_name] = None
            elif isinstance(value, list):
                # Serialize list of related objects
                serialized_list = []
                for item in value:
                    if isinstance(item, DBModel):
                        serialized_list.append(await item.to_json_dict(cache))
                    else:
                        serialized_list.append(await self._serialize_value(item, cache))
                new_dict[attr_name] = serialized_list
            elif isinstance(value, DBModel):
                new_dict[attr_name] = await value.to_json_dict(cache)
            else:
                # Serialize other types
                new_dict[attr_name] = await self._serialize_value(value, cache)

        return new_dict

    async def _serialize_value(self, val, cache: Dict[Tuple, Dict[str, Any]]):
        """Recursively serialize complex types to JSON-compatible format.

        Args:
            val: Value to serialize.
            cache: Dictionary mapping (class, id) tuples to their serialized form.
        """
        if val is None:
            return None

        # Handle basic JSON-serializable types
        if isinstance(val, (str, int, float, bool)):
            return val

        # Handle datetime objects
        if isinstance(val, (datetime, date)):
            return val.isoformat()

        # Handle Decimal
        if isinstance(val, Decimal):
            return float(val)

        # Handle lists/tuples
        if isinstance(val, (list, tuple)):
            return [await self._serialize_value(item, cache) for item in val]

        # Handle dictionaries
        if isinstance(val, dict):
            return {k: await self._serialize_value(v, cache) for k, v in val.items()}

        # Handle DBModel objects
        if isinstance(val, DBModel):
            return await val.to_json_dict(cache)

        # Handle other objects with to_json_dict method
        if hasattr(val, "to_json_dict") and callable(getattr(val, "to_json_dict")):
            return await val.to_json_dict()

        # Handle objects with __dict__ (but avoid internal attributes)
        if hasattr(val, "__dict__"):
            obj_dict = {}
            for attr_name, attr_val in val.__dict__.items():
                if not attr_name.startswith("_") and not attr_name.startswith("__"):
                    try:
                        obj_dict[attr_name] = await self._serialize_value(
                            attr_val, cache
                        )
                    except:
                        # Skip attributes that can't be serialized
                        pass
            return obj_dict

        # Last resort: try to convert to string or return None
        try:
            # Check if it's JSON serializable as-is
            json.dumps(val)
            return val
        except (TypeError, ValueError):
            try:
                return str(val)
            except:
                return None

    def sync_to_json_dict(
        self, cache: Optional[Dict[Tuple, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Synchronous version of to_json_dict().

        Args:
            cache: Dictionary mapping (class, id) tuples to their serialized form to prevent circular references.

        Returns:
            Dictionary with all columns and preloaded relationships serialized.
        """
        if cache is None:
            cache = {}

        # Check if we've already serialized this object
        my_key = (
            (self.__class__, self.id)
            if hasattr(self, "id") and self.id is not None
            else None
        )

        if my_key and my_key in cache:
            # Return the cached serialized version
            return cache[my_key]

        new_dict = {}

        # Add to cache immediately with a placeholder to prevent infinite recursion
        if my_key:
            cache[my_key] = new_dict

        # Serialize columns
        for col in self.COLUMNS.values():
            if hasattr(self, col.python_name):
                try:
                    val = getattr(self, col.python_name)
                    new_dict[col.db_name] = self._sync_serialize_value(val, cache)
                except Exception as e:
                    print(f"Error serializing {col.python_name}: {e}")
                    new_dict[col.db_name] = None

        # Add preloaded relationships
        preloaded = self._get_preloaded_attributes()
        for attr_name, value in preloaded.items():
            if value is None:
                # Include null relationships when preloaded
                new_dict[attr_name] = None
            elif isinstance(value, list):
                # Serialize list of related objects
                serialized_list = []
                for item in value:
                    if isinstance(item, DBModel):
                        serialized_list.append(item.sync_to_json_dict(cache))
                    else:
                        serialized_list.append(self._sync_serialize_value(item, cache))
                new_dict[attr_name] = serialized_list
            elif isinstance(value, DBModel):
                new_dict[attr_name] = value.sync_to_json_dict(cache)
            else:
                # Serialize other types
                new_dict[attr_name] = self._sync_serialize_value(value, cache)

        return new_dict

    def _sync_serialize_value(self, val, cache: Dict[Tuple, Dict[str, Any]]):
        """Synchronous version of _serialize_value().

        Args:
            val: Value to serialize.
            cache: Dictionary mapping (class, id) tuples to their serialized form.
        """
        if val is None:
            return None

        # Handle basic JSON-serializable types
        if isinstance(val, (str, int, float, bool)):
            return val

        # Handle datetime objects
        if isinstance(val, (datetime, date)):
            return val.isoformat()

        # Handle Decimal
        if isinstance(val, Decimal):
            return float(val)

        # Handle lists/tuples
        if isinstance(val, (list, tuple)):
            return [self._sync_serialize_value(item, cache) for item in val]

        # Handle dictionaries
        if isinstance(val, dict):
            return {k: self._sync_serialize_value(v, cache) for k, v in val.items()}

        # Handle DBModel objects
        if isinstance(val, DBModel):
            return val.sync_to_json_dict(cache)

        # Handle other model objects with sync_to_json_dict method
        if hasattr(val, "sync_to_json_dict") and callable(
            getattr(val, "sync_to_json_dict")
        ):
            return val.sync_to_json_dict()

        # Handle other model objects with to_json_dict method (sync)
        if hasattr(val, "to_json_dict") and callable(getattr(val, "to_json_dict")):
            # Try to call synchronously if it's not async
            try:
                result = val.to_json_dict()
                if not asyncio.iscoroutine(result):
                    return result
            except:
                pass

        # Handle objects with __dict__
        if hasattr(val, "__dict__"):
            obj_dict = {}
            for attr_name, attr_val in val.__dict__.items():
                if not attr_name.startswith("_") and not attr_name.startswith("__"):
                    try:
                        obj_dict[attr_name] = self._sync_serialize_value(
                            attr_val, cache
                        )
                    except:
                        # Skip attributes that can't be serialized
                        pass
            return obj_dict

        # Last resort: try to convert to string or return None
        try:
            # Check if it's JSON serializable as-is
            json.dumps(val)
            return val
        except (TypeError, ValueError):
            try:
                return str(val)
            except:
                return None

    # ---------- write API (async) ----------

    async def update(
        self, cursor: Optional[psycopg.AsyncCursor[DictRow]] = None
    ) -> None:
        db_dict = self.to_db_dict()
        if "updatedat" in db_dict:
            db_dict["updatedat"] = datetime.now()

        set_clause = ", ".join(f"{k} = %s" for k in db_dict.keys() if k != "id")
        values = [
            (v if not isinstance(v, Enum) else v.value)
            for k, v in db_dict.items()
            if k != "id"
        ]
        values.append(self.id)

        query = f"UPDATE {self.__class__.TABLE_NAME} SET {set_clause} WHERE id = %s"

        if cursor is not None:
            await cursor.execute(cast(Any, query), tuple(values))
        else:
            async with self.__class__.db_manager().get_cursor() as cur:
                await cur.execute(cast(Any, query), tuple(values))

    # ---------- write API (sync) ----------

    def sync_update(self, cursor: Optional[psycopg.Cursor[DictRow]] = None) -> None:
        """Synchronous version of update()"""
        db_dict = self.to_db_dict()
        if "updatedat" in db_dict:
            db_dict["updatedat"] = datetime.now()

        set_clause = ", ".join(f"{k} = %s" for k in db_dict.keys() if k != "id")
        values = [
            (v if not isinstance(v, Enum) else v.value)
            for k, v in db_dict.items()
            if k != "id"
        ]
        values.append(self.id)

        query = f"UPDATE {self.__class__.TABLE_NAME} SET {set_clause} WHERE id = %s"

        if cursor is not None:
            cursor.execute(cast(Any, query), tuple(values))
        else:
            with self.__class__.db_manager().sync_get_cursor() as cur:
                cur.execute(cast(Any, query), tuple(values))

    @classmethod
    async def delete(
        cls,
        id: int,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> None:
        query = f"DELETE FROM {cls.TABLE_NAME} WHERE id = %s"
        params = (id,)
        if cursor is not None:
            await cursor.execute(cast(Any, query), params)
        else:
            async with cls.db_manager().get_cursor() as cur:
                await cur.execute(cast(Any, query), params)

    @classmethod
    def sync_delete(
        cls,
        id: int,
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> None:
        """Synchronous version of delete()"""
        query = f"DELETE FROM {cls.TABLE_NAME} WHERE id = %s"
        params = (id,)
        if cursor is not None:
            cursor.execute(cast(Any, query), params)
        else:
            with cls.db_manager().sync_get_cursor() as cur:
                cur.execute(cast(Any, query), params)

    @classmethod
    async def delete_many(
        cls,
        ids: List[int],
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> None:
        if not ids:
            return
        placeholders = ",".join(["%s"] * len(ids))
        query = f"DELETE FROM {cls.TABLE_NAME} WHERE id IN ({placeholders})"
        params = tuple(ids)
        if cursor is not None:
            await cursor.execute(cast(Any, query), params)
        else:
            async with cls.db_manager().get_cursor() as cur:
                await cur.execute(cast(Any, query), params)

    @classmethod
    def sync_delete_many(
        cls,
        ids: List[int],
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> None:
        """Synchronous version of delete_many()"""
        if not ids:
            return
        placeholders = ",".join(["%s"] * len(ids))
        query = f"DELETE FROM {cls.TABLE_NAME} WHERE id IN ({placeholders})"
        params = tuple(ids)
        if cursor is not None:
            cursor.execute(cast(Any, query), params)
        else:
            with cls.db_manager().sync_get_cursor() as cur:
                cur.execute(cast(Any, query), params)

    @classmethod
    async def insert_many(
        cls: Type[T],
        items: List[T],
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> None:
        if not items:
            return
        columns = [c.db_name for c in cls.COLUMNS.values()]
        if "id" in columns:
            columns.remove("id")  # id is auto-generated
        placeholders = ", ".join(["%s"] * len(columns))
        query = f"INSERT INTO {cls.TABLE_NAME} ({', '.join(columns)}) VALUES ({placeholders})"
        values_matrix = [
            tuple(cast(Dict[str, Any], i.to_db_dict()).get(c) for c in columns)
            for i in items
        ]

        if cursor is not None:
            await cursor.executemany(cast(Any, query), values_matrix)
        else:
            async with cls.db_manager().get_cursor() as cur:
                await cur.executemany(cast(Any, query), values_matrix)

    @classmethod
    def sync_insert_many(
        cls: Type[T],
        items: List[T],
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> None:
        """Synchronous version of insert_many()"""
        if not items:
            return
        columns = [c.db_name for c in cls.COLUMNS.values()]
        if "id" in columns:
            columns.remove("id")  # id is auto-generated
        placeholders = ", ".join(["%s"] * len(columns))
        query = f"INSERT INTO {cls.TABLE_NAME} ({', '.join(columns)}) VALUES ({placeholders})"
        values_matrix = [
            tuple(cast(Dict[str, Any], i.to_db_dict()).get(c) for c in columns)
            for i in items
        ]

        if cursor is not None:
            cursor.executemany(cast(Any, query), values_matrix)
        else:
            with cls.db_manager().sync_get_cursor() as cur:
                cur.executemany(cast(Any, query), values_matrix)

    async def persist(
        self,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> Self:
        db_dict = self.to_db_dict()
        # Remove id for auto-generation
        db_dict.pop("id", None)

        # Fresh timestamps if present
        now = datetime.now()
        if "createdat" in db_dict:
            db_dict["createdat"] = now
        if "updatedat" in db_dict:
            db_dict["updatedat"] = now

        columns = ", ".join(db_dict.keys())
        placeholders = ", ".join(["%s"] * len(db_dict))
        seq = f"{self.__class__.TABLE_NAME}_seq"
        query = (
            f"INSERT INTO {self.__class__.TABLE_NAME} (id, {columns}) "
            f"VALUES (nextval('{seq}'), {placeholders}) RETURNING id"
        )

        self.id = await self.__class__.db_manager().execute_and_return_id(
            query, params=tuple(db_dict.values()), cursor=cursor
        )
        return self

    def sync_persist(
        self,
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> Self:
        """Synchronous version of persist()"""
        db_dict = self.to_db_dict()
        # Remove id for auto-generation
        db_dict.pop("id", None)

        # Fresh timestamps if present
        now = datetime.now()
        if "createdat" in db_dict:
            db_dict["createdat"] = now
        if "updatedat" in db_dict:
            db_dict["updatedat"] = now

        columns = ", ".join(db_dict.keys())
        placeholders = ", ".join(["%s"] * len(db_dict))
        seq = f"{self.__class__.TABLE_NAME}_seq"
        query = (
            f"INSERT INTO {self.__class__.TABLE_NAME} (id, {columns}) "
            f"VALUES (nextval('{seq}'), {placeholders}) RETURNING id"
        )

        self.id = self.__class__.db_manager().sync_execute_and_return_id(
            query, params=tuple(db_dict.values()), cursor=cursor
        )
        return self

    async def persist_with_id(
        self,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> Self:
        db_dict = self.to_db_dict()
        columns = ", ".join(db_dict.keys())
        placeholders = ", ".join(["%s"] * len(db_dict))
        query = f"INSERT INTO {self.__class__.TABLE_NAME} ({columns}) VALUES ({placeholders})"
        params = tuple(db_dict.values())

        if cursor is not None:
            await cursor.execute(cast(Any, query), params)
        else:
            async with self.__class__.db_manager().get_cursor() as cur:
                await cur.execute(cast(Any, query), params)
        return self

    def sync_persist_with_id(
        self,
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> Self:
        """Synchronous version of persist_with_id()"""
        db_dict = self.to_db_dict()
        columns = ", ".join(db_dict.keys())
        placeholders = ", ".join(["%s"] * len(db_dict))
        query = f"INSERT INTO {self.__class__.TABLE_NAME} ({columns}) VALUES ({placeholders})"
        params = tuple(db_dict.values())

        if cursor is not None:
            cursor.execute(cast(Any, query), params)
        else:
            with self.__class__.db_manager().sync_get_cursor() as cur:
                cur.execute(cast(Any, query), params)
        return self


class JoinTableDBModel(DBModel):
    """
    Base for join/composite key tables.
    Subclasses must set PRIMARY_KEY_DB_NAMES to a list of DB column names that compose the PK.
    """

    PRIMARY_KEY_DB_NAMES: List[str] = []

    @classmethod
    def _get_pk_python_names(cls) -> List[str]:
        pk_python_names: List[str] = []
        for db_name in cls.PRIMARY_KEY_DB_NAMES:
            found = False
            for _, col_obj in cls.COLUMNS.items():
                if col_obj.db_name == db_name:
                    pk_python_names.append(col_obj.python_name)
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Primary key DB name '{db_name}' not found in COLUMNS for {cls.__name__}"
                )
        return pk_python_names

    @classmethod
    async def get_by_pks(
        cls: Type[JT],
        *pk_values: Any,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> Optional[JT]:
        if not cls.PRIMARY_KEY_DB_NAMES:
            raise NotImplementedError(
                f"{cls.__name__} must define PRIMARY_KEY_DB_NAMES."
            )
        if len(pk_values) != len(cls.PRIMARY_KEY_DB_NAMES):
            raise ValueError(
                f"Expected {len(cls.PRIMARY_KEY_DB_NAMES)} primary key values, got {len(pk_values)} for {cls.__name__}."
            )

        where_clauses = [f"{db_name} = %s" for db_name in cls.PRIMARY_KEY_DB_NAMES]
        query = f"SELECT * FROM {cls.TABLE_NAME} WHERE {' AND '.join(where_clauses)}"

        row = await cls.db_manager().execute_and_fetch_one(
            query, params=pk_values, cursor=cursor
        )
        return None if row is None else cast(JT, cls.from_db_row(row))

    @classmethod
    def sync_get_by_pks(
        cls: Type[JT],
        *pk_values: Any,
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> Optional[JT]:
        """Synchronous version of get_by_pks()"""
        if not cls.PRIMARY_KEY_DB_NAMES:
            raise NotImplementedError(
                f"{cls.__name__} must define PRIMARY_KEY_DB_NAMES."
            )
        if len(pk_values) != len(cls.PRIMARY_KEY_DB_NAMES):
            raise ValueError(
                f"Expected {len(cls.PRIMARY_KEY_DB_NAMES)} primary key values, got {len(pk_values)} for {cls.__name__}."
            )

        where_clauses = [f"{db_name} = %s" for db_name in cls.PRIMARY_KEY_DB_NAMES]
        query = f"SELECT * FROM {cls.TABLE_NAME} WHERE {' AND '.join(where_clauses)}"

        row = cls.db_manager().sync_execute_and_fetch_one(
            query, params=pk_values, cursor=cursor
        )
        return None if row is None else cast(JT, cls.from_db_row(row))

    async def persist(
        self: Self,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> Self:
        if not self.PRIMARY_KEY_DB_NAMES:  # type: ignore[attr-defined]
            raise NotImplementedError(
                f"{self.__class__.__name__} must define PRIMARY_KEY_DB_NAMES."
            )

        db_dict = self.to_db_dict()

        # Ensure all PKs are present in the instance for ON CONFLICT
        for pk_db_name in self.PRIMARY_KEY_DB_NAMES:  # type: ignore[attr-defined]
            if pk_db_name not in db_dict or db_dict[pk_db_name] is None:
                python_pk_name = next(
                    (
                        col.python_name
                        for col in self.COLUMNS.values()
                        if col.db_name == pk_db_name
                    ),
                    None,
                )
                if not (
                    python_pk_name
                    and hasattr(self, python_pk_name)
                    and getattr(self, python_pk_name) is not None
                ):
                    raise ValueError(
                        f"Primary key '{pk_db_name}' must be set on the instance before persisting {self.__class__.__name__}."
                    )

        columns = ", ".join(db_dict.keys())
        placeholders = ", ".join(["%s"] * len(db_dict))
        conflict_target = ", ".join(self.PRIMARY_KEY_DB_NAMES)  # type: ignore[attr-defined]

        query = (
            f"INSERT INTO {self.__class__.TABLE_NAME} ({columns}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_target}) DO NOTHING"
        )

        params = tuple(db_dict.values())
        if cursor is not None:
            await cursor.execute(cast(Any, query), params)
        else:
            async with self.__class__.db_manager().get_cursor() as cur:
                await cur.execute(cast(Any, query), params)
        return self

    def sync_persist(
        self: Self,
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> Self:
        """Synchronous version of persist()"""
        if not self.PRIMARY_KEY_DB_NAMES:  # type: ignore[attr-defined]
            raise NotImplementedError(
                f"{self.__class__.__name__} must define PRIMARY_KEY_DB_NAMES."
            )

        db_dict = self.to_db_dict()

        # Ensure all PKs are present in the instance for ON CONFLICT
        for pk_db_name in self.PRIMARY_KEY_DB_NAMES:  # type: ignore[attr-defined]
            if pk_db_name not in db_dict or db_dict[pk_db_name] is None:
                python_pk_name = next(
                    (
                        col.python_name
                        for col in self.COLUMNS.values()
                        if col.db_name == pk_db_name
                    ),
                    None,
                )
                if not (
                    python_pk_name
                    and hasattr(self, python_pk_name)
                    and getattr(self, python_pk_name) is not None
                ):
                    raise ValueError(
                        f"Primary key '{pk_db_name}' must be set on the instance before persisting {self.__class__.__name__}."
                    )

        columns = ", ".join(db_dict.keys())
        placeholders = ", ".join(["%s"] * len(db_dict))
        conflict_target = ", ".join(self.PRIMARY_KEY_DB_NAMES)  # type: ignore[attr-defined]

        query = (
            f"INSERT INTO {self.__class__.TABLE_NAME} ({columns}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_target}) DO NOTHING"
        )

        params = tuple(db_dict.values())
        if cursor is not None:
            cursor.execute(cast(Any, query), params)
        else:
            with self.__class__.db_manager().sync_get_cursor() as cur:
                cur.execute(cast(Any, query), params)
        return self

    @classmethod
    async def delete_by_pks(
        cls: Type[JT],
        *pk_values: Any,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> None:
        if not cls.PRIMARY_KEY_DB_NAMES:
            raise NotImplementedError(
                f"{cls.__name__} must define PRIMARY_KEY_DB_NAMES."
            )
        if len(pk_values) != len(cls.PRIMARY_KEY_DB_NAMES):
            raise ValueError(
                f"Expected {len(cls.PRIMARY_KEY_DB_NAMES)} primary key values for deletion, got {len(pk_values)} for {cls.__name__}."
            )

        where_clauses = [f"{db_name} = %s" for db_name in cls.PRIMARY_KEY_DB_NAMES]
        query = f"DELETE FROM {cls.TABLE_NAME} WHERE {' AND '.join(where_clauses)}"

        if cursor is not None:
            await cursor.execute(cast(Any, query), pk_values)
        else:
            async with cls.db_manager().get_cursor() as cur:
                await cur.execute(cast(Any, query), pk_values)

    @classmethod
    def sync_delete_by_pks(
        cls: Type[JT],
        *pk_values: Any,
        cursor: Optional[psycopg.Cursor[DictRow]] = None,
    ) -> None:
        """Synchronous version of delete_by_pks()"""
        if not cls.PRIMARY_KEY_DB_NAMES:
            raise NotImplementedError(
                f"{cls.__name__} must define PRIMARY_KEY_DB_NAMES."
            )
        if len(pk_values) != len(cls.PRIMARY_KEY_DB_NAMES):
            raise ValueError(
                f"Expected {len(cls.PRIMARY_KEY_DB_NAMES)} primary key values for deletion, got {len(pk_values)} for {cls.__name__}."
            )

        where_clauses = [f"{db_name} = %s" for db_name in cls.PRIMARY_KEY_DB_NAMES]
        query = f"DELETE FROM {cls.TABLE_NAME} WHERE {' AND '.join(where_clauses)}"

        if cursor is not None:
            cursor.execute(cast(Any, query), pk_values)
        else:
            with cls.db_manager().sync_get_cursor() as cur:
                cur.execute(cast(Any, query), pk_values)

    # Explicitly disallow single-ID ops on composite-key models
    @classmethod
    def get(cls: Type[JT], id: Any, cursor=None) -> Optional[JT]:  # type: ignore[override]
        raise NotImplementedError(
            f"Use get_by_pks for {cls.__name__} as it has a composite primary key."
        )

    @classmethod
    def delete(cls: Type[JT], id: int, cursor=None) -> None:  # type: ignore[override]
        raise NotImplementedError(
            f"Use delete_by_pks for {cls.__name__} as it has a composite primary key."
        )

    def update(self, cursor=None) -> None:  # type: ignore[override]
        raise NotImplementedError(
            f"Updates are not typically performed on join table rows like {self.__class__.__name__}."
        )
