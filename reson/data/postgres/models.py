from __future__ import annotations

from datetime import datetime
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
)
import os
import threading

import psycopg
from psycopg.rows import DictRow
from psycopg.types.json import Json, Jsonb

from reson.data.postgres.manager import DatabaseManager

T = TypeVar("T", bound="DBModel")
JT = TypeVar("JT", bound="JoinTableDBModel")
R = TypeVar("R")


class LazyAttribute(Generic[R]):
    def __init__(self, func: Callable[[Any], R]):
        self.func = func
        self.lock = threading.Lock()

    def __get__(self, instance: Any, cls: Any) -> R:
        if instance is None:
            return self  # type: ignore[return-value]

        attr_name = f"_{self.func.__name__}"
        if not hasattr(instance, attr_name):
            with self.lock:
                if not hasattr(instance, attr_name):
                    value = self.func(instance)
                    setattr(instance, attr_name, value)
        return cast(R, getattr(instance, attr_name))

    def invalidate(self, instance: Any) -> None:
        attr_name = f"_{self.func.__name__}"
        if hasattr(instance, attr_name):
            delattr(instance, attr_name)


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
    ) -> Optional[T]:
        query = f"SELECT * FROM {cls.TABLE_NAME} WHERE id = %s"
        row = await cls.db_manager().execute_and_fetch_one(
            query, params=(id,), cursor=cursor
        )
        return None if row is None else cls.from_db_row(row)

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

    # ---------- serialization helpers ----------

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

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            col.db_name: getattr(self, col.python_name)
            for col in self.COLUMNS.values()
            if hasattr(self, col.python_name)
        }

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
