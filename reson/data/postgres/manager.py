# reson/data/postgres/manager.py  (async psycopg3 + DictRow typing + ContextVar reuse)

from __future__ import annotations

import os
import importlib
from threading import Lock
from typing import Any, Optional, Sequence, Mapping, AsyncIterator, cast
from contextvars import ContextVar
from contextlib import asynccontextmanager

import psycopg
from psycopg.rows import dict_row, DictRow
from psycopg_pool import AsyncConnectionPool


class DatabaseManager:
    """
    Async psycopg3 manager with:
      - one-instance-per-DSN (same behavior as your sync version)
      - ContextVar-based connection reuse across nested calls
      - DictRow typing so fetches are Mapping[str, Any]
      - correct commit/rollback at the outermost DB call
    """

    _instances: dict[str, dict] = {}
    _lock = Lock()
    _otel_instrumented = False  # instrument once per process

    def __new__(cls, dsn: str):
        with cls._lock:
            if dsn not in cls._instances:
                instance = super(DatabaseManager, cls).__new__(cls)
                cls._instances[dsn] = {"instance": instance, "initialized": False}
        return cls._instances[dsn]["instance"]

    def __init__(self, dsn: str):
        store = self.__class__._instances[dsn]
        if not store["initialized"]:
            self.initialize(dsn)

    def initialize(self, dsn: str = "") -> None:
        # Optional OpenTelemetry for psycopg v3 (async supported).
        # Enable by exporting: OTEL_PY_PG_AUTO_INSTRUMENT=1
        if (
            os.getenv("OTEL_PY_PG_AUTO_INSTRUMENT", "0") == "1"
            and not self.__class__._otel_instrumented
        ):
            try:
                mod = importlib.import_module("opentelemetry.instrumentation.psycopg")
                instr = getattr(mod, "PsycopgInstrumentor", None)
                if instr is not None:
                    instr().instrument()
                    self.__class__._otel_instrumented = True
            except Exception:
                # OTel is optional; ignore if not installed.
                pass

        # ✅ Correct: pool is generic over the CONNECTION TYPE, not the row type
        self.pool: AsyncConnectionPool[psycopg.AsyncConnection[DictRow]] = (
            AsyncConnectionPool(
                dsn,
                min_size=1,
                max_size=20,
                timeout=30.0,
                open=True,
            )
        )

        # Per-instance ContextVars so multiple DSNs don't clash
        self._current_conn: ContextVar[Optional[psycopg.AsyncConnection[DictRow]]] = (
            ContextVar(
                f"pg_current_conn_{id(self)}",
                default=None,
            )
        )
        self._conn_depth: ContextVar[int] = ContextVar(
            f"pg_conn_depth_{id(self)}",
            default=0,
        )

        self.__class__._instances[dsn]["initialized"] = True

    @asynccontextmanager
    async def get_connection(self) -> AsyncIterator[psycopg.AsyncConnection[DictRow]]:
        """
        Reuse one connection per task to prevent re-entrant pool.getconn() deadlocks.
        """
        existing = self._current_conn.get()
        if existing is not None:
            token_depth = self._conn_depth.set(self._conn_depth.get() + 1)
            try:
                yield existing
            finally:
                self._conn_depth.reset(token_depth)
            return

        conn: psycopg.AsyncConnection[DictRow] = await self.pool.getconn()
        # Return dict-like rows (Mapping[str, Any])
        conn.row_factory = dict_row

        token_conn = self._current_conn.set(conn)
        token_depth = self._conn_depth.set(1)
        try:
            yield conn
        finally:
            try:
                await self.pool.putconn(conn)
            finally:
                self._current_conn.reset(token_conn)
                self._conn_depth.reset(token_depth)

    @asynccontextmanager
    async def get_cursor(
        self,
        *,
        commit: bool = True,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> AsyncIterator[psycopg.AsyncCursor[DictRow]]:
        """
        Yield an async cursor.

        - If 'cursor' is provided, we do not open/commit/rollback/close anything.
        - If we open it:
            - commit on success only at the outermost depth
            - rollback on exception only at the outermost depth
        """
        if cursor is not None:
            yield cursor
            return

        async with self.get_connection() as conn:
            async with conn.cursor() as cur:
                try:
                    yield cur
                    if commit and self._conn_depth.get() == 1:
                        await conn.commit()
                except Exception:
                    if self._conn_depth.get() == 1:
                        await conn.rollback()
                    raise

    async def execute_query(
        self,
        query: Any,  # accept str/SQL/Composed/etc. (satisfies Pylance)
        params: Optional[Sequence[Any]] = None,
        *,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> Optional[Sequence[Mapping[str, Any]]]:
        """
        SELECT / RETURNING -> Sequence of mapping rows
        DML without RETURNING -> None
        """
        if cursor is not None:
            await cursor.execute(cast(Any, query), params)
            return await cursor.fetchall() if cursor.description is not None else None

        async with self.get_cursor() as cur:
            await cur.execute(cast(Any, query), params)
            return await cur.fetchall() if cur.description is not None else None

    async def execute_and_fetch_one(
        self,
        query: Any,
        params: Optional[Sequence[Any]] = None,
        *,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ) -> Optional[Mapping[str, Any]]:
        if cursor is not None:
            await cursor.execute(cast(Any, query), params)
            return await cursor.fetchone()
        async with self.get_cursor() as cur:
            await cur.execute(cast(Any, query), params)
            return await cur.fetchone()

    async def execute_and_return_id(
        self,
        query: Any,
        params: Optional[Sequence[Any]] = None,
        *,
        cursor: Optional[psycopg.AsyncCursor[DictRow]] = None,
    ):
        """
        Expects the SQL to include `RETURNING id`.
        """
        if cursor is not None:
            await cursor.execute(cast(Any, query), params)
            row = await cursor.fetchone()
        else:
            async with self.get_cursor() as cur:
                await cur.execute(cast(Any, query), params)
                row = await cur.fetchone()

        return row["id"] if row is not None else None

    async def close(self) -> None:
        await self.pool.close()
