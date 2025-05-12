# reson/stores.py  –– simple re-exports & an in-memory fallback
from asimov.caches.redis_cache import RedisCache as RedisStore  # new alias
from typing import Dict, Any, Optional, Set
from asimov.caches.cache import Cache as Store
from asimov.data.postgres.manager import DatabaseManager
import jsonpickle
from contextlib import asynccontextmanager

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class StoreKind(str, Enum):
    memory = "memory"
    redis = "redis"
    postgres = "postgres"


class StoreConfigBase(BaseModel):
    kind: StoreKind
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MemoryStoreConfig(StoreConfigBase):
    kind: StoreKind = Field(default=StoreKind.memory)


class RedisStoreConfig(StoreConfigBase):
    kind: StoreKind = Field(default=StoreKind.redis)
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=0, le=65535)
    db: int = Field(default=0, ge=0)
    password: str | None = None


class PostgresStoreConfig(StoreConfigBase):
    kind: StoreKind = Field(default=StoreKind.postgres)
    dsn: str = Field(default="postgresql://postgres:postgres@localhost:5432/postgres")
    table: str = Field()


_MEM: Dict[str, Any] = {}


class MemoryStore(Store):
    async def get(self, key, default=None, raw=False):
        return _MEM.get(key, default)
    async def set(self, key, value, raw=False):
        _MEM[key] = value
    async def delete(self, key):           
        _MEM.pop(key, None)
    async def clear(self):                 
        _MEM.clear()
    async def get_all(self):               
        return dict(_MEM)
    async def publish_to_mailbox(self, _, __): 
        pass
    async def get_message(self, __, timeout=None): 
        return None
    async def keys(self):                  
        return set(_MEM.keys())
    async def close(self):                 
        pass


class PostgresStore(Store):
    """Postgres-backed store implementation using asimov's DatabaseManager."""
    dsn: str
    table: str
    _db = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._db = DatabaseManager(self.dsn)
        self._ensure_table()
    
    def _ensure_table(self):
        """Ensure the cache table exists."""
        self._db.execute_query(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
    
    async def get(self, key: str, default=None, raw: bool = False):
        modified_key = key
        if not raw:
            modified_key = await self.apply_key_modifications(key)

        result = self._db.execute_and_fetch_one(
            f"SELECT value FROM {self.table} WHERE key = %s",
            (modified_key,)
        )
        
        if result:
            return jsonpickle.decode(result['value'])
        return default
    
    async def set(self, key: str, value, raw: bool = False):
        modified_key = key
        if not raw:
            modified_key = await self.apply_key_modifications(key)
            
        encoded_value = jsonpickle.encode(value)
        self._db.execute_query(
            f"""
            INSERT INTO {self.table} (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = %s
            """,
            (modified_key, encoded_value, encoded_value)
        )
    
    async def delete(self, key: str):
        modified_key = await self.apply_key_modifications(key)
        self._db.execute_query(
            f"DELETE FROM {self.table} WHERE key = %s",
            (modified_key,)
        )
    
    async def clear(self):
        prefix = await self.get_prefix()
        if prefix:
            self._db.execute_query(
                f"DELETE FROM {self.table} WHERE key LIKE %s",
                (f"{prefix}{self.affix_sep}%",)
            )
        else:
            self._db.execute_query(f"DELETE FROM {self.table}")
    
    async def get_all(self) -> Dict[str, Any]:
        prefix = await self.get_prefix()
        if prefix:
            rows = self._db.execute_query(
                f"SELECT key, value FROM {self.table} WHERE key LIKE %s",
                (f"{prefix}{self.affix_sep}%",)
            )
        else:
            rows = self._db.execute_query(f"SELECT key, value FROM {self.table}")
            
        result = {}
        if rows:
            for row in rows:
                result[row['key']] = jsonpickle.decode(row['value'])
        return result
    
    async def publish_to_mailbox(self, mailbox_id: str, value):
        # Basic implementation - in production would use LISTEN/NOTIFY
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        await self.set(f"mailbox:{modified_mailbox_id}", value)
    
    async def get_message(self, mailbox_id: str, timeout: Optional[float] = None):
        # Basic implementation - in production would use LISTEN/NOTIFY
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        key = f"mailbox:{modified_mailbox_id}"
        value = await self.get(key)
        await self.delete(key)
        return value
    
    async def keys(self) -> Set[str]:
        prefix = await self.get_prefix()
        suffix = await self.get_suffix()
        
        query = f"SELECT key FROM {self.table}"
        params = []
        
        conditions = []
        if prefix:
            conditions.append("key LIKE %s")
            params.append(f"{prefix}{self.affix_sep}%")
        if suffix:
            conditions.append("key LIKE %s")
            params.append(f"%{self.affix_sep}{suffix}")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        rows = self._db.execute_query(query, params or None)
        return {row['key'] for row in (rows or [])}
    
    async def close(self):
        # Connection pool is managed by DatabaseManager singleton
        pass
