# reson/stores.py  –– simple re-exports & an in-memory fallback
from reson.caches.redis_cache import RedisCache as RedisStore  # new alias
from typing import Dict, Any, Optional, Set
from reson.caches.cache import Cache as Store
from reson.data.postgres.manager import DatabaseManager
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
    table: str = Field()  # Table name
    column: str = Field()  # JSONB column name that contains all data


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
    """PostgreSQL-backed store using a JSONB column as the document storage."""
    dsn: str
    table: str
    column: str  # JSONB column name
    _db = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._db = DatabaseManager(self.dsn)
        self._ensure_table()
    
    def _ensure_table(self):
        """Ensure the table exists with a single JSONB column."""
        self._db.execute_query(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id SERIAL PRIMARY KEY,
                {self.column} JSONB NOT NULL DEFAULT '{{}}'::jsonb
            )
        """)
        
        # Make sure we have at least one row
        self._db.execute_query(f"""
            INSERT INTO {self.table} ({self.column})
            SELECT '{{}}'::jsonb
            WHERE NOT EXISTS (SELECT 1 FROM {self.table} LIMIT 1)
        """)
    
    async def get(self, key: str, default=None, raw: bool = False):
        modified_key = key if raw else await self.apply_key_modifications(key)

        # Extract the value from the JSONB using -> operator
        result = self._db.execute_and_fetch_one(
            f"SELECT {self.column}->%s AS value FROM {self.table} LIMIT 1",
            (modified_key,)
        )
        
        if result and result['value'] is not None:
            return result['value'] 
        return default
    
    async def set(self, key: str, value, raw: bool = False):
        modified_key = key if raw else await self.apply_key_modifications(key)
        
        import json
        # Use jsonb_set to update or insert the key in the JSONB
        self._db.execute_query(
            f"""
            UPDATE {self.table} 
            SET {self.column} = jsonb_set({self.column}, %s, %s, true)
            """,
            ([modified_key], json.dumps(value))  # Path must be an array
        )
    
    async def delete(self, key: str):
        modified_key = await self.apply_key_modifications(key)
        
        # Remove the key from JSONB using - operator
        self._db.execute_query(
            f"UPDATE {self.table} SET {self.column} = {self.column} - %s",
            (modified_key,)
        )
    
    async def clear(self):
        prefix = await self.get_prefix()
        if prefix:
            # For prefix matching in JSONB, we need to iterate through keys
            # First get all the data
            result = self._db.execute_and_fetch_one(
                f"SELECT {self.column} FROM {self.table} LIMIT 1"
            )
            
            if result and result[self.column]:
                # Get all keys from the JSONB
                keys_result = self._db.execute_query(
                    f"SELECT jsonb_object_keys({self.column}) AS key FROM {self.table} LIMIT 1"
                )
                
                if keys_result:
                    prefix_with_sep = f"{prefix}{self.affix_sep}"
                    # Filter keys with the prefix
                    for key_row in keys_result:
                        key = key_row['key']
                        if key.startswith(prefix_with_sep):
                            # Remove this key
                            self._db.execute_query(
                                f"UPDATE {self.table} SET {self.column} = {self.column} - %s",
                                (key,)
                            )
        else:
            # Reset to empty object
            self._db.execute_query(f"UPDATE {self.table} SET {self.column} = '{{}}'::jsonb")
    
    async def get_all(self) -> Dict[str, Any]:
        # Extract all keys and values from the JSONB
        result = self._db.execute_and_fetch_one(
            f"SELECT {self.column} FROM {self.table} LIMIT 1"
        )
        
        if result and result[self.column]:
            return result[self.column]
        return {}
    
    async def publish_to_mailbox(self, mailbox_id: str, value):
        # Use the same underlying storage
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        await self.set(f"mailbox:{modified_mailbox_id}", value)
    
    async def get_message(self, mailbox_id: str, timeout: Optional[float] = None):
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        key = f"mailbox:{modified_mailbox_id}"
        value = await self.get(key)
        await self.delete(key)
        return value
    
    async def keys(self) -> Set[str]:
        # Extract all keys from the JSONB
        keys_result = self._db.execute_query(
            f"SELECT jsonb_object_keys({self.column}) AS key FROM {self.table} LIMIT 1"
        )
        
        if not keys_result:
            return set()
            
        # Filter by prefix/suffix if needed
        prefix = await self.get_prefix()
        suffix = await self.get_suffix()
        
        keys = set()
        for row in keys_result:
            key = row['key']
            include = True
            
            if prefix and not key.startswith(f"{prefix}{self.affix_sep}"):
                include = False
            if suffix and not key.endswith(f"{self.affix_sep}{suffix}"):
                include = False
                
            if include:
                keys.add(key)
                
        return keys
    
    async def close(self):
        # Connection pool is managed by DatabaseManager singleton
        pass
