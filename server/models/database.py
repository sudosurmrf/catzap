import json
import uuid
from datetime import datetime

import asyncpg

from server.config import settings

_pool: asyncpg.Pool | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS zones (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    polygon JSONB NOT NULL,
    overlap_threshold REAL NOT NULL DEFAULT 0.3,
    cooldown_seconds INTEGER NOT NULL DEFAULT 3,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cats (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    model_version INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY,
    type TEXT NOT NULL,
    cat_id UUID,
    cat_name TEXT,
    zone_id UUID,
    zone_name TEXT,
    confidence REAL,
    overlap REAL,
    servo_pan REAL,
    servo_tilt REAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL
);
"""


async def init_db(database_url: str | None = None):
    global _pool
    url = database_url or settings.database_url
    _pool = await asyncpg.create_pool(url)
    async with _pool.acquire() as conn:
        await conn.execute(SCHEMA)


async def close_db():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _pool


async def create_zone(
    name: str,
    polygon: list[list[float]],
    overlap_threshold: float = 0.3,
    cooldown_seconds: int = 3,
    conn: asyncpg.Connection | None = None,
) -> str:
    zone_id = uuid.uuid4()
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        await c.execute(
            "INSERT INTO zones (id, name, polygon, overlap_threshold, cooldown_seconds) VALUES ($1, $2, $3, $4, $5)",
            zone_id, name, json.dumps(polygon), overlap_threshold, cooldown_seconds,
        )
        return str(zone_id)
    finally:
        if conn is None:
            await pool.release(c)


async def get_zones(conn: asyncpg.Connection | None = None) -> list[dict]:
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        rows = await c.fetch("SELECT * FROM zones ORDER BY created_at DESC")
        return [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "polygon": json.loads(row["polygon"]) if isinstance(row["polygon"], str) else row["polygon"],
                "overlap_threshold": row["overlap_threshold"],
                "cooldown_seconds": row["cooldown_seconds"],
                "enabled": row["enabled"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]
    finally:
        if conn is None:
            await pool.release(c)


async def update_zone(zone_id: str, conn: asyncpg.Connection | None = None, **kwargs) -> bool:
    allowed = {"name", "polygon", "overlap_threshold", "cooldown_seconds", "enabled"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False
    if "polygon" in updates:
        updates["polygon"] = json.dumps(updates["polygon"])

    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        set_parts = []
        values = []
        for i, (k, v) in enumerate(updates.items(), 1):
            set_parts.append(f"{k} = ${i}")
            values.append(v)
        values.append(uuid.UUID(zone_id))
        query = f"UPDATE zones SET {', '.join(set_parts)} WHERE id = ${len(values)}"
        result = await c.execute(query, *values)
        return result != "UPDATE 0"
    finally:
        if conn is None:
            await pool.release(c)


async def delete_zone(zone_id: str, conn: asyncpg.Connection | None = None) -> bool:
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        result = await c.execute("DELETE FROM zones WHERE id = $1", uuid.UUID(zone_id))
        return result != "DELETE 0"
    finally:
        if conn is None:
            await pool.release(c)


async def create_cat(name: str, conn: asyncpg.Connection | None = None) -> str:
    cat_id = uuid.uuid4()
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        await c.execute(
            "INSERT INTO cats (id, name) VALUES ($1, $2)",
            cat_id, name,
        )
        return str(cat_id)
    finally:
        if conn is None:
            await pool.release(c)


async def get_cats(conn: asyncpg.Connection | None = None) -> list[dict]:
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        rows = await c.fetch("SELECT * FROM cats ORDER BY name")
        return [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "model_version": row["model_version"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]
    finally:
        if conn is None:
            await pool.release(c)


async def delete_cat(cat_id: str, conn: asyncpg.Connection | None = None) -> bool:
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        result = await c.execute("DELETE FROM cats WHERE id = $1", uuid.UUID(cat_id))
        return result != "DELETE 0"
    finally:
        if conn is None:
            await pool.release(c)


async def create_event(
    event_type: str,
    cat_id: str | None = None,
    cat_name: str | None = None,
    zone_id: str | None = None,
    zone_name: str | None = None,
    confidence: float | None = None,
    overlap: float | None = None,
    servo_pan: float | None = None,
    servo_tilt: float | None = None,
    conn: asyncpg.Connection | None = None,
) -> str:
    event_id = uuid.uuid4()
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        await c.execute(
            """INSERT INTO events (id, type, cat_id, cat_name, zone_id, zone_name,
               confidence, overlap, servo_pan, servo_tilt)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
            event_id, event_type,
            uuid.UUID(cat_id) if cat_id else None, cat_name,
            uuid.UUID(zone_id) if zone_id else None, zone_name,
            confidence, overlap, servo_pan, servo_tilt,
        )
        return str(event_id)
    finally:
        if conn is None:
            await pool.release(c)


async def get_events(
    event_type: str | None = None,
    cat_name: str | None = None,
    zone_name: str | None = None,
    limit: int = 100,
    offset: int = 0,
    conn: asyncpg.Connection | None = None,
) -> list[dict]:
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        query = "SELECT * FROM events WHERE TRUE"
        params: list = []
        param_idx = 1

        if event_type:
            query += f" AND type = ${param_idx}"
            params.append(event_type)
            param_idx += 1
        if cat_name:
            query += f" AND cat_name = ${param_idx}"
            params.append(cat_name)
            param_idx += 1
        if zone_name:
            query += f" AND zone_name = ${param_idx}"
            params.append(zone_name)
            param_idx += 1

        query += f" ORDER BY timestamp DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([limit, offset])

        rows = await c.fetch(query, *params)
        return [
            {
                "id": str(row["id"]),
                "type": row["type"],
                "cat_id": str(row["cat_id"]) if row["cat_id"] else None,
                "cat_name": row["cat_name"],
                "zone_id": str(row["zone_id"]) if row["zone_id"] else None,
                "zone_name": row["zone_name"],
                "confidence": row["confidence"],
                "overlap": row["overlap"],
                "servo_pan": row["servo_pan"],
                "servo_tilt": row["servo_tilt"],
                "timestamp": row["timestamp"].isoformat(),
            }
            for row in rows
        ]
    finally:
        if conn is None:
            await pool.release(c)


async def get_setting(key: str, conn: asyncpg.Connection | None = None):
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        row = await c.fetchrow("SELECT value FROM settings WHERE key = $1", key)
        if row:
            val = row["value"]
            return json.loads(val) if isinstance(val, str) else val
        return None
    finally:
        if conn is None:
            await pool.release(c)


async def set_setting(key: str, value, conn: asyncpg.Connection | None = None) -> None:
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        await c.execute(
            "INSERT INTO settings (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value = $2",
            key, json.dumps(value),
        )
    finally:
        if conn is None:
            await pool.release(c)
