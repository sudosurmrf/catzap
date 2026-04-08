import json
import uuid
from datetime import datetime

import asyncpg

from server.config import settings

_pool: asyncpg.Pool | None = None

# ── In-memory zone cache (invalidated on create/update/delete) ──
_zone_cache: list[dict] | None = None


def invalidate_zone_cache():
    global _zone_cache
    _zone_cache = None
    # Also clear pre-cached Shapely polygons
    from server.vision.zone_checker import invalidate_poly_cache
    invalidate_poly_cache()

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

ALTER TABLE zones ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT '2d';
ALTER TABLE zones ADD COLUMN IF NOT EXISTS room_polygon JSONB;
ALTER TABLE zones ADD COLUMN IF NOT EXISTS height_min REAL NOT NULL DEFAULT 0.0;
ALTER TABLE zones ADD COLUMN IF NOT EXISTS height_max REAL NOT NULL DEFAULT 0.0;
ALTER TABLE zones ADD COLUMN IF NOT EXISTS furniture_id UUID;

CREATE TABLE IF NOT EXISTS cats (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    model_version INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cat_photos (
    id UUID PRIMARY KEY,
    cat_id UUID NOT NULL REFERENCES cats(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'upload',
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

CREATE TABLE IF NOT EXISTS furniture (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    base_polygon JSONB NOT NULL,
    height_min REAL NOT NULL DEFAULT 0.0,
    height_max REAL NOT NULL DEFAULT 0.0,
    depth_anchored BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
    mode: str = "2d",
    room_polygon: list[list[float]] | None = None,
    height_min: float = 0.0,
    height_max: float = 0.0,
    furniture_id: str | None = None,
    conn: asyncpg.Connection | None = None,
) -> str:
    zone_id = uuid.uuid4()
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        await c.execute(
            """INSERT INTO zones (id, name, polygon, overlap_threshold, cooldown_seconds,
               mode, room_polygon, height_min, height_max, furniture_id)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
            zone_id, name, json.dumps(polygon), overlap_threshold, cooldown_seconds,
            mode, json.dumps(room_polygon) if room_polygon is not None else None,
            height_min, height_max,
            uuid.UUID(furniture_id) if furniture_id else None,
        )
        invalidate_zone_cache()
        return str(zone_id)
    finally:
        if conn is None:
            await pool.release(c)


async def get_zones(conn: asyncpg.Connection | None = None) -> list[dict]:
    global _zone_cache
    if _zone_cache is not None and conn is None:
        return _zone_cache
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        rows = await c.fetch("SELECT * FROM zones ORDER BY created_at DESC")
        result = [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "polygon": json.loads(row["polygon"]) if isinstance(row["polygon"], str) else row["polygon"],
                "overlap_threshold": row["overlap_threshold"],
                "cooldown_seconds": row["cooldown_seconds"],
                "enabled": row["enabled"],
                "created_at": row["created_at"].isoformat(),
                "mode": row["mode"],
                "room_polygon": json.loads(row["room_polygon"]) if row["room_polygon"] and isinstance(row["room_polygon"], str) else row["room_polygon"],
                "height_min": row["height_min"],
                "height_max": row["height_max"],
                "furniture_id": str(row["furniture_id"]) if row["furniture_id"] else None,
            }
            for row in rows
        ]
        if conn is None:
            _zone_cache = result
        return result
    finally:
        if conn is None:
            await pool.release(c)


async def update_zone(zone_id: str, conn: asyncpg.Connection | None = None, **kwargs) -> bool:
    allowed = {"name", "polygon", "overlap_threshold", "cooldown_seconds", "enabled",
               "mode", "room_polygon", "height_min", "height_max", "furniture_id"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False
    if "polygon" in updates:
        updates["polygon"] = json.dumps(updates["polygon"])
    if "room_polygon" in updates:
        updates["room_polygon"] = json.dumps(updates["room_polygon"]) if updates["room_polygon"] else None

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
        invalidate_zone_cache()
        return result != "UPDATE 0"
    finally:
        if conn is None:
            await pool.release(c)


async def delete_zone(zone_id: str, conn: asyncpg.Connection | None = None) -> bool:
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        result = await c.execute("DELETE FROM zones WHERE id = $1", uuid.UUID(zone_id))
        invalidate_zone_cache()
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


async def create_furniture(
    name: str, base_polygon: list[list[float]],
    height_min: float = 0.0, height_max: float = 0.0,
    depth_anchored: bool = False, conn: asyncpg.Connection | None = None,
) -> str:
    furniture_id = uuid.uuid4()
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        await c.execute(
            """INSERT INTO furniture (id, name, base_polygon, height_min, height_max, depth_anchored)
               VALUES ($1, $2, $3, $4, $5, $6)""",
            furniture_id, name, json.dumps(base_polygon), height_min, height_max, depth_anchored,
        )
        return str(furniture_id)
    finally:
        if conn is None:
            await pool.release(c)


async def get_furniture(conn: asyncpg.Connection | None = None) -> list[dict]:
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        rows = await c.fetch("SELECT * FROM furniture ORDER BY created_at DESC")
        return [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "base_polygon": json.loads(row["base_polygon"]) if isinstance(row["base_polygon"], str) else row["base_polygon"],
                "height_min": row["height_min"],
                "height_max": row["height_max"],
                "depth_anchored": row["depth_anchored"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]
    finally:
        if conn is None:
            await pool.release(c)


async def update_furniture(furniture_id: str, conn: asyncpg.Connection | None = None, **kwargs) -> bool:
    allowed = {"name", "base_polygon", "height_min", "height_max", "depth_anchored"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False
    if "base_polygon" in updates:
        updates["base_polygon"] = json.dumps(updates["base_polygon"])

    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        set_parts = []
        values = []
        for i, (k, v) in enumerate(updates.items(), 1):
            set_parts.append(f"{k} = ${i}")
            values.append(v)
        values.append(uuid.UUID(furniture_id))
        query = f"UPDATE furniture SET {', '.join(set_parts)} WHERE id = ${len(values)}"
        result = await c.execute(query, *values)
        return result != "UPDATE 0"
    finally:
        if conn is None:
            await pool.release(c)


async def delete_furniture(furniture_id: str, conn: asyncpg.Connection | None = None) -> bool:
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        result = await c.execute("DELETE FROM furniture WHERE id = $1", uuid.UUID(furniture_id))
        return result != "DELETE 0"
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


async def create_cat_photo(
    cat_id: str,
    file_path: str,
    source: str = "upload",
    conn: asyncpg.Connection | None = None,
) -> str:
    photo_id = uuid.uuid4()
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        await c.execute(
            """INSERT INTO cat_photos (id, cat_id, file_path, source)
               VALUES ($1, $2, $3, $4)""",
            photo_id, uuid.UUID(cat_id), file_path, source,
        )
        return str(photo_id)
    finally:
        if conn is None:
            await pool.release(c)


async def get_cat_photos(cat_id: str, conn: asyncpg.Connection | None = None) -> list[dict]:
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        rows = await c.fetch(
            "SELECT * FROM cat_photos WHERE cat_id = $1 ORDER BY created_at DESC",
            uuid.UUID(cat_id),
        )
        return [
            {
                "id": str(row["id"]),
                "cat_id": str(row["cat_id"]),
                "file_path": row["file_path"],
                "source": row["source"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]
    finally:
        if conn is None:
            await pool.release(c)


async def delete_cat_photo(photo_id: str, conn: asyncpg.Connection | None = None) -> str | None:
    """Delete a photo record and return its file_path for cleanup, or None if not found."""
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        row = await c.fetchrow(
            "DELETE FROM cat_photos WHERE id = $1 RETURNING file_path",
            uuid.UUID(photo_id),
        )
        return row["file_path"] if row else None
    finally:
        if conn is None:
            await pool.release(c)


async def get_cat_photo_counts(conn: asyncpg.Connection | None = None) -> dict[str, int]:
    """Return {cat_id: photo_count} for all cats."""
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        rows = await c.fetch(
            "SELECT cat_id, COUNT(*) as count FROM cat_photos GROUP BY cat_id"
        )
        return {str(row["cat_id"]): row["count"] for row in rows}
    finally:
        if conn is None:
            await pool.release(c)
