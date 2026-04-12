import pytest
from server.models.database import (
    init_db, close_db, create_zone, get_zones, update_zone, delete_zone,
    create_cat, get_cats, delete_cat, create_event, get_events,
    get_setting, set_setting,
)

# These tests require a running PostgreSQL instance with a test database.
# Run: createdb catzap_test
# Set env: CATZAP_DATABASE_URL=postgresql://localhost:5432/catzap_test

TEST_DB_URL = "postgresql://localhost:5432/catzap_test"


@pytest.fixture(autouse=True)
async def setup_db():
    await init_db(TEST_DB_URL)
    from server.models.database import get_pool
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM events")
        await conn.execute("DELETE FROM zones")
        await conn.execute("DELETE FROM cats")
        await conn.execute("DELETE FROM settings")
    yield
    await close_db()


@pytest.mark.asyncio
async def test_create_and_get_zone():
    zone_id = await create_zone(
        name="Kitchen Counter",
        polygon=[[0.1, 0.2], [0.5, 0.2], [0.5, 0.8], [0.1, 0.8]],
        overlap_threshold=0.3,
    )
    zones = await get_zones()
    assert len(zones) == 1
    assert zones[0]["name"] == "Kitchen Counter"
    assert zones[0]["id"] == zone_id
    assert zones[0]["enabled"] == True
    assert zones[0]["overlap_threshold"] == 0.3
    assert "polygon" in zones[0]
    assert "created_at" in zones[0]


@pytest.mark.asyncio
async def test_update_zone():
    zone_id = await create_zone(name="Test", polygon=[[0, 0], [1, 0], [1, 1]])
    success = await update_zone(zone_id, name="Updated", enabled=False)
    assert success
    zones = await get_zones()
    assert zones[0]["name"] == "Updated"
    assert zones[0]["enabled"] == False


@pytest.mark.asyncio
async def test_delete_zone():
    zone_id = await create_zone(name="Test", polygon=[[0, 0], [1, 0], [1, 1]])
    success = await delete_zone(zone_id)
    assert success
    zones = await get_zones()
    assert len(zones) == 0


@pytest.mark.asyncio
async def test_create_and_get_cat():
    cat_id = await create_cat(name="Luna")
    cats = await get_cats()
    assert len(cats) == 1
    assert cats[0]["name"] == "Luna"
    assert cats[0]["id"] == cat_id


@pytest.mark.asyncio
async def test_delete_cat():
    cat_id = await create_cat(name="Luna")
    success = await delete_cat(cat_id)
    assert success
    cats = await get_cats()
    assert len(cats) == 0


@pytest.mark.asyncio
async def test_create_and_get_events():
    await create_event(
        event_type="ZAP", cat_name="Luna", zone_name="Kitchen Counter",
        confidence=0.92, overlap=0.65, servo_pan=45.0, servo_tilt=30.0,
    )
    events = await get_events()
    assert len(events) == 1
    assert events[0]["type"] == "ZAP"
    assert events[0]["cat_name"] == "Luna"
    assert events[0]["zone_name"] == "Kitchen Counter"


@pytest.mark.asyncio
async def test_get_events_filtered():
    await create_event(event_type="ZAP", cat_name="Luna", zone_name="Counter")
    await create_event(event_type="SYSTEM")
    events = await get_events(event_type="ZAP")
    assert len(events) == 1
    assert events[0]["type"] == "ZAP"


@pytest.mark.asyncio
async def test_settings():
    await set_setting("cooldown_default", 15)
    val = await get_setting("cooldown_default")
    assert val == 15

    missing = await get_setting("nonexistent")
    assert missing is None
