from server.spatial.projection import angle_depth_to_room, room_to_angle, project_furniture_to_angles


def test_angle_depth_to_room_center():
    camera_pos = (0.0, 0.0, 150.0)
    pos = angle_depth_to_room(pan=90.0, tilt=45.0, depth_cm=200.0, camera_pos=camera_pos)
    assert len(pos) == 3
    assert pos[2] < 150.0


def test_room_to_angle_roundtrip():
    camera_pos = (0.0, 0.0, 150.0)
    pan, tilt, depth = 75.0, 40.0, 180.0
    room_pos = angle_depth_to_room(pan, tilt, depth, camera_pos)
    pan2, tilt2 = room_to_angle(room_pos[0], room_pos[1], room_pos[2], camera_pos)
    assert abs(pan2 - pan) < 0.5
    assert abs(tilt2 - tilt) < 0.5


def test_project_furniture_to_angles():
    camera_pos = (0.0, 0.0, 150.0)
    base_polygon = [(100, 200), (200, 200), (200, 300), (100, 300)]
    angles = project_furniture_to_angles(base_polygon, 0.0, 75.0, camera_pos)
    assert len(angles) >= 4
    for pan, tilt in angles:
        assert 0 <= pan <= 180
        assert 0 <= tilt <= 90
