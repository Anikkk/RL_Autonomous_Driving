import numpy as np
from metadrive.utils import clip


def custom_reward_function(env, vehicle_id: str):
    vehicle = env.agents[vehicle_id]
    step_info = dict()
    reward = 0.0

    # Basic reward predefined
    if vehicle.lane in vehicle.navigation.current_ref_lanes:
        current_lane = vehicle.lane
        positive_road = 1
    else:
        current_lane = vehicle.navigation.current_ref_lanes[0]
        current_road = vehicle.navigation.current_road
        positive_road = 1 if not current_road.is_negative_road() else -1

    long_last, _ = current_lane.local_coordinates(vehicle.last_position)
    long_now, lateral_now = current_lane.local_coordinates(vehicle.position)
    lane_width = vehicle.navigation.get_current_lane_width()

    lateral_factor = clip(1 - 2 * abs(lateral_now) / lane_width, 0.0, 1.0) if env.config["use_lateral_reward"] else 1.0

    reward += env.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
    reward += env.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

    # Extra reward for maintaining in the center of lane for stablity
    if abs(lateral_now) < 0.5:
        reward += 0.3

    # Penalty for high speed (Required in DQN for discrete action space)
    if vehicle.speed_km_h > 20:
        reward -= 1.0

    # Curve reward and penalty
    try:
        road_type = getattr(vehicle.navigation.current_road, "road_type", "").lower()
        is_intersection = any(key in road_type for key in ["intersection", "roundabout"])
    except Exception:
        is_intersection = False

    try:
        curvature = abs(current_lane.curvature_at(long_now))
        if curvature > 0.02:
            if vehicle.speed_km_h > 20:
                reward -= 0.5
            else:
                reward += 0.2
    except:
        curvature = 0.0

    if is_intersection:
        if vehicle.speed_km_h > 15:
            reward -= 1.0
        try:
            future_pos = vehicle.navigation.current_ref_lanes[0].position(long_now + 2.0, 0)
            expected_heading = current_lane.heading_at(long_now + 2.0)
            heading_diff = abs(vehicle.heading_theta - expected_heading)
            if heading_diff < 20:
                reward += 0.3
            else:
                reward -= 0.5
        except:
            reward -= 0.1

        try:
            final_long, _ = current_lane.local_coordinates(vehicle.navigation.checkpoints[-1])
            if abs(final_long - long_now) < 5:
                reward += 0.4
        except:
            pass

    # Steering smoothness
    if hasattr(vehicle, "steering_change"):
        steering_change = abs(vehicle.steering_change)
        if steering_change > 0.3:
            reward -= 0.2
        else:
            reward += 0.1

    # Add more penalty for crashing into sidewalk
    if vehicle.crash_sidewalk:
        reward -= 4.0

    # Avoid crashing with other vehicles (Not apply now for not using lidar data)
    detected_objects = vehicle.lidar.get_surrounding_objects(vehicle)
    front_cars = []
    side_lanes_clear = {"left": True, "right": True}

    for obj in detected_objects:
        if hasattr(obj, "position"):
            rel_pos = vehicle.convert_to_local_coordinates(obj.position, vehicle.position)
            rel_x, rel_y = rel_pos[0], rel_pos[1]
            dist = np.linalg.norm(rel_pos)
            if 0 < rel_x < 25 and abs(rel_y) < 2.0:
                front_cars.append((obj, dist))
            if abs(rel_y) > 2.0 and dist < 20:
                if rel_y > 0:
                    side_lanes_clear["left"] = False
                else:
                    side_lanes_clear["right"] = False

    front_cars.sort(key=lambda x: x[1])

    if front_cars:
        if side_lanes_clear["left"] or side_lanes_clear["right"]:
            reward += 0.2
        else:
            reward -= 0.5
            if vehicle.speed_km_h > 20:
                reward -= 0.5

    # Terminal state
    if env._is_arrive_destination(vehicle):
        reward = +env.config["success_reward"]
    elif env._is_out_of_road(vehicle):
        reward = -env.config["out_of_road_penalty"]
    elif vehicle.crash_vehicle:
        reward = -env.config["crash_vehicle_penalty"]
    elif vehicle.crash_object:
        reward = -env.config["crash_object_penalty"]
    elif vehicle.crash_sidewalk:
        reward = -env.config["crash_sidewalk_penalty"]

    step_info["step_reward"] = reward
    step_info["route_completion"] = vehicle.navigation.route_completion
    return reward, step_info
