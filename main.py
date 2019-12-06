import functools
import inspect
import math
import sys


def reward_f(scale=1.0):
    def wrap(f):
        def wrapped_f(params, reward):
            return reward + f(params, reward) * scale

        wrapped_f.scale = scale
        return wrapped_f

    return wrap


@reward_f(scale=2.0)
def stay_near_center(params, reward):
    """Keep the car close to centerline"""
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    rel_dist = distance_from_center / (track_width / 2)
    new_reward = max(0, min(1, 1 - (rel_dist - 0.3)))
    return new_reward


@reward_f(scale=1.0)
def increasing_progress(params, reward):
    """The closer to the finish, the higher the reward"""
    progress = params['progress']
    new_reward = progress * .01
    return new_reward


@reward_f(scale=1.0)
def reward_speed(params, reward):
    """Reward going fast"""
    progress = params['progress']
    steps = params['steps']
    TOTAL_NUM_STEPS = 300
    # New reward should be ~1.0 throughout lap to complete within TOTAL_NUM_STEPS
    new_reward = progress / 100.0 / steps * TOTAL_NUM_STEPS
    return new_reward


@reward_f(scale=1.0)
def reduce_high_speed_steering(params, reward):
    """Penalise steering, more at higher speed"""
    speed = params['speed']
    steering_angle = params['steering_angle']
    new_reward = 1 - 0.02 * speed * abs(steering_angle)
    return new_reward


@reward_f(scale=1.0)
def heading_reward(params, reward):
    """Reward going in the right direction"""
    heading = params['heading']
    track_direction = _track_direction(params, waypoints_ahead=1)
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff
    new_reward = max(1 - direction_diff / 30.0, 0.0)
    return new_reward


@reward_f(scale=2.0)
def steering_heading_reward(params, reward):
    """Reward steering in the right direction"""
    heading = params['heading']
    steering_angle = params['steering_angle']
    track_direction = _track_direction(params, waypoints_ahead=3)
    direction_diff = abs(track_direction - heading - steering_angle)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff
    new_reward = max(1 - direction_diff / 30.0, 0.0)
    return new_reward


def _track_direction(params, waypoints_ahead=0):
    """Return track direction in degrees for current car position"""
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    next_point = waypoints[(closest_waypoints[1] + waypoints_ahead) % len(waypoints)]
    prev_point = waypoints[closest_waypoints[0]]
    track_direction = math.degrees(math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]))
    return track_direction


reward_functions = [
    o[1]  # the function
    for o in inspect.getmembers(
        sys.modules[__name__],  # current module
        lambda f: hasattr(f, 'scale')
    )
    ]


def reward_function(params):
    init_reward = .1
    return functools.reduce(
        lambda r1, r2: r2(params, r1),
        [init_reward] + reward_functions,
    )


def test():
    inp = {
        'progress': 50,
        'steps': 150,
        'track_width': 0.6,
        'distance_from_center': 0.0,
        'speed': 1.0,
        'steering_angle': 15,
        'waypoints': [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        'closest_waypoints': [3, 4],
        'heading': 45.0,
    }
    print(reward_function(inp))


if __name__ == '__main__':
    test()
