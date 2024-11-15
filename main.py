import functools
import inspect
import sys
import math

MAX_SPEED = 1.0
SPEED_STEPS = 2


def reward_f(scale=1.0):
    def wrap(f):
        def wrapped_f(params, reward):
            return reward + f(params, reward) * scale

        wrapped_f.scale = scale
        return wrapped_f

    return wrap


@reward_f(scale=2.0)
def stay_on_track(params, reward):
    all_wheels_on_track = params['all_wheels_on_track']
    return 1.0 if all_wheels_on_track else 0.0


@reward_f(scale=1.0)
def reduce_high_speed_steering(params, reward):
    """Penalise steering, more at higher speed"""
    speed = params['speed']
    steering_angle = params['steering_angle']
    new_reward = 1 - 0.02 * speed * abs(steering_angle)
    return new_reward


@reward_f(scale=1.0)
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


@reward_f(scale=2.0)
def no_steering_on_straight(params, reward):
    """Reward not steering on a straight track"""
    steering_angle = params['steering_angle']
    if abs(_track_curve(params)) < 1.0:
        if abs(steering_angle) < 1.0:
            return 1.0
    return 0.0


@reward_f(scale=3.0)
def speedup_on_straight(params, reward):
    speed = params['speed']
    if abs(_track_curve(params)) < 1.0:
        if abs(speed - MAX_SPEED) < 0.1:
            return 1.0
        else:
            return 0.0
    else:
        if abs(speed - MAX_SPEED) < 0.1:
            return 0.0
        else:
            return 1.0


def _track_direction(params, waypoints_ahead=0, waypoints_arear=0):
    """Return track direction in degrees for current car position"""
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    next_point = waypoints[(closest_waypoints[1] + waypoints_ahead) % len(waypoints)]
    prev_point = waypoints[(closest_waypoints[0] - waypoints_arear) % len(waypoints)]
    track_direction = math.degrees(math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]))
    return track_direction


def _track_curve(params):
    """Return the change in track direction between current position and next waypoints"""
    curr_direction = _track_direction(params, waypoints_ahead=0, waypoints_arear=0)
    fut_direction = _track_direction(params, waypoints_ahead=3, waypoints_arear=-1)
    return fut_direction - curr_direction


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
        'all_wheels_on_track': True,
    }
    print(reward_function(inp))


if __name__ == '__main__':
    test()
