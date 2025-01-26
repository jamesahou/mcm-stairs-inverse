import numpy as np
from utils import in2pix, pix2in

interpedal_distance = in2pix(15)

def random_sampler(length, width, height, foot_dims):
    return (np.random.randint(foot_dims[0]//2, length-foot_dims[0]//2), np.random.randint(foot_dims[1]//2, width-foot_dims[1]//2))

def normal_sampler(dimension, mean, std):
    sample = np.random.normal(mean, std)

    clamped_sample = sample #np.clip(sample, 0, dimension-1)

    return int(clamped_sample)


## Assumption that human is single center of mass
def normal_left_sampler(length, width, height, foot_dims):
    y_mean = length // 2
    y_std = length // 8

    x_mean = width // 4
    x_std = width // 8

    return (normal_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))


def normal_right_sampler(length, width, height, foot_dims):
    y_mean = length // 2
    y_std = length // 8

    x_mean = 3 * width // 4
    x_std = width // 8

    return (normal_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))

def up_middle_sampler(length, width, height, foot_dims):
    y_mean = length // 4
    y_std = length // 8

    x_mean = width // 2
    x_std = width // 8

    return (normal_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))

def down_middle_sampler(length, width, height, foot_dims):
    y_mean = length // 2
    y_std = length // 8

    x_mean = width // 2
    x_std = width // 8

    return (normal_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))

def up_left_sampler(length, width, height, foot_dims):
    y_mean = length // 4
    y_std = length // 8

    x_mean = width // 4
    x_std = width // 8

    return (normal_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))

def down_left_sampler(length, width, height, foot_dims):
    y_mean = length // 2
    y_std = length // 8

    x_mean = width // 4
    x_std = width // 8

    return (normal_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))

def up_right_sampler(length, width, height, foot_dims):
    y_mean = length // 4
    y_std = length // 8

    x_mean = 3 * width // 4
    x_std = width // 8

    return (normal_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))

def down_right_sampler(length, width, height, foot_dims):
    y_mean = length // 2
    y_std = length // 8

    x_mean = 3 * width // 4
    x_std = width // 8

    return (normal_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))

## Assumption that human is two feet
def foot_sampler(dimension, mean, std):
    left_mean = mean - interpedal_distance // 2
    right_mean = mean + interpedal_distance // 2

    if np.random.rand() < 0.5:
        sample = np.random.normal(left_mean, std)
        clamped_sample = sample #np.clip(sample, 0, dimension-1)
    else:
        sample = np.random.normal(right_mean, std)
        clamped_sample = sample #np.clip(sample, 0, dimension-1)

    return int(clamped_sample)

def up_middle_sampler_two_feet(length, width, height, foot_dims):
    y_mean = length // 4
    y_std = length // 8

    x_mean = width // 2
    x_std = width // 8

    return (foot_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))


def down_middle_sampler_two_feet(length, width, height, foot_dims):
    y_mean = length // 2
    y_std = length // 8

    x_mean = width // 2
    x_std = width // 8

    return (foot_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))

def up_left_sampler_two_feet(length, width, height, foot_dims):
    y_mean = length // 4
    y_std = length // 8

    x_mean = width // 4
    x_std = width // 8

    return (foot_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))

def down_left_sampler_two_feet(length, width, height, foot_dims):
    y_mean = length // 2
    y_std = length // 8

    x_mean = width // 4
    x_std = width // 8

    return (foot_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))

def up_right_sampler_two_feet(length, width, height, foot_dims):
    y_mean = length // 4
    y_std = length // 8

    x_mean = 3 * width // 4
    x_std = width // 8

    return (foot_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))

def down_right_sampler_two_feet(length, width, height, foot_dims):
    y_mean = length // 2
    y_std = length // 8

    x_mean = 3 * width // 4
    x_std = width // 8

    return (foot_sampler(width, x_mean, x_std), normal_sampler(length, y_mean, y_std))