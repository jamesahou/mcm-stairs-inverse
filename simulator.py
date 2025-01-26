import numpy as np
import matplotlib.pyplot as plt
from sample_fn import *
from wear_fn import *
from utils import in2pix, pix2in

def plot_step_profile(step_profile):
    step_profile_outlined = np.zeros((step_profile.shape[0] + 2, step_profile.shape[1] + 2))
    step_profile_outlined[1:-1, 1:-1] = step_profile
    fig = plt.figure()
    xx, yy = np.mgrid[0:step_profile_outlined.shape[0], 0:step_profile_outlined.shape[1]]
    ax = fig.add_subplot(projection = '3d')
    ax.set_xlim(-step_profile_outlined.shape[1] // 2 + 20, step_profile_outlined.shape[1] // 2 + 20)
    ax.set_ylim(-10, step_profile_outlined.shape[1]+10)
    ax.invert_xaxis()
    ax.set_zlim(0, np.max(step_profile_outlined)+10)
    ax.plot_surface(xx, yy, step_profile_outlined ,rstride=1, cstride=1, cmap='viridis',
            linewidth=0)
    plt.show()


def simulate(length, width, height, foot_dims, days, wear_fn, conditions):
    samplers = [up_middle_sampler_two_feet, up_left_sampler_two_feet, up_right_sampler_two_feet, down_middle_sampler_two_feet, down_left_sampler_two_feet, down_right_sampler_two_feet]
    # samplers = [up_middle_sampler, down_middle_sampler, up_left_sampler, down_left_sampler, up_right_sampler, down_right_sampler]
    p_sampler = conditions[:-1]
    steps_per_day = conditions[-1]
    total_steps = int(steps_per_day * days)

    step_profile = np.ones((length, width)) * height
    sampler_idx = np.random.choice(np.arange(len(samplers)), total_steps, p=p_sampler)
    for t in range(total_steps):
        sampler = samplers[sampler_idx[t]]
        UP_FLAG = sampler_idx[t] < 3

        (foot_x, foot_y) = sampler(length, width, height, foot_dims)
        foot_ybottom = foot_y - foot_dims[1]//2
        foot_ytop = foot_ybottom + foot_dims[1]

        foot_ybottom = int(np.clip(foot_ybottom, 0, length-1))
        foot_ytop = int(np.clip(foot_ytop, 0, length-1))

        foot_xleft = foot_x - foot_dims[0]//2
        foot_xright = foot_xleft + foot_dims[0]

        foot_xleft = int(np.clip(foot_xleft, 0, width-1))
        foot_xright = int(np.clip(foot_xright, 0, width-1))

        wear_fn(step_profile[foot_ybottom:foot_ytop, foot_xleft:foot_xright], UP_FLAG)

        # if t % (20 * steps_per_day) == 0:
            # print("Day", t // steps_per_day)
            # plot_step_profile(step_profile)

    return step_profile


if __name__ == "__main__":
    step_width = in2pix(60)
    step_tread = in2pix(15)
    step_rise = in2pix(8)

    foot_dims = (in2pix(6), in2pix(10))

    profile = simulate(step_tread, step_width, step_rise, foot_dims, 100, onepix_wear, np.array([0.2, 0.05, 0.05, 0.25, 0.2, 0.25, 200]))

    profile2 = simulate(step_tread, step_width, step_rise, foot_dims, 100, onepix_wear, np.array([0.2, 0.05, 0.05, 0.25, 0.2, 0.25, 200]))

    print(np.sum((profile - profile2)**2))

    profile_cat = np.concatenate((profile, profile2), axis=1)
    plot_step_profile(profile_cat)