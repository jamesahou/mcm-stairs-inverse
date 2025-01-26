from simulator import simulate, plot_step_profile
import numpy as np
from wear_fn import onepix_wear
from utils import in2pix
import time

num_samples = 10

step_width = in2pix(60)
step_tread = in2pix(15)
step_rise = in2pix(8)

foot_dims = (in2pix(6), in2pix(10))

start = time.time()
steps = []
for i in range(num_samples):
    p_single = np.random.rand()
    p_double = 1 - p_single

    # samplers = [up_middle_sampler_two_feet, up_left_sampler_two_feet, up_right_sampler_two_feet, down_middle_sampler_two_feet, down_left_sampler_two_feet, down_right_sampler_two_feet]
    sampler_scores = np.array([p_single * np.random.rand(), p_double * np.random.rand(), p_double * np.random.rand(), p_single * np.random.rand(), p_double * np.random.rand(), p_double * np.random.rand()])
    sampler_ps = sampler_scores / np.sum(sampler_scores)
    steps_per_day = np.random.randint(100, 300)
    total_days = np.random.randint(50, 150)

    conditions = np.append(sampler_ps, steps_per_day)
    profile = simulate(step_tread, step_width, step_rise, foot_dims, total_days, onepix_wear, conditions)
    
    np.save(f"data/profiles/profile_{i}.npy", profile)
    np.save(f"data/conditions/conditions_{i}.npy", conditions)

    if (i % 10 == 0):
        print(i)

# plot_step_profile(np.concatenate(steps, axis=1))
print(time.time() - start)