import numpy as np
import matplotlib.pyplot as plt

# ==============================
# ENHANCED WEAR MODEL WITH PROPER UNITS
# ==============================

def archard_wear_factor(step_hardness_pa: float, is_descending: bool,
                        archard_k: float = 1e-5,  # Adjusted for stone materials
                        force: float = 700,       # 70kg person (N)
                        slip_distance: float = 0.5) -> float:
    """Proper implementation with hardness in Pascals"""
    return (archard_k * force * slip_distance) / step_hardness_pa

def directional_gaussian_kernel(matrix_shape: tuple, is_descending: bool) -> np.ndarray:
    """Stronger downward-shifted asymmetric wear pattern"""
    height, width = matrix_shape
    y, x = np.ogrid[-height//2:height//2, -width//2:width//2]

    # Increase downward shift
    y_shift = -height  # More aggressive downward shift
    kernel = np.exp(-(x**2 / (width/1.5)**2 + (y-y_shift)**2 / (height/1.8)**2))

    # Apply a stronger gradient boost for the lower section
    vertical_boost = np.linspace(1, 5, height).reshape(-1, 1)  # More wear at the bottom
    kernel *= vertical_boost

    return kernel



def apply_wear(step_matrix: np.ndarray, is_descending: bool) -> np.ndarray:
    """Improved wear application with stronger erosion effect"""

    if step_matrix.size == 0:
        return step_matrix


    kernel = directional_gaussian_kernel(step_matrix.shape, is_descending)

    # Normalize but prevent excessive dilution of wear
    kernel /= np.max(kernel)  # Use max instead of sum to retain impact

    wear_factor = 1.8 if is_descending else 1.0
    # Increase wear depth calculation
    wear_mm = archard_wear_factor(
        step_hardness_pa=1e9,
        is_descending=is_descending,
        archard_k=1e-4
    ) * 7500000 * wear_factor  # Increase factor from 1000 to 1e6 to amplify effect

    step_matrix -= kernel * wear_mm
    return np.clip(step_matrix, 0, np.inf, out=step_matrix)

# ==============================
# SIMULATION CONTROL FUNCTIONS
# ==============================

def run_simulation(initial_matrix: np.ndarray,
                  requested_steps: list,
                  descent_ratio: float = 0.9) -> dict:
    """Run simulation and save requested steps"""
    results = {}
    max_step = max(requested_steps)
    current_matrix = initial_matrix.copy()

    results[0] = current_matrix.copy()

    for step in range(1, max_step + 1):
        # Randomly determine step direction
        is_descending = np.random.rand() < descent_ratio
        current_matrix = apply_wear(current_matrix, is_descending)

        if step in requested_steps:
            results[step] = current_matrix.copy()

    return results

def visualize_results(results: dict, material_name: str):
    """Dynamic visualization with improved text contrast"""
    steps = sorted(results.keys())
    ncols = len(steps)

    fig, axs = plt.subplots(1, ncols, figsize=(ncols * 4, 5))

    # Find global min/max for consistent color scaling
    all_values = np.concatenate([m.flatten() for m in results.values()])
    vmin, vmax = np.min(all_values), np.max(all_values)

    for idx, step in enumerate(steps):
        ax = axs[idx] if ncols > 1 else axs
        mat = results[step]
        im = ax.imshow(mat, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')

        # Annotate values with dynamic color contrast
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                color = "white" if (value - vmin) / (vmax - vmin) < 0.5 else "black"
                ax.text(j, i, f"{value:.1f}", ha='center', va='center',
                        color=color, size=8, fontweight='bold')

        ax.set_title(f"{material_name}\nStep {step*1000}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


# ==============================
# MAIN EXECUTION (USER-CONFIGURABLE)
# ==============================

requested_steps = [10000]  # how many steps we want to see
matrix_shape = (12, 7)
# initial mm of material
initial_height = 350.0


# Run simulation for each material
# initial_matrix = np.full(matrix_shape, initial_height)
# results = run_simulation(initial_matrix, requested_steps)
# visualize_results(results, "granite")
