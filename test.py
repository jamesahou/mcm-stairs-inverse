import torch
import numpy as np
import matplotlib.pyplot as plt
from simulator import simulate
from wear_fn import apply_wear
from utils import in2pix
from model import TrafficCNN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained inverse model
model = TrafficCNN().to(device)
model.load_state_dict(torch.load("traffic_cnn.pth", map_location=device))
model.eval()

# Step dimensions
step_width = in2pix(60)
step_tread = in2pix(15)
step_rise = in2pix(20)
foot_dims = (in2pix(6), in2pix(10))

def test_inverse_model(model, device, i):
    """Generates a step profile and evaluates how well the inverse model reconstructs the probability distribution."""
    
    # Step 1: Sample random probability distribution
    p_single = np.random.rand()
    p_double = 1 - p_single

    sampler_scores = np.array([
        p_single * np.random.rand(),
        p_double * np.random.rand(),
        p_double * np.random.rand(),
        p_single * np.random.rand(),
        p_double * np.random.rand(),
        p_double * np.random.rand()
    ])
    true_distribution = sampler_scores / np.sum(sampler_scores)

    # Step 2: Generate step profile
    steps_per_day = np.random.randint(20, 80)
    total_days = np.random.randint(6, 20) * 365
    conditions = np.append(true_distribution, steps_per_day)

    step_profile = simulate(step_tread, step_width, step_rise, foot_dims, total_days, apply_wear, conditions)

    # Step 3: Convert step profile to tensor
    step_profile_tensor = torch.tensor(step_profile, dtype=torch.float32).to(device).unsqueeze(0)  # Add batch dimension
    print(step_profile_tensor.shape)

    # Step 4: Pass through model to get predicted distribution
    with torch.no_grad():
        predicted_distribution = model(step_profile_tensor).squeeze(0)  # Remove batch dimension

    # Step 5: Apply softmax to convert logits into probabilities
    predicted_distribution = torch.softmax(predicted_distribution, dim=0).cpu().numpy()

    # Step 6: Print & Compare Results
    print("\nTrue Probability Distribution:", true_distribution)
    print("Predicted Probability Distribution:", predicted_distribution)

    # Step 7: Plot true vs. predicted distribution
    plt.figure(figsize=(8, 5))
    x_labels = np.arange(1, len(true_distribution) + 1)

    plt.bar(x_labels - 0.2, true_distribution, width=0.4, label="True Distribution", alpha=0.7)
    plt.bar(x_labels + 0.2, predicted_distribution, width=0.4, label="Predicted Distribution", alpha=0.7)

    plt.xlabel("Sampler Index")
    plt.ylabel("Probability")
    plt.title("True vs. Predicted Probability Distribution")
    plt.xticks(x_labels)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figs/{i}.png")

# Run the test
for i in range(100):
    test_inverse_model(model, device, i)
