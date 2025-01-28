import torch
import torchviz
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from simulator import simulate
from wear_fn import apply_wear
from utils import in2pix
from model import TrafficCNN  # Ensure your model is imported

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = TrafficCNN().to(device)
model.load_state_dict(torch.load("traffic_cnn.pth", map_location=device))
model.eval()

# Print Model Summary
print("\n📜 Model Summary:")
try:
    summary(model, (1, 30, 120))  # Adjust shape to match step profile
except Exception as e:
    print(f"❌ Error in summary: {e}")

# Step dimensions
step_width = in2pix(60)
step_tread = in2pix(15)
step_rise = in2pix(20)
foot_dims = (in2pix(6), in2pix(10))

# Simulate a step profile
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

steps_per_day = np.random.randint(20, 80)
total_days = np.random.randint(6, 20) * 365
conditions = np.append(true_distribution, steps_per_day)

# Generate step profile using the simulator
step_profile = simulate(step_tread, step_width, step_rise, foot_dims, total_days, apply_wear, conditions)

# Convert to tensor and ensure correct shape [batch, channels, height, width]
step_profile_tensor = torch.tensor(step_profile, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)

# Generate Model Graph
try:
    output = model(step_profile_tensor)  # Ensure model produces an output
    model_graph = torchviz.make_dot(output, params=dict(model.named_parameters()))

    # Save and Display the Graph
    model_graph.render("traffic_cnn_architecture", format="png")

    # Show the visualization
    img = plt.imread("traffic_cnn_architecture.png")
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

except Exception as e:
    print(f"❌ Error in visualization: {e}")
