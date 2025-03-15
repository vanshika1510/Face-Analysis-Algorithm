import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# List of model names
model_names = ['EfficientNetB0', 'EfficientNetV2B0', 'MobileNetV2', 'MobileNetV3Small', 'ResNet50', 'vgg16']

# Directory for JSON files
json_dir = 'Json Files/'

# Directory for output plots
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Initialize dictionary to store final training accuracy
model_accuracies = {}

for model_name in model_names:
    model_data = pd.read_json(os.path.join(json_dir, f'{model_name}_history.json'))
    
    # Extract final training accuracy
    final_accuracy = model_data["accuracy"].iloc[-1]
    model_accuracies[model_name] = final_accuracy

# Add Roboflow accuracy
model_accuracies["Roboflow"] = 97.1 / 100  # Convert to 0-1 scale

# Convert to DataFrame for table display
accuracy_df = pd.DataFrame(list(model_accuracies.items()), columns=["Model", "Final Training Accuracy"])
accuracy_df = accuracy_df.sort_values(by="Final Training Accuracy", ascending=False)

# Display table
print(accuracy_df)

# Create Pie Chart
plt.figure(figsize=(8, 8))
sns.set_palette("husl")
plt.pie(model_accuracies.values(), labels=model_accuracies.keys(), autopct='%1.1f%%', startangle=140, 
        wedgeprops={'edgecolor': 'black'}, pctdistance=0.85)

# Title
plt.title("Training Accuracy Distribution")

# Save Pie Chart
plt.savefig(os.path.join(output_dir, 'train_accuracy_piechart.png'), bbox_inches='tight')

# Show Pie Chart
plt.show()
