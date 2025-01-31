import torch
import os
from torchvision.models import resnet50

# Step 1: Define the model
model = resnet50(num_classes=751)  # Adjust num_classes for your dataset

# Step 2: Load the .pth checkpoint (replace with your checkpoint file path)
checkpoint_path = '/home/arsenalducvy/Desktop/training_output/market1501/ckpt.pth'

if os.path.isfile(checkpoint_path):
    print(f"=> Loading checkpoint '{checkpoint_path}'")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Step 1: Inspect the structure of the checkpoint
    print("Checkpoint type:", type(checkpoint))  # Should be a dict or list
    if isinstance(checkpoint, dict):
        print("Keys in the checkpoint:", checkpoint.keys())  # Print keys to understand structure
    elif isinstance(checkpoint, list):
        print("Checkpoint is a list of length:", len(checkpoint))  # Print the length of the list
        # Print the first element to inspect its structure
        if len(checkpoint) > 0:
            print("First element:", checkpoint[0])
    else:
        raise ValueError("Checkpoint is neither a dict nor a list.")
else:
    print(f"=> No checkpoint found at '{checkpoint_path}'")

# Set the model to evaluation mode
model.eval()

# Step 4: Prepare an example input tensor
example_input = torch.randn(1, 3, 224, 224)  # Adjust batch size and shape as necessary

# Step 5: Export the model to ONNX
onnx_path = "resnet50_model.onnx"
torch.onnx.export(
    model,                      # The model being exported
    example_input,              # A sample input tensor
    onnx_path,                  # Output ONNX file path
    export_params=True,         # Store the trained parameter weights inside the model file
    opset_version=11,           # ONNX opset version (change if needed)
    do_constant_folding=True,   # Whether to execute constant folding for optimization
    input_names=['input'],      # The model's input names
    output_names=['output'],    # The model's output names
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Enable dynamic batching
)

print(f"Model successfully converted to ONNX and saved at {onnx_path}")
