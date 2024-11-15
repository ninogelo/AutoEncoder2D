from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images
])

# Load the dataset
data_dir = "./chest_xray_pneumonia/chest_xray"  # Path to the dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Create a DataLoader for batching
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Check the dataset size
print("Number of images in the dataset:", len(dataset))
