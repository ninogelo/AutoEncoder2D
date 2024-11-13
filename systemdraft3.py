import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the Autoencoder Model
class AutoencoderModel(nn.Module):
    def __init__(self):
        super(AutoencoderModel, self).__init__()
        # Encoder: Adding more layers, filters, batch normalization, and dropout
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # Input: (1, 28, 28) -> Output: (32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: (64, 7, 7)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, padding=1),  # Output: (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # Decoder: Adding more layers and filters to reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),  # Output: (64, 7, 7)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # Output: (32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # Output: (1, 28, 28)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training Function with Real-Time Loss Plotting
def train_autoencoder(model, data_loader, device, num_epochs=20, learning_rate=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses = []

    # Set up real-time plotting
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Training Loss Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for data, _ in data_loader:  # Ignore labels
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate average loss and update plot
        avg_loss = total_loss / len(data_loader)
        losses.append(avg_loss)
        ax.clear()
        ax.plot(losses, label="Loss")
        ax.set_title("Training Loss Over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.pause(0.1)  # Pause to update the plot

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}')

    plt.ioff()
    plt.show()

# Main function
def main():
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize and train the Autoencoder
    model = AutoencoderModel().to(device)
    train_autoencoder(model, train_loader, device)

    # Test the Autoencoder on some images
    model.eval()
    data_iter = iter(train_loader)
    images, _ = next(data_iter)  # Get a batch of images
    images = images.to(device)

    with torch.no_grad():
        reconstructed_images = model(images).cpu()

    # Plot the original and reconstructed images
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        # Original images
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        # Reconstructed images
        axes[1, i].imshow(reconstructed_images[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
