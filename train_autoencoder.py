import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Improved Autoencoder Model Using VGG16 Encoder
class ImprovedAutoencoder(nn.Module):
    def __init__(self):
        super(ImprovedAutoencoder, self).__init__()

        # Use a pre-trained VGG16 model as the encoder
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(vgg16.features.children())[:-1])  # Remove the last max-pooling layer

        # Decoder: Build a decoder to reconstruct images from the encoded features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Use Sigmoid to keep the output between 0 and 1
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
        plt.pause(0.1)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}')

    plt.ioff()
    plt.show()


# Main function
def main():
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the chest X-ray dataset with the updated transformation
    data_dir = "./chest_xray_pneumonia/chest_xray"  # Path to your downloaded dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize and train the Autoencoder
    model = ImprovedAutoencoder().to(device)
    train_autoencoder(model, data_loader, device)


if __name__ == "__main__":
    main()
