import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from models import ResNet2D

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the hyperparameters
num_epochs = 10
batch_size = 128
learning_rate = 0.001

# Define the transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the dataset
train_dataset = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='data/', train=False, transform=transform, download=True)

# Create the dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
model = ResNet2D().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the summary writer
writer = SummaryWriter('runs/mnist')

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()

        # Print statistics
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), accuracy.item() * 100))

        # Write to tensorboard
        writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar('training accuracy', accuracy.item(), epoch * len(train_loader) + i)

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, argmax = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (labels == argmax.squeeze()).sum().item()

        accuracy = correct / total
        print('Test Accuracy of the model on the 10000 test images: {:.2f}%'.format(accuracy * 100))

    # Write to tensorboard
    writer.add_scalar('test accuracy', accuracy, epoch)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'resnet2d_mnist.ckpt')

# Close the summary writer
writer.close()
