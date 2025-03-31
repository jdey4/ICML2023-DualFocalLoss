################################################################
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.models import vit_b_16
import pickle
import numpy as np
import random
from dual_focal_loss import DualFocalLoss
from tqdm import tqdm
# Define device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# ImageNet normalization (after scaling to [0,1])
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Define transforms: First normalize to [0,1], then apply ImageNet normalization
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [0,1]
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [0,1]
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Function to set seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Training function
def train_and_save_model(seed):
    print(f"\nTraining with seed {seed}...")
    set_seed(seed)

    # # Load CIFAR-100 dataset
    dataset = datasets.SVHN(root='./data', split='train', transform=transform_train, download=True)
    test_dataset = datasets.SVHN(root='./data', split='test', transform=transform_test, download=True)

    # Split training data into train/validation sets (70% train, 30% validation)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load Pretrained ViT-B/16
    model = vit_b_16(weights="IMAGENET1K_V1")
    model.heads = nn.Linear(model.heads.head.in_features, 10)  # Adjust for CIFAR-10 (10 classes)
    model.to(device)

    ########
    # checkpoint_path = f"vit_cifar100_seed{seed}.pth"
    # model.load_state_dict(torch.load(checkpoint_path))
    # Loss and optimizer
    criterion = DualFocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Training Loop
    num_epochs = 10  # Reduce if it's too slow
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Save the final trained model
    torch.save(model.state_dict(), f"vit_svhn_seed{seed}.pth")
    print(f"Model saved for seed {seed}.")

    # Evaluate and Save Predictions on CIFAR-100, SVHN, CIFAR-10 test, and noise data
    evaluate_and_save_predictions(model, seed)

# Function to evaluate on CIFAR-100, SVHN, CIFAR-10 test, and noise data
def evaluate_and_save_predictions(model, seed):
    model.eval()

    # Load CIFAR-100
    cifar100_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform_test, download=True)
    cifar100_loader = DataLoader(cifar100_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load SVHN
    svhn_dataset = datasets.SVHN(root='./data', split='test', transform=transform_test, download=True)
    svhn_loader = DataLoader(svhn_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load CIFAR-10 test set
    cifar10_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
    cifar10_loader = DataLoader(cifar10_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Generate noise data (values in [0,1])
    x_noise = np.random.randint(0, 256, size=(1000, 224, 224, 3)).astype('float32') / 255.0  # Normalize to [0,1]

    # Convert noise data to PyTorch tensors and apply ImageNet normalization
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    x_noise_tensors = torch.stack([transform_to_tensor(img) for img in x_noise])
    noise_dataset = TensorDataset(x_noise_tensors)
    noise_loader = DataLoader(noise_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Function to get predictions
    def get_predictions(loader, dataset_name, is_noise=False):
        all_preds = []
        all_labels = [] if not is_noise else None  # Noise has no labels

        with torch.no_grad():
            for batch in loader:
                if is_noise:
                    images = batch[0]  # Noise data has no labels
                    labels = None
                else:
                    images, labels = batch
                    labels = labels.to(device)

                images = images.to(device)
                outputs = model(images)
                preds = torch.softmax(outputs, dim=1).cpu().numpy()  # Get softmax probabilities
                all_preds.append(preds)
                
                if labels is not None:
                    all_labels.append(labels.cpu().numpy())

        return np.concatenate(all_preds), (np.concatenate(all_labels) if all_labels else None)


    # Get predictions
    proba_cifar100, cifar100_labels = get_predictions(cifar100_loader, "CIFAR-100")
    proba_in, svhn_labels = get_predictions(svhn_loader, "SVHN")
    proba_cifar10, cifar10_labels = get_predictions(cifar10_loader, "CIFAR-10 Test")
    proba_noise, _ = get_predictions(noise_loader, "Noise Data", is_noise=True)

    summary = (proba_in, proba_cifar10, proba_cifar100, proba_noise)

    # Save predictions to pickle files
    file_to_save = 'dual_focal_vit_svhn_'+str(seed)+'.pickle'

    with open(file_to_save, 'wb') as f:
        pickle.dump(summary, f)

# Run training and evaluation for multiple seeds
seeds = [0, 1, 2, 3, 4]
for seed in seeds:
    train_and_save_model(seed)
