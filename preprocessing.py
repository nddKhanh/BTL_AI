from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Cấu hình
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load ảnh
train_dataset = datasets.ImageFolder("dataset/train/", transform=train_transform)
valid_dataset = datasets.ImageFolder("dataset/valid/", transform=test_transform)
test_dataset = datasets.ImageFolder("dataset/test/", transform=test_transform)

# Resize ảnh
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)