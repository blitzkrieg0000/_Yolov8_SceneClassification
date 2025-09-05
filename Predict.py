import os
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from ultralytics import YOLO

# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


#! Params
RESULT_PATH = "./results/classification_train"
TEST_PATH = "./dataset/VR01/Real/test"
BATCH = 32

#!LOAD DATA
DATATRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()
])

test_dataset = ImageFolder(TEST_PATH, DATATRANSFORMS)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, drop_last=True)

#! LOAD MODEL
model = YOLO(os.path.join(RESULT_PATH, "weights/best.pt")).to(DEVICE)
nc = len(test_dataset.classes)

#! Test Model----------------------------------------------------------------------------------------------------------
# Modelin doğru tahmin etme oranını hesapla
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(nc)]
    n_class_samples = [0 for i in range(nc)]
    pred_y = []
    ground_y = []
    for images, labels in test_loader:
        images = images.to(DEVICE)
        # labels = labels.to(DEVICE)

        # Inference
        outputs = model(images, verbose=False)
        
        predicted = torch.Tensor([res.probs.top1 for res in outputs])
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        pred_y += [*np.array(predicted.numpy(), dtype=int)]
        ground_y += [*np.array(labels.numpy(), dtype=int)]
        for i in range(BATCH):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network: {acc:.4f} %")

    for i in range(nc):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f"Accuracy of {outputs[0].names[i]}: {acc:.2f} %")        
                
np.save(os.path.join(RESULT_PATH, "labels.npy") , outputs[0].names)
np.save(os.path.join(RESULT_PATH, "real.npy") , ground_y)
np.save(os.path.join(RESULT_PATH, "predicted.npy"), pred_y, allow_pickle=True)