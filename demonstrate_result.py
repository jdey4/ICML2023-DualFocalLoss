#%%
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from keras.models import Model
from kdg import kdf, kdn, get_ece, get_ace
import pickle
from tensorflow.keras.datasets import cifar10, cifar100
import timeit
from scipy.io import loadmat
import random
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch 
from tqdm import tqdm

#%%
def fpr_at_95_tpr(conf_in, conf_out):
    TPR = 95
    PERC = np.percentile(conf_in, 100-TPR)
    #FP = np.sum(conf_out >=  PERC)
    FPR = np.sum(conf_out >=  PERC)/len(conf_out)
    return FPR

# %%
# ImageNet normalization (after scaling to [0,1])
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [0,1]
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])


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

# %%
seeds = [0, 1, 2, 3, 4]
auroc_noise = []
fpr_noise = []
oce_noise = []
auroc_cifar100 = []
fpr_cifar100 = []
oce_cifar100 = []
auroc_svhn = []
fpr_svhn = []
oce_svhn = []
accuracy = []
mce = []
mmc = []
ace = []

for seed in tqdm(seeds):
    with open("/Users/jayantadey/kdg_rebuttal/ICML2023-DualFocalLoss/dual_focal_vit_cifar10_"+str(seed)+".pickle", 'rb') as f:
        (proba_in, proba_cifar100, proba_svhn, proba_noise) = pickle.load(f)

    total_data = len(cifar10_dataset)
    correct = 0
    labels = []
    for ii in range(total_data):
        labels.append(cifar10_dataset[ii][1])
        estimated_label = np.argmax(proba_in[ii])

        if labels[-1] == estimated_label:
            correct += 1
    
    labels = np.array(labels)
    idx = np.where(np.argmax(proba_in,axis=1)!=labels.ravel())[0]
    mmc.append(
        np.mean(np.max(proba_in[idx],axis=1))
    )
    accuracy.append(
        np.mean(correct/total_data)
    )
    mce.append(
        get_ece(proba_in, labels.ravel(),n_bins=30)
    )
    ace.append(
        get_ace(proba_in, labels.ravel(),R=30)
    )

    in_conf = np.max(proba_in, axis=1)
    out_conf = np.max(proba_cifar100, axis=1)
    conf_cifar100 = np.hstack((in_conf, out_conf))
    true_labels = np.hstack((np.ones(len(proba_in), ), np.zeros(len(proba_cifar100), )))
    auroc_cifar100.append(
        roc_auc_score(true_labels, conf_cifar100)
    )
    oce_cifar100.append(
        np.mean(np.abs(out_conf - 0.1))
    )
    fpr_cifar100.append(
        fpr_at_95_tpr(in_conf, out_conf)
    )



    out_conf = np.max(proba_svhn, axis=1)
    conf_svhn = np.hstack((in_conf, out_conf))
    true_labels = np.hstack((np.ones(len(proba_in), ), np.zeros(len(proba_svhn), )))
    auroc_svhn.append(
        roc_auc_score(true_labels, conf_svhn)
    )
    oce_svhn.append(
        np.mean(np.abs(out_conf - 0.1))
    )
    fpr_svhn.append(
        fpr_at_95_tpr(in_conf, out_conf)
    )

    out_conf = np.max(proba_noise, axis=1)
    conf_noise = np.hstack((in_conf, out_conf))
    true_labels = np.hstack((np.ones(len(proba_in), ), np.zeros(len(proba_noise), )))
    auroc_noise.append(
        roc_auc_score(true_labels, conf_noise)
    )
    oce_noise.append(
        np.mean(np.abs(out_conf - 0.1))
    )
    fpr_noise.append(
        fpr_at_95_tpr(in_conf, out_conf)
    )

    
 

    
# %%
print('accuracy ', np.mean(accuracy), '(+-',np.std(accuracy),')')
print('MCE ', np.mean(mce), '(+-',np.std(mce),')\n')
print('OE ACE ', np.mean(ace), '(+-',np.std(ace),')\n')
print('AUROC cifar100', np.mean(auroc_cifar100), '(+-',np.std(auroc_cifar100),')')
print('FPR@95 cifar100', np.mean(fpr_cifar100), '(+-',np.std(fpr_cifar100),')')
print('OCE cifar100', np.mean(oce_cifar100), '(+-',np.std(oce_cifar100),')')

print('AUROC svhn', np.mean(auroc_svhn), '(+-',np.std(auroc_svhn),')')
print('FPR@95 svhn', np.mean(fpr_svhn), '(+-',np.std(fpr_svhn),')')
print('OCE svhn', np.mean(oce_svhn), '(+-',np.std(oce_svhn),')')

print('AUROC noise', np.mean(auroc_noise), '(+-',np.std(auroc_noise),')')
print('FPR@95 noise', np.mean(fpr_noise), '(+-',np.std(fpr_noise),')')
print('OCE noise', np.mean(oce_noise), '(+-',np.std(oce_noise),')')

print('MMC', np.mean(mmc), '(+-',np.std(mmc),')')

# %%
