import os
import sys
import math
import numpy as np
import torch
from torchvision import datasets, transforms
import random

# number of samples in each representation
n = 3000

subset = "val"
mode = "eval"

reps_folder = f"reps/{subset}/{n}_{mode}"
if not os.path.exists(reps_folder):
    os.makedirs(reps_folder)

batch_size = 100
total_batches = math.ceil(n / batch_size)
print(f"batch size: {batch_size}", flush=True)

model_names = []
file_names = os.listdir("/content/drive/MyDrive/UKP/imagenet_experiments/models")

model_names = file_names
random.Random(4).shuffle(model_names)

total_models = len(model_names)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
dataset = datasets.ImageFolder(f"/content/drive/MyDrive/UKP/imagenet_experiments/datasets/ImageNet/{subset}/", transform=transform)

print(reps_folder)

for model_name in model_names:
    print(f"Computing representation for {model_name}", flush=True)

    model = torch.load(f"/home/soumya/PycharmProjects/UKP/imagenet_experiments/models/{model_name}")
    if mode == "eval":
        model.eval()
    print(model, flush=True)

    g_cpu = torch.Generator()
    g_cpu.manual_seed(1234)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g_cpu, num_workers=0)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    try:
        print("trying classifier")
        model.classifier[-2].register_forward_hook(get_activation("pre_classifier"))
    except:
        print("not classifier")
        list(model.children())[-2].register_forward_hook(get_activation("pre_classifier"))

    rep = []
    i = 0
    for batch_data, batch_labels in loader:
        if i >= n:
            break
        nb = min(batch_data.shape[0], n-i)
        with torch.no_grad():
            model(batch_data)
            rep.append(activation["pre_classifier"].flatten(start_dim=1)[0:nb, :])
        i += nb
        print(f"{i} / {n}", flush=True)
        print(batch_labels, flush=True)
        print(flush=True)
    rep = np.vstack(rep)

    np.save(f"{reps_folder}/{model_name[:-4]}_rep.npy", rep.T)
