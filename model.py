#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 23:14:13 2024

@author: ahmetemin
"""

import torch
import numpy as np
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class ModelUtils:
    def __init__(self, model_weights_path, device):
        self.device = device
        self.model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.to(device)
        self.model.eval()

    def unnormalize(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        tensor = tensor.clone().detach()
        tensor.mul_(std).add_(mean)
        tensor = tensor.permute(1, 2, 0)
        tensor = tensor.cpu().numpy()
        tensor = np.clip(tensor, 0, 1)
        return tensor

    def predict_and_visualize(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)

        cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]])
        targets = [ClassifierOutputTarget(predicted[0].item())]

        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        img = self.unnormalize(image_tensor[0])
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        return predicted[0].item(), cam_image