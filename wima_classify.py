#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 18:27:10 2024

@author: ahmetbakcaci
"""

from flask import Flask, request, jsonify
import io
from model import ModelUtils
from PIL import Image
import torch
import base64

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    
    image_data = request.files['image'].read()
    
    image = Image.open(io.BytesIO(image_data))
    
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    weights_dir = './resnet18_4class_model_weights.pth'
    model_utils = ModelUtils(weights_dir, device)
    
    prediction, cam_image = model_utils.predict_and_visualize(image)
    
    prediction = str(prediction)
    
    cam_pil_image = Image.fromarray(cam_image)
    
    cam_pil_image = cam_pil_image.rotate(-90, expand=True)

    buffer = io.BytesIO()
    cam_pil_image.save(buffer, format='PNG')  # You can change the format as needed
    buffer.seek(0)
    
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return jsonify({'prediction': prediction, 'gradImageData': encoded_image})

if __name__ == '__main__':
    app.run(host='172.20.10.5', port=8081)
