#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 18:27:10 2024

@author: ahmetbakcaci
"""

from flask import Flask, request, jsonify, send_file
import io
import base64
#import your_ai_module  # Import your AI model

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    # Receive image data from the frontend
    image_data = request.files['image']

    # Preprocess image_data if necessary

    # Perform prediction using your AI model
    #prediction, generated_image = your_ai_module.predict_with_image(image_data)
    prediction = "3"
    # Convert generated image data to bytes
    """
    generated_image = image_data
    image_bytes = io.BytesIO()
    generated_image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    """
    encoded_image = base64.b64encode(image_data.read()).decode('utf-8')
    # Return the prediction and generated image as JSON response
    return jsonify({'prediction': prediction, 'gradImageData': encoded_image})

if __name__ == '__main__':
    app.run(host='172.20.10.5', port=8081)