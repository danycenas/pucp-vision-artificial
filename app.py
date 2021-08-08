import os
import sys
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import json

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from PIL import Image

# from imageai.Prediction import ImagePrediction
# execution_path = os.getcwd()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'img'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def upload_form():
   app.logger.info('/raiz')
   user = {'firstname': 'Harry', 'lastname': 'Potter'}
   return render_template('index.html', user=user)
	
def allowed_file(filename):
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpg'])
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/file-upload-batch/2', methods=['GET', 'POST'])
def upload_file2():
   if request.method == 'POST':

      file = request.files['input44[]']
      if file and allowed_file(file.filename):
         filename = secure_filename(file.filename)
         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

      # files = request.files.getlist('input44[]')
      # for file in files:
      #    if file and allowed_file(file.filename):
      #          filename = secure_filename(file.filename)
      #          file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

      app.logger.info('*********************************')

      # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      device = torch.device('cpu')

      ft_model = models.resnet18(pretrained=True)
      ft_model.fc = nn.Linear(512, 8)
      ft_model_state_dict = torch.load('model.pth', map_location=device)
      ft_model.load_state_dict(ft_model_state_dict)
      ft_model = ft_model.to(device)

      ft_optimizer = torch.optim.SGD(ft_model.parameters(), lr=0.001, momentum=0.9)
      ft_optimizer_state_dict = torch.load('optimizer.pth', map_location=device)
      ft_optimizer.load_state_dict(ft_optimizer_state_dict)

      app.logger.info("File: " + filename)
      img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))

      preprocess = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])

      input_tensor = preprocess(img)
      input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

      # move the input and model to GPU for speed if available
      if torch.cuda.is_available():
         input_batch = input_batch.to('cuda')
         ft_model.to('cuda')

      ft_model.eval()

      with torch.no_grad():
         output = ft_model(input_batch)
      # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
      # print(output)
      # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
      # probabilities = torch.nn.functional.softmax(output[0], dim=0)
      probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
      # print(probabilities)

      class_names = ['auto', 'bulldozer', 'camion', 'camion_minero', 'camioneta', 'excavadora', 'otro', 'persona']

      predictions = []

      top5_prob, top5_catid = torch.topk(probabilities, 8)
      for i in range(top5_prob.size(0)):
         app.logger.info(class_names[top5_catid[i]] + ' ' + str(top5_prob[i].item()))
         predictions.append([ class_names[top5_catid[i]], top5_prob[i].item() ])
      
      app.logger.info(predictions)

      app.logger.info('*********************************')

      # return jsonify({'file': 'success'})
      result = {'maquinaria_pesada': 'si', 'probabilidad': '70%'}
      return json.dumps(predictions)

@app.route('/img/<filename>')
def send_uploaded_file(filename=''):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
		
if __name__ == '__main__':
   app.run(host='0.0.0.0')