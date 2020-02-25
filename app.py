from flask import Flask,render_template,url_for,request, send_from_directory, redirect, flash, Markup
from werkzeug.utils import secure_filename

import numpy as np
import shutil
import os
import cv2
import random
from matplotlib import colors, pyplot as plt
import json

import requests



import torch
import torch.nn as nn
import numpy as np
import PIL
from PIL import Image
import pickle
from pathlib import Path
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import cv2
import os
DEVICE = torch.device("cpu")

RESCALE_SIZE = 512


#UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = 'some_secret'


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, "static")


 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
      


@app.route('/about')
def about():
	return render_template('about.html')
	



class GemstonesDataset(Dataset):
  
    
    def __init__(self, files, mode):
        super().__init__()
        # список файлов для загрузки
        self.files = files
        # режим работы
        self.mode = mode

        
        self.len_ = len(self.files)
     
        self.label_encoder = LabelEncoder()

        
                      
    def __len__(self):
        return self.len_
      
    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image
  
    def __getitem__(self, index):
        
        
        if self.mode == 'test':
          transform = transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
                 
                       
          x = self.load_sample(self.files)
          x = self._prepare_sample(x)
          x = np.array(x / 255, dtype='float32')
          x = transform(x)
          return x
        
        
    def _prepare_sample(self, image):
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)





def predict_one_sample1(model, inputs, device=DEVICE):
    """Предсказание, для одной картинки"""
    with torch.no_grad():
        inputs = inputs
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
        

    return probs
	
	
	
	
@app.route('/gemstones', methods=['GET', 'POST'])
def upload_file1():
		
    if request.method == 'POST':
		
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            shutil.copy(os.path.join(app.config['UPLOAD_FOLDER'], filename),os.path.join(app.config['UPLOAD_FOLDER'], 'stone_image.jpg') )
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Загружено. Нажмите "Определить" и подождите.')
            return redirect(url_for('upload_file1'))
    
    
    
    return render_template('gemstones.html')	
	
	
	
	
@app.route('/predict_gemstones', methods=['GET','POST'])	
def predict_gemstones():
	
    
  
  n_classes = 87
  
  
  test_labels = ['Alexandrite', 'Almandine', 'Amazonite', 'Amber', 'Amethyst', 'Ametrine', 'Andalusite', 'Andradite', 'Aquamarine', 'Aventurine', 'Aventurine Green',  'Benitoite', 'Beryl Golden', 'Beryl Red', 'Bloodstone', 'Blue Lace Agate', 'Carnelian', 'Cats Eye',  'Chalcedony',  'Chalcedony Blue', 'Chrome Diopside', 'Chrysoberyl', 'Chrysocolla', 'Chrysoprase', 'Citrine', 'Coral', 'Danburite', 'Diamond', 'Diaspore', 'Dumortierite', 'Emerald', 'Fluorite', 'Garnet Red',  'Goshenite', 'Grossular', 'Hessonite', 'Hiddenite', 'Iolite', 'Jade', 'Jasper', 'Kunzite', 'Kyanite', 'Labradorite', 'Lapis Lazuli', 'Larimar', 'Malachite', 'Moonstone', 'Morganite',  'Onyx Black', 'Onyx Green', 'Onyx Red', 'Opal', 'Pearl','Peridot', 'Prehnite', 'Pyrite', 'Pyrope', 'Quartz Beer', 'Quartz Lemon', 'Quartz Rose', 'Quartz Rutilated', 'Quartz Smoky','Rhodochrosite', 'Rhodolite', 'Rhodonite', 'Ruby', 'Sapphire Blue', 'Sapphire Pink', 'Sapphire Purple', 'Sapphire Yellow','Scapolite', 'Serpentine', 'Sodalite', 'Spessartite', 'Sphene', 'Spinel', 'Spodumene',  'Sunstone', 'Tanzanite', 'Tigers Eye', 'Topaz', 'Tourmaline',  'Tsavorite','Turquoise', 'Variscite', 'Zircon', 'Zoisite']
  test_labels1 = ['александрит', 'альмандин', 'амазонит', 'янтарь', 'аметист', 'аметрин', 'андалузит', 'андрадит', 'аквамарин', 'авантюрин', 'авантюрин зеленый','бенитоит', 'берилл желтый', 'берилл красный', 'кровавый камень', 'голубой кружевной агат', 'сердолик', 'кошачий глаз',  'халцедон',  'голубой халцедон', 'хромдиопсид', 'хризоберилл', 'хризоколла', 'хризопраз', 'цитрин', 'коралл', 'данбурит', 'алмаз', 'диаспор', 'дюмортьерит', 'изумруд', 'флюорит', 'красный гранат',  'гошенит', 'гроссуляр', 'гессонит', 'гидденит', 'иолит', 'нефрит', 'яшма', 'кунцит', 'кианит', 'лабрадорит', 'лазурит', 'ларимар', 'малахит', 'лунный камень', 'морганит',  'оникс черный', 'оникс зеленый', 'оникс красный', 'опал', 'жемчуг','перидот (хризолит)', 'пренит', 'пирит', 'пироп (гранат)', 'кварц пивной', 'кварц лимонный', 'кварц розовый', 'кварц рутиловый ', 'кварц дымчатый','родохрозит', 'родолит', 'родонит', 'рубин', 'сапфир голубой', 'сапфир розовый', 'сапфир пурпурный', 'сапфир желты','скаполит', 'серпентин', 'содалит', 'спессартин (гранат)', 'сфен (титанит)', 'шпинель', 'сподумен',  'солнечный камень', 'танзанит', 'тигровый глаз', 'топаз', 'турмалин',  'цаворит','бирюза', 'варисцит', 'циркон', 'цоизит']
  img = os.path.join(app.config['UPLOAD_FOLDER'], 'stone_image.jpg')
  test_files = str(img)
  
  
  label_encoder = LabelEncoder()
  label_encoder.fit(test_labels)
  with open('C:/Users/GOODBUY/Downloads/gemstones/data/gemstones/label_encoder.pkl', 'wb') as le_dump_file:
	  pickle.dump(label_encoder, le_dump_file)
            
  
  test_dataset = GemstonesDataset(test_files, mode='test')
  
  
  
  from torchvision.models.resnet import resnet50
  resnet_model=resnet50(pretrained=True).to(DEVICE)
  resnet_model.fc = nn.Sequential(
    nn.Dropout2d(),
    nn.Linear(resnet_model.fc.in_features, out_features=n_classes)
    )
  
  resnet_model.load_state_dict(torch.load("C:/Users/GOODBUY/Downloads/gemstones/data/gemstones/resnet_model_weights.pth", map_location=torch.device('cpu')))
    
  
  ex_img = test_dataset[0]
  
  
  probs_im_resnet = predict_one_sample1(resnet_model, ex_img.unsqueeze(0))
  

  
  predicted_proba_r = np.max(probs_im_resnet)*100
  predicted_proba_lr = np.argmax((probs_im_resnet))
  pthr=label_encoder.classes_[predicted_proba_lr]
  for i in range(len(test_labels)) :
	  if test_labels[i] == pthr:
		  pthr=test_labels1[i]
	  
    
  
  flash('Это камень {}.'.format(str(pthr)))
  flash('Нажмите "Очистить" перед загрузкой следующией картинки.')
  
  im1= str(os.path.join('C:/Users/GOODBUY/Documents/flask_project/static', str.lower(pthr)+'_0.jpg'))
  
  
 
  return render_template('gemstones.html',name1=pthr, im1=im1) 






@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


if __name__ == '__main__':
  app.run(debug=True)

