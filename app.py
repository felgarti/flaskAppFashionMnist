import numpy as np
import pandas as pd
from flask import Flask, request, render_template

import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

from PIL import Image
from numpy import asarray

app = Flask(__name__)

model = keras.models.load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html' ,text ="Go ahead and upload an image ")


@app.route('/predict',methods=['POST'])
def predict():
    labels={0: "T-shirt/top", 1 : "Trouser", 2:"Pullover" , 3:"Dress",4:"Coat",5:"Sandal",6:"Shirt",7:"Sneaker",8:"Bag",9:"Ankle boot"}
    image=request.files["img"]
    img = Image.open(image)
    img.save('./images/'+image.filename)
    img=img.resize((28,28), Image.Resampling.LANCZOS)
    img = img.convert('L')
    data = asarray(img)


    inputs=data.reshape(1,28,28,1)
    # numpydata = asarray(img).reshape((1,1,784))
    # print(numpydata.shape)
    result= model.predict(inputs)
    print(result[0][7]  )

    for i in range(len(result[0])):
        if result[0][i]==np.float32(1) :
            label_result=labels[i]
            print(label_result)
    return render_template('index.html'  , text = label_result )




if __name__ == "__main__":
    app.run(debug=True)