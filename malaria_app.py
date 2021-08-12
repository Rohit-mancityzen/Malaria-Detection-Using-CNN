from flask import Flask, render_template, request
from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model('malaria.h5')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded/image'

@app.route('/')
def upload_f():
    return render_template('index.html')

def finds():
    test_datagen = ImageDataGenerator(rescale = 1./255)
    #vals = ['Malaria Infected', 'Uninfected']
    test_dir = 'uploaded'
    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size =(64, 64),
            color_mode ="grayscale",
            shuffle = False,
            class_mode ='categorical',
            batch_size = 1)

    pred = model.predict_generator(test_generator)
    print(pred)
    a = pred[0][0]
    pred = np.delete(pred,0,0)
    print(a)
    print(pred)
    c = 0
    if a>0.5:
        return str(" Unifected"+", Accuracy="+str(round((a)*100,4))+"%")
    else:
        return str(" Malaria Infected"+", Accuracy="+str(round((1-a)*100,4))+"%")
    c = c+1

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        val = finds()
        return render_template('index.html', ss = val)

if __name__ == '__main__':
    app.run()
