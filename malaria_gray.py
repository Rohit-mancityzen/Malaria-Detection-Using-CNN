import pandas as pd
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as mp

DATASET="TRAIN"
TEST_DATASET="TEST"

CATEGORIES=["Parasitized", "Uninfected"]

train_data=[]
test_data=[]
file=[]

for category in CATEGORIES:
        label=CATEGORIES.index(category)
        path=os.path.join(DATASET, category)
        for img_file in os.listdir(path):
            file.append(img_file)
            img=cv.imread(os.path.join(path, img_file), 1)
            img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            img=cv.resize(img,(64, 64))            
            train_data.append([img, label])

for category in CATEGORIES:
        label=CATEGORIES.index(category)
        path=os.path.join(TEST_DATASET, category)
        for img_file in os.listdir(path):
            img=cv.imread(os.path.join(path, img_file), 1)
            img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            img=cv.resize(img,(64, 64))
            test_data.append([img, label])

import random
random.shuffle(train_data)
random.shuffle(test_data)
    
X_train=[]
y_train=[]

for features,label in train_data:
    X_train.append(features)
    y_train.append(label)

Y=[]
for i in y_train:
    if i==0:
        Y.append("Parasitized")
    else:
        Y.append("Uninfected")
    
X_test=[]
y_test=[]

for features,label in test_data:
    X_test.append(features)
    y_test.append(label)
    
Z=[]
for i in y_test:
    if i==0:
        Z.append("Parasitized")
    else:
        Z.append("Uninfected")

X_train=np.array(X_train).reshape(-1,64,64,1)
X_train=X_train/255.0
X_train.shape

X_test=np.array(X_test).reshape(-1,64,64,1)
X_test=X_test/255.0
X_test.shape

order=['Parasitized', 'Uninfected']

import seaborn as sns

ax=sns.countplot(Y, order=order)
ax.set_xlabel("Status")
ax.set_ylabel("Image Count")

ax2=sns.countplot(Z, order=order)
ax2.set_xlabel("Status")
ax2.set_ylabel("Image Count")

from keras.utils import to_categorical

one_hot_train=to_categorical(y_train)
one_hot_train = np.delete(one_hot_train,0,1)

one_hot_test=to_categorical(y_test)
one_hot_test = np.delete(one_hot_test,0,1)

from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout

classifier=Sequential()

classifier.add(Conv2D(32, (3,3), input_shape=(64,64,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))

classifier.add(Conv2D(64, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))

classifier.add(Conv2D(128, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.4))

classifier.add(Flatten())

classifier.add(Dense(activation='relu', units=64))
classifier.add(Dense(activation='relu', units=128))
classifier.add(Dense(activation='relu', units=64))
classifier.add(Dense(activation='sigmoid', units=1))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.summary()

hist=classifier.fit(X_train, one_hot_train, validation_split=0.2, epochs=20, batch_size=64)

test_loss,test_acc=classifier.evaluate(X_test, one_hot_test)

mp.plot(hist.history['accuracy'])
mp.plot(hist.history['val_accuracy'])
mp.ylabel('Accuracy')
mp.xlabel('Epoch')
mp.title('Classifier Accuracy')
mp.legend(['Train','Validation'],loc='upper left')
mp.show()

mp.plot(hist.history['loss'])
mp.plot(hist.history['val_loss'])
mp.ylabel('Loss')
mp.xlabel('Epoch')
mp.title('Classifier Loss')
mp.legend(['Train','Validation'],loc='upper right')
mp.show()

y_pred=classifier.predict_classes(X_test)
y_pred

y_prob=classifier.predict_proba(X_test)
y_prob

from sklearn.metrics import roc_curve, auc
  
fpr, tpr, thresh = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
      
mp.plot(fpr, tpr, color='orange', label='ROC AUC Score = %0.3f' % roc_auc)
mp.title('Malaria ROC curve')
mp.xlabel('False Positive Rate')
mp.ylabel('True Positive rate')
mp.legend(loc='best')

from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)

classifier.save('malaria.h5')