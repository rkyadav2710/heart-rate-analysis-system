import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, AveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from keras.layers import Input, multiply
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.model_selection import train_test_split

train = pd.read_csv('train_new.csv')
test = pd.read_csv('test_new.csv')

train_image = []
train_reading = []
test_image = []
test_reading = []

Train_count = 0

for i in range(train.shape[0]):
    img = image.load_img(train['image'][i], target_size=(50, 50, 3))
    img = image.img_to_array(img)
    train_image.append(img)
    Train_count = Train_count + 1
    print(Train_count)
  
X_train = np.array(train_image)
X_train.shape
y_train = train['reading']

print(train.shape[0])
print(test.shape[0])



Test_count = 0
for i in range(test.shape[0]):
	img = image.load_img(test['image'][i], target_size=(50, 50, 3))
	img = image.img_to_array(img)
	test_image.append(img)
	Test_count = Test_count + 1
	print(Test_count)

X_test = np.array(test_image)
y_test = test['reading']

print(X_test.shape)
print(y_test.shape)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=33)
datagen = ImageDataGenerator(width_shift_range = 0.1,
                            height_shift_range = 0.1,
                            zoom_range = 0.3,
                            shear_range = 0.1,
                            rotation_range = 15.)

datagen.fit(X_train)

batches = datagen.flow(X_train, y_train, batch_size = 20)
X_batch, y_batch = next(batches)




def Convolutional_model():
    #model.Sequential()
    inputA = Input(shape=(50, 50, 3))
    inputB = Input(shape=(50, 50, 3))
    
    x = Conv2D(150, (5, 5), activation='relu')(inputA)
    x = Conv2D(120, (3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Conv2D(120, (3, 3), activation='relu')(x)
    x = Conv2D(90, (3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dense(1000, activation='relu')(x)   
    x = Model(inputs=inputA, outputs=x)

    y = Conv2D(150, (5, 5), activation='relu')(inputB)
    y = Conv2D(120, (3, 3), activation='relu')(y)
    y = AveragePooling2D(pool_size=(2, 2))(y)
    y = Conv2D(120, (3, 3), activation='relu')(y)
    y = Conv2D(90, (3, 3), activation='relu')(y)
    y = AveragePooling2D(pool_size=(2, 2))(y)
    y = Dense(1000, activation='relu')(y)
    y = Model(inputs=inputB, outputs=y)
    
    combined = multiply([x, y])([x, y])
    z = Flatten()(combined)
    z = Dense(512, activation='relu')(z)
    z = Dropout(0.3)(z)
    z = Dense(32, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(1, activation='linear')(z)
    model = Model(inputs=[x.input, y.input], outputs=z)
    
    return model
  
MODEL = Convolutional_model()
print(MODEL)

