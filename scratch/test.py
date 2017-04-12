'''

'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import os.path
import numpy
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 56, 56

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

f_log = './log'
#model saving location
f_model = './model'
model_filename = 'cnn_model.json'
# weights_filename = 'cnn_model_weights.hdf5' 
weights_filename = 'weights.hdf5'

train_sample_number = 32
test_sample_number = 26

nb_train_samples = train_sample_number*30
nb_validation_samples = test_sample_number*20
train_batch = train_sample_number
test_batch = test_sample_number
nb_epoch = 100


#training model
model = Sequential()

model.add(Convolution2D(32, 3, 3,border_mode='valid', input_shape=(img_width, img_height, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=5e-3, decay=1e-6, momentum=0.9, nesterov=True)
    
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    zca_whitening=False,
    zoom_range=0.2,
    rotation_range=20, 
    width_shift_range=0.3,
	height_shift_range=0.3,
	fill_mode='nearest',
	vertical_flip=False,
    horizontal_flip=True
)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(

    zoom_range=0.2,
    rotation_range=20, 
    vertical_flip=False,
    horizontal_flip=True
    )


train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=train_batch,

	# save_to_dir="saved",   
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=test_batch,
        class_mode='binary')

checkpointer = ModelCheckpoint(filepath="./saved/weights.hdf5", verbose=1, save_best_only=True)

history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=[checkpointer])


print('save the architecture of a model')
json_string = model.to_json()
open(os.path.join(f_model,'cnn_model.json'), 'w').write(json_string)
yaml_string = model.to_yaml()
open(os.path.join(f_model,'cnn_model.yaml'), 'w').write(yaml_string)
print('save weights')
model.save_weights(os.path.join(f_model,'cnn_model_weights.hdf5'))

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

