from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

IMG_SIZE = 224
BATCH_SIZE = 32

""" 
Creates data generator for test and validation. 
Data is taken from ../data/train and ../data/validation respectively.
Note that the labels are based on the names of directories found in ../data/train and 
../data/validation.
"""
def get_data_generator():
    train_datagen = image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        # rescale=1./255, # to be left out if it doesn't work well with vgg19
        fill_mode='nearest')
    
    test_datagen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        '../data/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)

    validation_generator = test_datagen.flow_from_directory(
        '../data/validation',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)
    
    return [train_generator, validation_generator]
    