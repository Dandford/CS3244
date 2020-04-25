"""
To change the number of categories, modify NUM_CLASSES in Tcnn.py
"""

import matplotlib.pyplot as plt

from datagen import get_data_generator
from tcnn.TcnnWithFC import TcnnWithFC
from tcnn.TcnnWithoutFC import TcnnWithoutFC
from tensorflow.keras import applications
from tensorflow.keras.models import load_model

# save model to FILENAME
FILENAME = 'butterfly_classification.h5'

def save_statistics(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig('test.jpg')

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    # create model
    # base_model = applications.VGG19()
    base_model = applications.mobilenet_v2.MobileNetV2() 
    tcnn = TcnnWithFC(base_model) # or TcnnWithoutFC (refer to TcnnWithoutFC.py for more information)
    # model.unfreeze_layers(0) # hyperparameter to be tuned (refer to Tcnn.py for more information)
    tcnn.model.compile(optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy', 'acc'])

    # get data generators
    [train_generator, validation_generator] = get_data_generator()

    # train and save history
    history = tcnn.model.fit(
        train_generator,
        steps_per_epoch=4,
        epochs=60, 
        validation_data=validation_generator,
        validation_steps=2) 

    # save training statistics
    save_statistics(history)

    # to save the model
    tcnn.model.save(FILENAME)