from datagen import get_data_generator
from TcnnWithFC import TcnnWithFC
from TcnnWithoutFC import TcnnWithoutFC

from tensorflow.keras import applications

if __name__ == "__main__":
    # create model
    base_model = applications.VGG19()
    tcnn = TcnnWithFC(base_model) # or TcnnWithoutFC (refer to TcnnWithoutFC.py for more information)
    # model.unfreeze_layers(0) # hyperparameter to be tuned (refer to Tcnn.py for more information)
    tcnn.model.compile(optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy', 'acc'])

    # get data generators
    [train_generator, validation_generator] = get_data_generator()

    # train
    tcnn.model.fit(
        train_generator,
        steps_per_epoch=0,
        epochs=0,
        validation_data=validation_generator,
        validation_steps=0)
