from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

NUM_CLASSES = 3

""" 
Uses base_model for transfer learning.
Builds model without removal of hidden layers. Only output layer is removed.
Two additional adaptation layers are added. 
""" 
def build_model(base_model):
    # freeze layers
    base_model.trainable = False
    # remove output layer
    feature_list = [layer.output for layer in base_model.layers[:-1]]
    # add adaptation layers
    feature_fc = feature_list[len(feature_list) - 1]
    fc = Dense(2048, activation='relu')(feature_fc)
    output = Dense(NUM_CLASSES, activation='softmax')(fc)
    # create model
    model = Model(inputs=base_model.input, outputs=output)
    # model.summary() # to view model summary
    return model

""" 
To unfreeze last n layers
"""
def unfreeze_layers(model, n):
    num_layers = len(model.layers)
    for i in range(n):
        curr_index = num_layers - n + i
        model.layers[curr_index].trainable = True

if __name__ == "__main__":
    vgg19 = applications.VGG19()
    model = build_model(vgg19)
    
