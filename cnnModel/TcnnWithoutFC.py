from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

from Tcnn import Tcnn

""" 
Uses base_model for transfer learning.
Builds model with removal of fully-connected layers. 
Two additional adaptation layers are added. 
""" 
class TcnnWithoutFC(Tcnn):
    def __init__(self, base_model):
        super().__init__(base_model)

    def build_model(self):
        # freeze layers
        self.base_model.trainable = False
        # obtain layers
        feature_list = [layer.output for layer in self.base_model.layers[:-3]]
        # add adaptation layers
        feature_fc = feature_list[len(feature_list) - 1]
        fc = Dense(2048, activation='relu')(feature_fc)
        output = Dense(Tcnn.NUM_CLASSES, activation='softmax')(fc)
        # create model
        model = Model(inputs=self.base_model.input, outputs=output)
        # model.summary() # to view model summary
        return model
