from abc import ABC, abstractmethod

class Tcnn(ABC):
    NUM_CLASSES = 9

    def __init__(self, base_model):
        self.base_model = base_model
        self.model = self.build_model()
    
    @abstractmethod
    def build_model(self):
        """
        Build transfer learning model using base_model.
        """
        return 's'
    
    """ 
    To unfreeze last n layers of base model
    """
    def unfreeze_layers(self, n):
        num_layers = len(self.model.layers)
        for i in range(n): 
            curr_index = num_layers - 1 - i # add 2 to account for the added layers
            self.model.layers[curr_index].trainable = True
            