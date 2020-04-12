import sys
import numpy as np

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image

# indicates where model is saved to (should correspond to that in train.py)
FILENAME = 'butterfly_classification.h5'
# default image size to use
IMG_SIZE = 224

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def predict_butterfly(img_path):
    img = preprocess_image(img_path)
    model = load_model(FILENAME)
    result = model.predict(img)
    print(result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Incorrect number of arguments given! Please input path of image.')
    else:
        img_path = sys.argv[1]
        predict_butterfly(img_path)
