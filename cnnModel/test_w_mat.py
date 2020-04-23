""" 
Tests the model with 10 images from each category and generates a confusion 
matrix.
"""

import sys, os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

CATEGORIES = ['Chocolate Pansy', 'Common Mormon', 'Common Palmfly', 'Grass Blue', 'Grass Yellow',
    'Lime', 'Little Tiger', 'Painted Jezebel', 'Tawny Coster']
TEST_DIR = '../data/test_small/'

IMG_SIZE = 224
FILENAME = 'models/butterfly_classification_3.h5'

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
    return result

def test():
    dirs = [file for file in os.listdir(TEST_DIR) if not file.startswith('.')]
    dirs = sorted(dirs)

    actual_labels = []
    predicted_labels = []

    for i, butterfly in enumerate(dirs, 0):
        prefix = TEST_DIR + butterfly + '/'
        images = os.listdir(prefix)
        num_images = 0
        
        for img in images:
            if img.startswith('.'):
                continue
            img = prefix + img
            result = predict_butterfly(img)
            prediction = np.argmax(result)
            predicted_labels.append(prediction)
            num_images += 1
        
        # update matrix containing actual labels
        actual_labels = actual_labels + [i for j in range(num_images)]

    # create confusion matrix
    data = {'y_Actual': [CATEGORIES[i] for i in actual_labels], 
        'y_Predicted': [CATEGORIES[i] for i in predicted_labels]}
    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], 
        rownames=['Actual'], 
        colnames=['Predicted'])
    sn.heatmap(confusion_matrix)
    plt.show()

if __name__ == "__main__":
    test()