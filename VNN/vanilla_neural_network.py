import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
from math import floor
#from matplotlib import pyplot as plt

data_dir = "/Users/meiannn/Documents/NUS/Y2/S2/3244/leedsbutterfly/images"
num_of_files = len([f for f in os.listdir(data_dir)if os.path.isfile(os.path.join(data_dir, f))])
separate = floor(0.8 * num_of_files);


# track total running number of imgs
i = 0
# track running number in train
j = 0
# track running number in test
k = 0
# 1 represents train array, 2 represents test array: to randomise the images into train and test
choice = [1, 2]
probability_output = np.random.choice(choice, num_of_files, p = [0.2, 0.8])

# used for training the model
init_train_images = [0] * separate
init_train_labels = [0] * separate

# used to test the model
init_test_images = [0] * (num_of_files - separate)
init_test_labels = [0] * (num_of_files - separate)


for img in os.listdir(data_dir):
    # #img_array = keras.imread(os.path.join(data_dir, img))
    img_array = cv2.imread(os.path.join(data_dir, img))
    new_img = cv2.resize(img_array, (28, 28))
    if probability_output[i] == 1:
        if j >= separate:
            init_test_images[k] = new_img
            inti_test_labels[k] = int(img.split(".")[0][:3]) - 1
            k = k + 1
        else: 
            init_train_images[j] = new_img
            init_train_labels[j] = int(img.split(".")[0][:3]) - 1
            j = j + 1
    elif probability_output[i] == 2:
        if k >= (num_of_files - separate) :
            init_train_images[j] = new_img
            init_train_labels[j] = int(img.split(".")[0][:3]) - 1
            j = j + 1
        else:
            init_test_images[k] = new_img
            init_test_labels[k] = int(img.split(".")[0][:3]) - 1
            k = k + 1
    i = i + 1


train_images = np.array(init_train_images)
train_labels = np.array(init_train_labels)
test_images = np.array(init_test_images)
test_labels = np.array(init_test_labels)

#10 class names
class_names = ['Danaus plexippus', 'Heliconius charitonius', 'Heliconius erato', 'Junonia coenia', 'Lycaena phlaeas', 
                'Nymphalis antiopa', 'Papilio cresphontes', 'Pieris rapae', 'Vanessa atalanta', 'Vanessa cardui']

train_images = train_images/255.0
test_images = test_images/255.0

train_grayscale = tf.image.rgb_to_grayscale(train_images)
test_grayscale = tf.image.rgb_to_grayscale(test_images)

new_train_grayscale = [0] * separate
new_test_grayscale = [0] * (num_of_files - separate)

for i in range(0, len(train_grayscale)):
    new_img = [0] * 28
    for j in range(0, len(train_grayscale[0])):
        new_row = np.array(train_grayscale[i][j]).flatten()
        new_img[j] = new_row
    new_train_grayscale[i] = np.array(new_img)


for i in range(0, len(test_grayscale)):
    new_img = [0] * 28
    for j in range(0, len(test_grayscale[0])):
        new_row = np.array(test_grayscale[i][j]).flatten()
        new_img[j] = new_row
    new_test_grayscale[i] = np.array(new_img)

new_train_grayscale = np.array(new_train_grayscale)
new_test_grayscale = np.array(new_test_grayscale)

print(new_train_grayscale.shape)

model = keras.Sequential([
    #input layer
    keras.layers.Flatten(input_shape=(28, 28)), 
    #hidden layer, 128 neurons is just an arbitrary number
    #Dense = fully connected layer, relu = rectified linear unit
    keras.layers.Dense(128, activation="relu"), 
    #output layer
    #softmax: makes all the output add tgt = 1, somewhat a probability output
    #10 outputs cos each output stands for 1 classname
    keras.layers.Dense(10, activation="softmax")
 ])

#adam is standard??
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#epoch = how many times the model is going to see the same information
#because the order that the images are coming in will tweak the model differently?
#but higher epoch != higher accuracy
model.fit(new_train_grayscale, train_labels, epochs = 15)


prediction = model.predict(new_test_grayscale)
predicted_class = class_names[np.argmax(prediction[0])]
print(predicted_class)
print("Actual label, ", test_labels[0])



