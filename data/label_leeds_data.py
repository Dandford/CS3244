""" 
To be used with Leeds butterfly dataset.
"""
import os, random, shutil
import numpy as np

NUM_CATEGORIES = 10
CATEGORY_NAMES = ['', 'danaus_plexippus', 'heliconius_charitonius', 'heliconius_erato', 'junonia_coenia', 'lycaena_phlaeas',
                    'nymphalis_antiopa', 'papilio_cresphontes', 'pieris_rapae', 'vanessa_atalanta', 'vanessa_cardui']

TRAIN_PATH = "train"
VALIDATION_PATH = "validation"

# to determine fraction of images that are used for training
THRESHOLD = 0.8

# used to determine image's label. Note that 0th index is a 'dummy'
train_label = np.zeros(NUM_CATEGORIES + 1)
validation_label = np.zeros(NUM_CATEGORIES + 1)

# max images allowed per category (to ensure no. of images is almost the same for all categories)
MAX = 80

def label():
    dataset_path = "../leedsbutterfly/images"
    try:
        # ensure that dataset exists
        if os.path.isdir(dataset_path) == False:
            raise FileNotFoundError("Please download Leeds butterfly dataset and place it in cnnModel dir")

        # create main dir to place train and validation data
        create_main_dir()
        # create subdirectories
        for name in CATEGORY_NAMES:
            create_subdir(name)

        # get all images
        images = os.scandir(dataset_path)

        # for each image, copy and place image in its respective train and validation dir
        for img in images:
            filename, ext = os.path.splitext(img)
            if not ext == '.png':
                continue
            label_data(img)

    except FileNotFoundError as e:
        print(e)

def create_main_dir():
    # clears dirs and re-create, if they exist
    if os.path.isdir(TRAIN_PATH):
        os.rmdir(TRAIN_PATH)
    if os.path.isdir(VALIDATION_PATH):
        os.rmdir(VALIDATION_PATH)
    
    os.makedirs(TRAIN_PATH)
    os.makedirs(VALIDATION_PATH)

def create_subdir(name):
    train_subdir_label = TRAIN_PATH + '/' + name
    validation_subdir_label = VALIDATION_PATH + '/' + name
    
    os.makedirs(train_subdir_label)
    os.makedirs(validation_subdir_label)

def label_data(img):
    category = get_category(img)
    num_img_in_category = train_label[category] + validation_label[category]

    if num_img_in_category > MAX:
        # skip img to ensure almost uniform dataset
        return

    name = CATEGORY_NAMES[category]
    # randomly decide whether image is to be used for train or validation
    is_train = random.random() < THRESHOLD
    # get file extension
    filename, ext = os.path.splitext(img)

    if is_train:
        label = train_label[category] + 1
        dest_path = TRAIN_PATH + '/' + name + '/' + str(label) + ext
        shutil.copy(img, dest_path)
        # update label index
        train_label[category] += 1
    else:
        label = validation_label[category] + 1
        dest_path = VALIDATION_PATH + '/' + name + '/' + str(label) + ext
        shutil.copy(img, dest_path)
        # ucpate label index
        validation_label[category] += 1

def get_category(img):
    category = img.name[0:3]
    return int(category)

if __name__ == "__main__":
    label()