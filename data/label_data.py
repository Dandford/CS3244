""" 
To be used with dataset collected by @leongjwm and @huaren06
Before running this script, please place directories (not nested directories) which contain the images in
../butterflies.
"""
import os, random, shutil

TRAIN_PATH = "train"
VALIDATION_PATH = "validation"
TEST_PATH = "test"

# to determine fraction of images that are used for training
THRESHOLD = 0.8

# max images allowed per category (to ensure no. of images is almost the same for all categories)
MAX = 200

def label():
    dataset_path = "../butterflies"
    try:
        # ensure that dataset exists
        if os.path.isdir(dataset_path) == False:
            raise FileNotFoundError("Please place subdirectories of images in ../butterflies")

        # create main dir to place train and validation data
        create_main_dir()

        # for each image folder, copy and place image in its respective train and validation dir
        image_dirs = os.scandir(dataset_path)

        for image_dir in image_dirs:
            if not os.path.isdir(image_dir):
                # not a directory with images, can skip
                continue
            label_data(image_dir)

    except FileNotFoundError as e:
        print(e)

def create_main_dir():
    # clears dirs and re-create, if they exist
    if os.path.isdir(TRAIN_PATH):
        shutil.rmtree(TRAIN_PATH)
    if os.path.isdir(VALIDATION_PATH):
        shutil.rmtree(VALIDATION_PATH)
    if os.path.isdir(TEST_PATH):
        shutil.rmtree(TEST_PATH)
    
    os.makedirs(TRAIN_PATH)
    os.makedirs(VALIDATION_PATH)
    os.makedirs(TEST_PATH)

def label_data(image_dir):
    dir_name = create_subdir(image_dir)
    # obtain images
    images = os.scandir(image_dir)

    # used for labelling of image
    train_label = 1
    validation_label = 1
    test_label = 1
    total_num = 0

    for img in images:
        # determine whether maximum number of images have been exceeded
        if total_num > MAX:
            # add image to test directory instead
            dest_path = TEST_PATH + '/' + dir_name + '/' + str(test_label) + ext
            shutil.copy(img, dest_path)
            test_label += 1
            continue

        # randomly decide whether image is to be used for train or validation
        is_train = random.random() < THRESHOLD
        # get file extension
        filename, ext = os.path.splitext(img)

        if is_train:
            dest_path = TRAIN_PATH + '/' + dir_name + '/' + str(train_label) + ext
            shutil.copy(img, dest_path)
            train_label += 1
        else:
            dest_path = VALIDATION_PATH + '/' + dir_name + '/' + str(validation_label) + ext
            shutil.copy(img, dest_path)
            validation_label += 1
        
        total_num += 1

def create_subdir(image_dir):
    name = image_dir.name.lower().replace(' ', '_')
    train_subdir_label = TRAIN_PATH + '/' + name
    validation_subdir_label = VALIDATION_PATH + '/' + name
    test_subdir_label = TEST_PATH + '/' + name

    os.makedirs(train_subdir_label)
    os.makedirs(validation_subdir_label)
    os.makedirs(test_subdir_label)

    return name

if __name__ == "__main__":
    label()