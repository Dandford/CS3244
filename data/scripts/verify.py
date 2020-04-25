""" 
Verifies that the total number of images for each category remains the same
after they are separated into train, test and validation folders. 
In addition, it verifies the train:validation ratio.
"""

import os

NUM_CLASS = 9
MAIN_IMG_PATH = '../butterflies/'
TEST_ING_PATH = 'test/'
TRAIN_IMG_PATH = 'train/'
VALIDATION_IMG_PATH = 'validation/'

def get_num(main_path):
    temp_arr = os.listdir(main_path)
    temp_arr = sorted(os.listdir(main_path))
    dirs = []

    for element in temp_arr:
        path = main_path + element
        if not os.path.isdir(path):
            continue
        folder = os.listdir(path)
        dirs.append(len(folder))
    
    return dirs

def is_sum_correct(expected_arr, arr_1, arr_2, arr_3):
    actual_arr = []

    for i in range(NUM_CLASS):
        curr_sum = arr_1[i] + arr_2[i] + arr_3[i]
        actual_arr.append(curr_sum)
    return [expected_arr == actual_arr, actual_arr]

def verify():
    expected_num_images = get_num(MAIN_IMG_PATH)
    test_num_images = get_num(TEST_ING_PATH)
    train_num_images = get_num(TRAIN_IMG_PATH)
    validation_num_images = get_num(VALIDATION_IMG_PATH)
    
    [is_correct, actual_arr] = is_sum_correct(expected_num_images, 
        test_num_images, 
        train_num_images, 
        validation_num_images)
    
    if not is_correct:
        print('Sum does not tally!')
        print('Expected:')
        print(expected_num_images)
        print('Actual:')
        print(actual_arr)

    ratio_arr = []

    for i in range(NUM_CLASS):
        total = train_num_images[i] + validation_num_images[i]
        ratio_arr.append(train_num_images[i] / total)
    
    print(ratio_arr)


if __name__ == "__main__":
    verify()