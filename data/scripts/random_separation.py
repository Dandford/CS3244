""" Randomly assign images into validation/test/train """

import random, sys, os, shutil

TEST_PATH = 'test/'
TRAIN_PATH = 'train/'
VALIDATION_PATH = 'validation/'

def random_separation(source_path):
    src_dirs = sorted(os.listdir(source_path))
    src_subdirs = [source_path + subdir + '/' for subdir in src_dirs if not subdir.startswith('.')]
    if len(src_subdirs) != 2:
        print('Incorrect number of files. Expected 2 files but got ' + str(len(src_subdirs)) + ' files.')
        return
    src_subdir_1 = [src_subdirs[0] + imgs + '/' for imgs in os.listdir(src_subdirs[0])]
    src_subdir_1 = sorted(src_subdir_1)
    src_subdir_2 = [src_subdirs[1] + imgs + '/' for imgs in os.listdir(src_subdirs[1])]
    src_subdir_2 = sorted(src_subdir_2)

    make_dirs('cropped/' + TEST_PATH, 
        'cropped/' + TRAIN_PATH, 
        'cropped/' + VALIDATION_PATH)
    make_dirs('uncropped/' + TEST_PATH, 
        'uncropped/' + TRAIN_PATH, 
        'uncropped/' + VALIDATION_PATH)

    for (images_1, images_2) in zip(src_subdir_1, src_subdir_2):
        if not os.path.isdir(images_1):
            continue
        cat_1 = images_1.split('/')[-2]
        cat_2 = images_2.split('/')[-2]
        if cat_1 != cat_2:
            print('Expected both categories to be identical but got \'' + cat_1 + '\' and \'' + cat_2 + '\'.')
            break
        cat = cat_1 # since both are identical

        prefix_test = TEST_PATH + cat
        prefix_train = TRAIN_PATH + cat
        prefix_validation = VALIDATION_PATH + cat
        make_dirs('cropped/' + prefix_test, 
            'cropped/' + prefix_train, 
            'cropped/' + prefix_validation)
        make_dirs('uncropped/' + prefix_test, 
            'uncropped/' + prefix_train, 
            'uncropped/' + prefix_validation)

        images_1 = [images_1 + img for img in os.listdir(images_1) if img.endswith('.jpg')]
        images_1 = sorted(images_1)
        images_2 = [images_2 + img for img in os.listdir(images_2) if img.endswith('.jpg')]
        images_2 = sorted(images_2)
        if len(images_1) != len(images_2):
            print('Expected no. of images in both folders to be identical but got ' 
                + str(len(images_1)) + ' and ' + str(len(images_2)))
            break
        # use 200 images for training, 50 images for validation and the rest for test
        randomiser = [0 for img in range(200)]
        randomiser = randomiser + [1 for img in range(50)]
        randomiser = randomiser + [2 for img in range(len(images_1) - 250)]
        random.shuffle(randomiser)
    
        for i, img_cropped, img_uncropped in zip(range(len(images_2)), images_1, images_2):
            typ = randomiser[i]
            if typ == 0:
                shutil.copy(img_cropped, 'cropped/' + prefix_train)
                shutil.copy(img_uncropped, 'uncropped/' + prefix_train)
            elif typ == 1:
                shutil.copy(img_cropped, 'cropped/' + prefix_validation)
                shutil.copy(img_uncropped, 'uncropped/' + prefix_validation)
            else: 
                shutil.copy(img_cropped, 'cropped/' + prefix_test)
                shutil.copy(img_uncropped, 'uncropped/' + prefix_test) 

def make_dirs(*dirs):
    for d in dirs:
        if os.path.isdir(d):
            # delete
            shutil.rmtree(d)
        os.makedirs(d)
        

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # if main direction is specified
        source = sys.argv[1] + '/'
        random_separation(source)
    else:
        print('Please enter source paths.')