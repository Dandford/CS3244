# ButterSnap: Butterfly Species Identification with CNN

<!-- GETTING STARTED -->
## Getting Started 

In this section, you will find out how you can train and/or test the models used for *ButterSnap*.

### Predicting Butterfly Species

1. Run `test.py`, and pass the path of the image `<image>` as an argument.
```sh
python3 test.py <image>
```

### Testing the CNN Model with Default Images

1. Run the script `../data/label_data.py` to obtain default test images.
```sh
python3 ../data/label_data.py
```

2. Next, run `test.py`.
```sh
python3 test.py
```

### Testing the CNN Model with Customised Images

1. Place your test images in appropriately-labelled sub-folders. Each of the sub-folder should contain images of
butterflies of a specific species. For instance, 
 ```
    >> train    >> chocolate_pansy_butterfly  >> img_1.jpg 
                                              >> img_2.jpg
                >> common_mormon_(male)       >> img_1.jpg  
                >> common_palmfly             >> img_1.jpg  
                >> little_tiger               >> img_1.jpg  
                >> painted_jezebel            >> img_1.jpg  
                                              >> img_2.jpg  
```

2. Then, place these sub-folders in a main folder at `<folder>`.

3. Now, run `test.py` and pass the directory `<folder>` as an argument.
```sh
python3 test.py <folder>
```