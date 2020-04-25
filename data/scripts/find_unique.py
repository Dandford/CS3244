import os, shutil

SOURCE_PATH = '../butterflies/Grass Blue Butterfly/'
DESTINATION_PATH = '../../../Downloads/cropped_vs_uncropped-2/cropped/grass_blue_butterfly/'
NEW_PATH = 'to_crop/'

def find_unique():
    src_imgs = os.listdir(SOURCE_PATH)
    # src_dirs = sorted(src_dirs)
    dest_imgs = set(os.listdir(DESTINATION_PATH))
    # dest_dirs = sorted(dest_dirs)

    prefix = NEW_PATH

    for img in src_imgs:
        filename, ext = os.path.splitext(img)
        if not ext == '.jpg':
            # is not an image file
            continue
        if (filename + '_detected' + ext in dest_imgs) or img in dest_imgs:
            # is already in file     
            continue
        shutil.copy(SOURCE_PATH + img, prefix + img)

    """
    for (src_dir, dest_dir) in zip(src_dirs, dest_dirs):
        if not os.path.isdir(SOURCE_PATH + src_dir):
            continue

        src_images = os.listdir(src_dir)
        dest_images = set(os.listdir(dest_dir))

        prefix = NEW_PATH + src_dir + '/'

        for img in src_images:
            filename, ext = os.path.splitext(img)
            if not ext == '.jpg':
                # is not an image file
                continue
            if img in dest_images:
                # is already in file
                continue
            shutil.copy(img, prefix + img)
    """

if __name__ == "__main__":
    find_unique()