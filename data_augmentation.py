# Helper functions for image data augmentation to stop / prevent overfitting the neural net onto dataset.

import numpy as np
import cv2
import matplotlib.pyplot as plt

# crop image by given top left point coordinate and image width and height
def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

# random crop to the original size
def random_crop(x, r_pix=16):
    r, c, *_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = np.random.uniform(0,1)
    rand_c = np.random.uniform(0,1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

# crop image from center by the given number of row pixels
def center_crop(x, r_pix=16):
    r, c, *_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)

# random crop to both image and its respective bbox mask
def random_cropXY(x, Y, r_pix=16):
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = np.random.uniform(0,1)
    rand_c = np.random.uniform(0,1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    img_crop = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    mask_crop = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return img_crop, mask_crop

# transform input image and its bbox array to resized image and respective bbox array

def transformsXY(path, bb, transforms, sz=480):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    Y = create_mask(bb, x)
    if transforms:
        #rdeg = (np.random.random()-.50)*20
        #x = rotate_cv(x, rdeg)
        #Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5: 
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
        x, Y = resize_image_bb_for_augmentation(x, mask_to_bb(Y), sz)
    else:
        x, Y = resize_image_bb_for_augmentation(x, mask_to_bb(Y), sz)
    return x, Y

# create rectangle object representing a bbox
def create_bbox(bb, colour='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[0], bb[1]), bb[2], bb[3], color=colour, fill=False, lw=3)

# add bbox to figure
def draw_bbox(im, bb, colour):
    plt.imshow(im)
    plt.gca().add_patch(create_bbox(bb, colour))

# create black and white mask from bbox array format
def create_mask(bb, x):
    rows, cols, *_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(np.int)
    Y[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]] = 1.
    return Y

# convert black (0) and white (1) mask to bbox array format
def mask_to_bb(Y):
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([top_row, left_col, right_col-left_col, bottom_row-top_row], dtype=np.float32)

# generate bbox array using bbox top left point x,y coordinates and bbox width and height
def create_bb_array(x):
    return np.array([x[1],x[2],x[4],x[3]])

# return resized image and its resized bbox array
def resize_image_bb_for_augmentation(im, bb, sz):
    im_resized = cv2.resize(im, (int(round(1.33*sz,-1)), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(round(1.33*sz,-1)), sz))
    return im_resized, mask_to_bb(Y_resized)

# resize image and its bbox and save image to new path
def resize_image_bb(read_path,write_path,bb,sz):
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(round(1.33*sz,-1)), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(round(1.33*sz,-1)), sz))
    new_path = str(write_path/read_path.parts[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)

# read image using opencv module and convert to RGB as it reads images in BGR format
def read_image(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)