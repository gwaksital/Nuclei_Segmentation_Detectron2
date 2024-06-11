import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io as sio
import time
import yaml
import random
import json
import shutil

from skimage import io, color
from fitter import Fitter
from tqdm import tqdm
from randstainna import RandStainNA
from sklearn.model_selection import train_test_split


''' 1. Extracting binary masks from .mat files '''
def make_binary_mask(matPath):
    mask = sio.loadmat(matPath)

    inst_map = mask['inst_map']
    nuclear_map = inst_map + 0
    nuclear_map[nuclear_map != 0] = 1

    return nuclear_map


def make_binary_masks(matDir, binDir):
    if not os.path.exists(binDir):
        os.makedirs(binDir)

    matFiles = glob.glob(matDir + '/*.mat')
    for matFile in tqdm(matFiles):
        mask = make_binary_mask(matFile)
        mask = mask.astype(np.uint8)
        mask = mask * 255
        mask = np.stack((mask,)*3, axis=-1)
        mask = mask.astype(np.uint8)
        maskName = os.path.basename(matFile).replace('.mat', '.png')
        maskPath = os.path.join(binDir, maskName)
        io.imsave(maskPath, mask)






''' 2. Image Size Preprocessing '''
# fill zeros around the image
def zero_padding(img, patchSize):
    h, w = img.shape[:2]
    
    if h < patchSize:    
        top_pad = (patchSize-h) // 2
        bottom_pad = patchSize - h - top_pad
        img = np.pad(img, ((top_pad, bottom_pad), (0, 0), (0, 0)), mode='constant', constant_values=0)        

    if w < patchSize:
        left_pad = (patchSize-w) // 2
        right_pad = patchSize - w - left_pad
        img = np.pad(img, ((0, 0), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0)

    return img


# convert image into 512x512 "1 center patch"
#   small image -> fill zeros around the image
#   large image -> cut the image into 512x512 patches and return the center patch
def get_1_patch(img, patchSize):
    zero_padded_img = zero_padding(img, patchSize)
    h, w = zero_padded_img.shape[:2]
    center_h, center_w = h//2, w//2
    half_size = patchSize // 2
    return zero_padded_img[center_h-half_size:center_h-half_size+patchSize, center_w-half_size:center_w-half_size+patchSize]


# convert image into 512x512 "# of patches"
#   small image -> fill zeros around the image
#   large image -> cut the image into 512x512 patches
def get_n_patches(img, patchSize):
    patches = []
    h, w = img.shape[:2]
    for i in range(0, h, patchSize):
        for j in range(0, w, patchSize):            
            if i + patchSize > h or j + patchSize > w:
                patch = zero_padding(img[i:i+patchSize, j:j+patchSize], patchSize)
            else:
                patch = img[i:i+patchSize, j:j+patchSize]
            patches.append(patch)
    return patches


def img_size_preprocess(imgDir, patchSize, saveDir):
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    imgFiles = glob.glob(imgDir + '/*.png')
    imgFiles.extend(glob.glob(imgDir + '/*.jpg'))
    
    for imgFile in tqdm(imgFiles):
        img = io.imread(imgFile)

        ''' This is for "1 center patch '''
        # patch = get_1_patch(img, patchSize)
        # patchPath = os.path.join(saveDir, os.path.basename(imgFile))
        # io.imsave(patchPath, patch)
        ''' end '''   

        ''' This is for "# of patches" '''
        patches = get_n_patches(img, patchSize)
        for i, patch in enumerate(patches):
            # if file is .jpg, save as .jpg
            if imgFile.endswith('.jpg'):
                patchName = os.path.basename(imgFile).replace('.jpg', '_') + str(i) + '.jpg'
            else:
                patchName = os.path.basename(imgFile).replace('.png', '_') + str(i) + '.png'

            patchPath = os.path.join(saveDir, patchName)
            io.imsave(patchPath, patch)
        ''' end '''



''' 3. Convert label images to COCO json format '''
def get_image_mask_pairs(data_dir, original, labeled):
    image_paths = []
    mask_paths = []
    
    for root, _, files in tqdm(os.walk(data_dir), desc='get_image_mask_pairs:'):
        if original in root:
            for file in files:
                if file.endswith('.png'):
                    image_paths.append(os.path.join(root, file))
                    mask_paths.append(os.path.join(root.replace(original, labeled), file))
    
    return image_paths, mask_paths


def mask_to_polygons(mask, epsilon=1.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) > 2:
            poly = contour.reshape(-1).tolist()
            if len(poly) > 4:  # Ensure valid polygon
                polygons.append(poly)
    return polygons


def process_data(image_paths, mask_paths, output_dir):
    annotations = []
    images = []
    image_id = 0
    ann_id = 0
        
    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), desc='process_data:'):
        image_id += 1
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        
        # Copy image to output directory
        shutil.copy(img_path, os.path.join(output_dir, os.path.basename(img_path)))
        
        images.append({
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "height": img.shape[0],
            "width": img.shape[1]
        })
        
        unique_values = np.unique(mask)
        
        
        for value in unique_values:
            if value == 0:  # Ignore background
                continue
            
            
            object_mask = (mask == value).astype(np.uint8) * 255
                        
            polygons = mask_to_polygons(object_mask)
            
            for poly in polygons:
                ann_id += 1
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,  # Only one category: Nuclei
                    "segmentation": [poly],
                    "area": cv2.contourArea(np.array(poly).reshape(-1, 2)),
                    "bbox": list(cv2.boundingRect(np.array(poly).reshape(-1, 2))),
                    "iscrowd": 0
                })
    
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "Nuclei"}]
    }
    
    with open(os.path.join(output_dir, 'coco_annotations.json'), 'w') as f:
        json.dump(coco_output, f)



    



''' main process '''
def main():
    
    ''' 1. Make binary masks from .mat files '''
    
    matDir = 'D:\hjkwak\PROJ_Nuclei_Segmentation\Lizard_Dataset\lizard_labels\Lizard_Labels\Labels'
    binDir = 'D:\hjkwak\PROJ_Nuclei_Segmentation\Lizard_Dataset\Data\Lizard_labeled'
    if not os.path.exists(binDir):
        os.makedirs(binDir)

    make_binary_masks(matDir, binDir)
    print('1. Make binary masks from .mat files is done!')
    print('Number of binary masks:', len(glob.glob(binDir + '/*.png')))



    ''' 2. Image Size Preprocessing '''
    noises = []
    patchSize = 512

    imgDir = 'D:\hjkwak\PROJ_Nuclei_Segmentation\Lizard_Dataset\Data\Lizard_original'
    saveimgDir = 'D:\hjkwak\PROJ_Nuclei_Segmentation\Lizard_Dataset\Data\Lizard_original_patches'


    maskDir = 'D:\hjkwak\PROJ_Nuclei_Segmentation\Lizard_Dataset\Data\Lizard_labeled'
    savemaskDir = 'D:\hjkwak\PROJ_Nuclei_Segmentation\Lizard_Dataset\Data\Lizard_labeled_patches'


    overlayDir = 'D:\hjkwak\PROJ_Nuclei_Segmentation\Lizard_Dataset\overlay\Overlay'
    saveOverlayDir = 'D:\hjkwak\PROJ_Nuclei_Segmentation\Lizard_Dataset\overlay\Overlay_patches'

    img_size_preprocess(imgDir, patchSize, saveimgDir)
    img_size_preprocess(maskDir, patchSize, savemaskDir)
    img_size_preprocess(overlayDir, patchSize, saveOverlayDir)


    # remove outliers
    imgFiles = glob.glob(saveimgDir + '/*.png')
    imgFiles.extend(glob.glob(saveimgDir + '/*.jpg'))

    print('2-1. Image Size Preprocessing is done!')
    print('Number of init patches:', len(imgFiles))

    for imgFile in imgFiles:
        img = io.imread(imgFile)

        if np.mean(img[img != 0], axis=0) > 234:
            noises.append(os.path.basename(imgFile))
            continue
        
        black_ratio = np.sum(img == 0) / img.size
        if black_ratio > 0.1:
            noises.append(os.path.basename(imgFile))
    
    for noise in tqdm(noises):
        os.remove(os.path.join(saveimgDir, noise))
        os.remove(os.path.join(savemaskDir, noise))
        if imgFile.endswith('.png'):
            noise = noise.replace('.png', '.jpg')            
        os.remove(os.path.join(saveOverlayDir, noise))

    print('2-2. Remove outliers is done!')
    print('Number of noises:', len(noises))
    print('Number of final patches:', len(glob.glob(saveimgDir + '/*.png')))




    ''' 3. Convert label images to COCO json format '''
    
    data_dir = r'D:\hjkwak\PROJ_Nuclei_Segmentation\Lizard_Dataset\Data'
    original_name = saveimgDir.split('\\')[-1]
    labeled_name = savemaskDir.split('\\')[-1]

    output_dir = r'D:\hjkwak\PROJ_Nuclei_Segmentation\Lizard_Dataset\COCO_Dataset'
    os.makedirs(output_dir, exist_ok=True)


    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    image_paths, mask_paths = get_image_mask_pairs(data_dir, original_name, labeled_name)
    
    ''' Split data into train and val '''
    # train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

    ''' Split data into train, val, and test '''
    train_img_paths, rest_img_paths, train_mask_paths, rest_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.3, random_state=42)
    val_img_paths, test_img_paths, val_mask_paths, test_mask_paths = train_test_split(rest_img_paths, rest_mask_paths, test_size=0.33, random_state=42)

    # Process train and val data
    process_data(train_img_paths, train_mask_paths, train_dir)
    process_data(val_img_paths, val_mask_paths, val_dir)
    process_data(test_img_paths, test_mask_paths, test_dir)

    print('3. Convert label images to COCO json format is done!')
    print('Number of train images:', len(train_img_paths))
    print('Number of val images:', len(val_img_paths))
    print('Number of test images:', len(test_img_paths))






if __name__ == '__main__':
    main()