import os

import numpy as np

import cv2
from albumentations import (CLAHE, Blur, Compose, Flip, GaussNoise,
                            GridDistortion, HorizontalFlip, HueSaturationValue,
                            IAAAdditiveGaussianNoise, IAAEmboss,
                            IAAPerspective, IAAPiecewiseAffine, IAASharpen,
                            MedianBlur, MotionBlur, OneOf, OpticalDistortion,
                            RandomBrightnessContrast, RandomRotate90,
                            ShiftScaleRotate, Transpose)
from tqdm import tqdm


def strong_aug(p=0.5):
    return Compose([
        HorizontalFlip(p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.8),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.7),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2,
                         rotate_limit=35, p=0.8),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.7),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.7),
        HueSaturationValue(p=0.5),
    ], p=p)


input_folder = 'images_v2/'
output_folder = 'images_aug/'

images = os.listdir(input_folder)
images = list(filter(lambda x: x.endswith('jpg'), images))


def save_image(path, img):
    conv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, conv_img)


aug = strong_aug(p=1)
for image in tqdm(images):
    input_image_path = input_folder + image
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    save_image(output_folder + image, img)
    aug_num = 5
    for i in range(aug_num):
        img_aug = aug(image=img)['image']
        output_image_path = output_folder + str(image)[:-4] + f'_aug_{i}.jpg'
        save_image(output_image_path, img_aug)
