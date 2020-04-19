import argparse
import datetime
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Darknet
from utils.datasets import pad_to_square, resize
from utils.utils import load_classes, non_max_suppression, rescale_boxes


def init_model(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(opt.weights_path))
        else:
            model.load_state_dict(torch.load(
                opt.weights_path, map_location=torch.device('cpu')))

    model.eval()

    return model, device


def detect(origin_frame, opt):
    frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
    frame = transforms.ToTensor()(frame)
    frame, _ = pad_to_square(frame, 0)
    frame = resize(frame, opt.img_size)

    x = frame.unsqueeze(0).to(device)
    input_img = Variable(x.type(Tensor))

    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(
            detections, opt.conf_thres, opt.nms_thres)

    detections = detections[0]
    if detections is not None:
        detections = rescale_boxes(
            detections, opt.img_size, origin_frame.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if cls_conf.item() < 0.7:
                break
            class_name = classes[int(cls_pred)]
            topleft = (x1, y2)
            botomright = (x2, y1)
            color = colors[int(cls_pred)]
            thickness = 2

            cv2.rectangle(origin_frame, topleft, botomright, color, thickness)

            text = f'{classes[int(cls_pred)]} {cls_conf.item():.2f}'
            leftbottom = (x1, y1)
            put_text(origin_frame, text, color, leftbottom)

    return origin_frame


def put_text(image, text, color, leftbottom):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    text_thickness = 1
    cv2.putText(image, text, leftbottom, font, font_scale,
                color, text_thickness, cv2.LINE_AA)

    rectangle_bgr = color
    text_color = (255, 255, 255)
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=text_thickness)[0]

    text_offset_x, text_offset_y = leftbottom

    box_coords = ((text_offset_x, text_offset_y), (text_offset_x +
                                                   text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(image, box_coords[0],
                  box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(image, text, (text_offset_x, text_offset_y), font,
                fontScale=font_scale, color=text_color, thickness=text_thickness)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str,
                        default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str,
                        default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str,
                        default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float,
                        default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4,
                        help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int,
                        default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416,
                        help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str,
                        help="path to checkpoint model")
    opt = parser.parse_args()

    model, device = init_model(opt)
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    np.random.seed(42)
    colors = [tuple(np.random.randint(255, size=3)) for _ in classes]
    # colors = [
    #     (255, 0, 0),  # addidas
    #     (0, 255, 0),  # nike
    #     (0, 0, 255)  # puma
    # ]

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, origin_frame = cap.read()
        origin_frame = cv2.resize(
            origin_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        origin_frame = cv2.flip(origin_frame, 180)
        origin_frame = detect(origin_frame, opt)
        cv2.imshow('Input', origin_frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
