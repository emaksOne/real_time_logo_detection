### PyTorch Tiny-Yolo-v3 for real-time detection on webcam

Yolo v3 implementation was taken from https://github.com/eriklindernoren/PyTorch-YOLOv3

I have chosen 3 classes: adidas, nike, puma. Collect dataset by shooting photos from a webcam. It is clothes with brand logos. Then I have performed data augmentation to increase my dataset. 

Go to weights folder and run download_weights.sh to download weights for logo detection (yolov3_ckpt_98.pth)

Also, you can train your weights on your data that located in:
* images - data/custom/images
* annotations - data/custom/labels
* images that used for train - data/custom/train.txt
* images that used for validation - data/custom/valid.txt

```
python train.py --model_def config/yolov3-tiny-custom.cfg --data_config config/custom.data --pretrained_weights weights/yolov3-tiny.conv.15
```

For detection from webcam run command
```
python webcam.py --model_def config/yolov3-tiny-custom.cfg --class_path data/custom/classes.names --weights_path weights/yolov3_ckpt_98.pth
```