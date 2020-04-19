#!/bin/bash
# Download weights for logo detection
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1Uk8qKPkBtpevkGtuLrNONaPEIjKSxyQO" -O yolov3_ckpt_98.pth
# Download weights for backbone network
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1aCHAEYvyUouoPkW22dGrUi8oz6MYOOFC" -O yolov3-tiny.conv.15
