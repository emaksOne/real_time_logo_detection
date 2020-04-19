FROM python:3.6

WORKDIR .

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./webcam.py --model_def config/yolov3-tiny-custom.cfg --class_path data/custom/classes.names --weights_path weights/yolov3_ckpt_98.pth" ]