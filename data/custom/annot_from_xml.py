from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
import os

images_folder = 'images/'
labels_folder = 'labels/'
xml_folder = 'xmls/'
class_name_file = 'classes.names'

class_names = []
with open(class_name_file, 'r') as f:
    class_names = f.readlines()
    class_names = [x.strip() for x in class_names]

print(class_names)

xmls = os.listdir(xml_folder)

def parse_xml(file_path):
    in_file = open(file_path)
    tree=ET.parse(in_file)
    root = tree.getroot()
    file_name = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    # print(file_name)
    # print(width, height)
    # print(file_path)
    with open(labels_folder + str(file_name)[:-4] + '.txt', 'w') as f:
        for i, obj in enumerate(root.iter('object')):
            cls = obj.find('name').text
            
            cls_id = class_names.index(cls)
            xmlbox = obj.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            ymin = int(xmlbox.find('ymin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymax = int(xmlbox.find('ymax').text)


            r_xmin = float(xmin) / float(width)
            r_xmax = float(xmax) / float(width)
            r_ymin = float(ymin) / float(height)
            r_ymax = float(ymax) / float(height)


            r_x_cnt = (r_xmax + r_xmin)/2.0
            r_y_cnt = (r_ymax + r_ymin)/2.0
            r_width = r_xmax - r_xmin
            r_height = r_ymax - r_ymin


            row = f'{cls_id} {r_x_cnt} {r_y_cnt} {r_width} {r_height}'
            f.write(row)
            f.write('\n')

for xml in xmls:
    file_path = xml_folder + xml
    parse_xml(file_path)


images = os.listdir(images_folder)
images = list(filter(lambda x: x.endswith('jpg'), images))

train, val = train_test_split(images, test_size=0.2, random_state=69)
print(len(train), len(val))

with open('train.txt', 'w') as f:
    for sample in train:
        path = 'data/custom/images/' + sample
        f.write(path)
        f.write('\n')

with open('valid.txt', 'w') as f:
    for sample in val:
        path = 'data/custom/images/' + sample
        f.write(path)
        f.write('\n')


