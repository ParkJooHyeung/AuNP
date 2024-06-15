import os
import subprocess
from glob import glob
import yaml

def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with error: {result.stderr}")
    else:
        print(result.stdout)

# Step 1: Download and unzip dataset
run_command("curl -L 'https://universe.roboflow.com/ds/D8XyZgm0SJ?key=fP6OxvAgZh' > roboflow.zip")
run_command("unzip roboflow.zip -d circle_dataset")
run_command("rm roboflow.zip")

# YOLOv5 저장소 복제
if not os.path.exists('yolov5'):
    run_command('git clone https://github.com/ultralytics/yolov5')

# YOLOv5 요구사항 설치
run_command('pip install -r yolov5/requirements.txt')

# Step 4: Prepare the dataset
train_img_list = glob('/home/server3/jhpark/AuNP/circle_dataset/train/images/*.jpg')
print(f"Number of training images: {len(train_img_list)}")
val_img_list = glob('/home/server3/jhpark/AuNP/circle_dataset/valid/images/*.jpg')
print(f"Number of validation images: {len(val_img_list)}")
test_img_list = glob('/home/server3/jhpark/AuNP/circle_dataset/test/images/*.jpg')
print(f"Number of validation images: {len(test_img_list)}")

# Write the image paths to the respective text files
with open('/home/server3/jhpark/AuNP/circle_dataset/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open('/home/server3/jhpark/AuNP/circle_dataset/val.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')

with open('/home/server3/jhpark/AuNP/circle_dataset/test.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')

# Load the existing data.yaml file and update the paths
with open('/home/server3/jhpark/AuNP/circle_dataset/data.yaml', 'r') as f:
    data = yaml.safe_load(f)

print(data)

data['train'] = '/home/server3/jhpark/AuNP/circle_dataset/train.txt'
data['val'] = '/home/server3/jhpark/AuNP/circle_dataset/val.txt'
data['test'] = '/home/server3/jhpark/AuNP/circle_dataset/test.txt'

with open('/home/server3/jhpark/AuNP/circle_dataset/data.yaml', 'w') as f:
    yaml.dump(data, f)

print(data)
