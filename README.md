# AuNP

step 1: custom image dataset download
python3 custom_data_load.py

step 2: custom dataset train
cd yolov5
python3 train.py --img 600 --batch 16 --epochs 100 --data ../circle_dataset/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name circle_yolov5s --nosave --optimizer Adam --patience 10

step 3: AuNP detect & crop
cd ..
python3 main.py



