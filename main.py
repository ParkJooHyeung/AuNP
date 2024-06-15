import os
import subprocess
import cv2
from rembg import remove
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import numpy as np
import csv


# 함수 정의
def run_command(command):
    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(process.stdout.decode())
    print(process.stderr.decode())


def is_image_file(file_path):
    try:
        Image.open(file_path).verify()
        return True
    except (IOError, SyntaxError):
        return False


def read_images_from_subdirectories(dataset_path):
    image_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff')):
                full_path = os.path.join(root, file)
                if is_image_file(full_path):
                    image_files.append(full_path)
    return image_files


# RGB 값을 CSV 파일로 저장하는 함수
def save_rgb_to_csv(image_path, csv_path):
    try:
        image = Image.open(image_path)
    except UnidentifiedImageError:
        print(f'이미지 파일을 열 수 없습니다: {image_path}')
        return

    image = image.convert("RGB")
    width, height = image.size

    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["R", "G", "B"])

        for y in range(height):
            for x in range(width):
                r, g, b = image.getpixel((x, y))
                csvwriter.writerow([r, g, b])

    print(f'RGB 값이 {csv_path}에 저장되었습니다.')


# CSV 파일을 읽고 RGB 평균 값을 계산하는 함수
def calculate_rgb_average_from_csv(csv_path):
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # 헤더 스킵

        r_values, g_values, b_values = [], [], []

        for row in csvreader:
            r, g, b = map(int, row)
            r_values.append(r)
            g_values.append(g)
            b_values.append(b)

        r_mean = np.mean(r_values)
        g_mean = np.mean(g_values)
        b_mean = np.mean(b_values)

        return r_mean, g_mean, b_mean


# 설치가 필요한 패키지 설치
run_command('pip install rembg')
run_command('pip install torch torchvision torchaudio')  # YOLOv5의 종속성 설치

# 사용자 입력
dataset_path = "./240409_trypsin_activity_test"  # 예: /path/to/dataset
weights_path = "./yolov5/runs/train/circle_yolov5s/weights/last.pt" # 예: /path/to/weights/best.pt

# 이미지 경로 읽기
image_paths = read_images_from_subdirectories(dataset_path)
print(f"총 읽은 이미지 수: {len(image_paths)}")

# 상위 디렉토리 설정
output_dir = "./processed_images"
os.makedirs(output_dir, exist_ok=True)

# 모든 이미지의 RGB 평균값을 저장할 CSV 파일 생성
all_avg_rgb_csv_path = os.path.join(output_dir, 'all_avg_rgb_values.csv')
with open(all_avg_rgb_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Image Name", "R Mean", "G Mean", "B Mean"])

# 이미지 처리 반복
for img_input in image_paths:
    file_name, file_extension = os.path.splitext(os.path.basename(img_input))
    print(file_name)
    print(file_extension)

    # YOLOv5 검출 실행
    detect_command = f'python yolov5/detect.py --weights "{weights_path}" --img 600 --conf 0.5 --source "{img_input}" --save-txt --project "{output_dir}" --name "{file_name}"'
    run_command(detect_command)

    # 이미지 읽기
    image = cv2.imread(img_input)
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {img_input}")
        continue

    # 이미지 크기 가져오기
    height, width, _ = image.shape

    # bounding box 좌표 파일 경로
    bbox_file_path = f'{output_dir}/{file_name}/labels/{file_name}.txt'
    if not os.path.exists(bbox_file_path):
        print(f"bounding box 파일을 찾을 수 없습니다: {bbox_file_path}")
        continue

    # bounding box 좌표 읽기
    with open(bbox_file_path, 'r') as f:
        lines = f.readlines()

    # bounding box 좌표 처리 및 이미지 크롭 반복
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        bbox = list(map(float, parts[1:]))

        # bounding box의 좌표 변환 (중앙 좌표에서 좌상단 좌표로 변환)
        x_center, y_center, box_width, box_height = bbox
        x1 = max(int((x_center - box_width / 2) * width), 0)
        y1 = max(int((y_center - box_height / 2) * height), 0)
        x2 = min(int((x_center + box_width / 2) * width), width)
        y2 = min(int((y_center + box_height / 2) * height), height)

        if x1 >= x2 or y1 >= y2:
            print(f"유효하지 않은 bounding box 좌표: {x1}, {y1}, {x2}, {y2}")
            continue

        # 크롭
        cropped_image = image[y1:y2, x1:x2]

        # 크롭된 이미지 저장 경로 생성
        cropped_image_path = f'{output_dir}/{file_name}/cropped_{idx}_{file_name}.png'

        # 크롭된 이미지 저장
        cv2.imwrite(cropped_image_path, cropped_image)
        print(f'크롭된 이미지가 저장되었습니다: {cropped_image_path}')

        # 크롭된 이미지 출력
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Cropped Image {idx}')
        plt.axis('off')
        plt.show()

        try:
            remove_input = Image.open(cropped_image_path)
        except UnidentifiedImageError:
            print(f'이미지 파일을 열 수 없습니다: {cropped_image_path}')
            continue

        # 배경 제거
        output = remove(remove_input)

        output_path = f'{output_dir}/{file_name}/back_remove_RGB_{idx}_{file_name}.png'

        # 이미지 저장
        output.save(output_path)
        print(f'배경 제거된 이미지가 저장되었습니다: {output_path}')

        # 배경 제거된 이미지 출력
        output_rgb = output.convert("RGB")
        plt.imshow(output_rgb)
        plt.title(f'Background Removed Image {idx}')
        plt.axis('off')
        plt.show()

        # RGB 값을 CSV 파일로 저장
        csv_path = f'{output_dir}/{file_name}/rgb_values_{idx}_{file_name}.csv'
        save_rgb_to_csv(output_path, csv_path)

        # CSV 파일을 읽고 RGB 평균 값을 계산
        r_mean, g_mean, b_mean = calculate_rgb_average_from_csv(csv_path)
        print(f'RGB 평균값 - R: {r_mean}, G: {g_mean}, B: {b_mean}')

        # 평균값을 파일에 저장
        with open(all_avg_rgb_csv_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([f'{file_name}_{idx}', r_mean, g_mean, b_mean])
        print(f'RGB 평균값이 {all_avg_rgb_csv_path}에 저장되었습니다.')
