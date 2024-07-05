import cv2
import numpy as np
import pytesseract
from PIL import Image

# 이미지 로드
image_path = '/Users/leechanhyeon/Desktop/ocr/test2.png'  # 이미지 경로를 올바르게 설정하세요
image = cv2.imread(image_path)

# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 이진화
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# 노이즈 제거 및 경계 강화
kernel = np.ones((1, 1), np.uint8)
dilation = cv2.dilate(binary, kernel, iterations=1)
erosion = cv2.erode(dilation, kernel, iterations=1)

# Tesseract OCR 적용
custom_config = r'--oem 3 --psm 6 -l kor'
extracted_text = pytesseract.image_to_string(Image.fromarray(erosion), config=custom_config)

print(extracted_text)



