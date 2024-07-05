import cv2
import pytesseract
import numpy as np 
from PIL import Image, ImageEnhance, ImageFilter

# 이미지 로드
image_path = '/Users/leechanhyeon/Desktop/ocr/test4.jpeg'
image = Image.open(image_path)

# 이미지 전처리
image = image.convert('L')  # 그레이스케일
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(2)  # 명암 대비 강화
image = image.filter(ImageFilter.SHARPEN)  # 선명도 강화

# OpenCV로 이미지 처리
open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

# Tesseract를 사용한 텍스트 추출
custom_config = r'--oem 1 --psm 4 -l kor+eng'  # 한글+영어 인식 설정
extracted_text = pytesseract.image_to_string(gray_image, config=custom_config)

print(extracted_text)

