import cv2
import pytesseract
from pytesseract import Output
import numpy as np

def preprocess_and_ocr(image_path):
    img = cv2.imread(image_path)
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 명암 대비 강화
    contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    # 노이즈 제거
    denoised = cv2.GaussianBlur(contrast, (5, 5), 0)
    # 이진화
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tesseract 설정
    custom_config = r'-l kor+eng --oem 1 --psm 4'
    # 텍스트 인식
    text = pytesseract.image_to_string(binary, config=custom_config, output_type=Output.STRING)
    return text

# 이미지 경로
image_path = '/Users/leechanhyeon/Desktop/ocr/test4.jpeg'
recognized_text = preprocess_and_ocr(image_path)
print("인식된 텍스트:")
print(recognized_text)

