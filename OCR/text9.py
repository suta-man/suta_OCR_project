import cv2
import pytesseract
from pytesseract import Output
import numpy as np

def preprocess_and_ocr(image_path):
    # 이미지 로드
    img = cv2.imread(image_path)
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 명암 대비 강화
    contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    
    # 노이즈 제거
    denoised = cv2.GaussianBlur(contrast, (5, 5), 0)

    return denoised

def detect_and_ocr_text(image_path):
    # 이미지 전처리
    denoised_image = preprocess_and_ocr(image_path)
    
    # 컨투어 검출
    contours, _ = cv2.findContours(denoised_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 각 컨투어에 대해 박스 그리기 및 텍스트 인식
    for contour in contours:
        # 컨투어를 감싸는 사각형 표시
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(denoised_image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        
        # 컨투어 영역에서 텍스트 추출
        custom_config = r'-l kor+eng --oem 1 --psm 4'
        text = pytesseract.image_to_string(denoised_image[y:y+h, x:x+w], config=custom_config)
        
        # 텍스트 출력
        print("인식된 텍스트:")
        print(text)
    
    # 결과 이미지 표시
    cv2.imshow('Processed Image', denoised_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 이미지 경로 설정
image_path = '/Users/leechanhyeon/Desktop/ocr/test5.png'
detect_and_ocr_text(image_path)

