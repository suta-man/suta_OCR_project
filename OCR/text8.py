import cv2
import pytesseract
from pytesseract import Output
import numpy as np

def preprocess_and_ocr(image_path):
    # 이미지 불러오기
    img = cv2.imread(image_path)
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러링
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 캐니 에지 검출
    edges = cv2.Canny(blurred, 100, 200)
    
    # 침식 연산
    kernel = np.ones((5,5),np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # 컨투어 검출
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 컨투어 중 가장 큰 영역 찾기
    max_contour = max(contours, key=cv2.contourArea)
    
    # 컨투어를 둘러싸는 사각형 구하기
    x,y,w,h = cv2.boundingRect(max_contour)
    
    # 회전 및 크롭
    rotated = img.copy()
    angle = cv2.minAreaRect(max_contour)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    center = (x + w//2, y + h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(rotated, M, (rotated.shape[1], rotated.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cropped = cv2.getRectSubPix(rotated, (w, h), center)
    
    # 흑백처리
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # 텍스트 인식
    custom_config = r'-l kor+eng --oem 3 --psm 11'
    recognized_text = pytesseract.image_to_string(gray_cropped, config=custom_config)
    
    return recognized_text

# 이미지 경로
image_path = '/Users/leechanhyeon/Desktop/ocr/test4.jpeg'
recognized_text = preprocess_and_ocr(image_path)
print("인식된 텍스트:")
print(recognized_text)
