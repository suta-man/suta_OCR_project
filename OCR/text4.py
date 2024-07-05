import cv2
import pytesseract
from pytesseract import Output

# Tesseract의 설치 경로를 지정 (경로는 설치 환경에 따라 다를 수 있습니다)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

def ocr_image(image_path):
    # 이미지 로드 및 전처리
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # 팽창과 폐쇄를 통한 이미지 개선
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Tesseract를 사용한 텍스트 인식
    custom_config = r'-l kor+eng --oem 3 --psm 4'
    details = pytesseract.image_to_string(opening, config=custom_config, output_type=Output.DICT)

    print("인식된 텍스트:")
    print(details['text'])

# 이미지 경로
image_path = '/Users/leechanhyeon/Desktop/ocr/test4.jpeg'  # 이미지 경로 지정
ocr_image(image_path)
