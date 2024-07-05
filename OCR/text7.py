from gpiozero import Button
from picamera2 import Picamera2
from signal import pause
from time import sleep
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import pytesseract

button_pin = 15

button = Button(button_pin)#, bounce_time=0.1)
camera = Picamera2()

print(button)

def compress_image(input_image_path, output_image_path, quality = 10):
    with Image.open(input_image_path)as img:
        img.save(output_image_path, 'JPEG', quality = quality)

def button_pushed():
    print(button)
    print("Button pushed!")
    camera.resolution = (2592, 1944)
    camera.framerate = 15
    camera.brightness = 50
    camera.awb_mode = 'auto'
    camera.exposure_mode = 'auto'
    camera.start_preview()
    sleep(1)
    camera.start_and_capture_file('/home/dot/boinDOT/code/ocr/3.jpeg')
    camera.stop_preview()
   
    src = cv2.imread('/home/dot/boinDOT/code/ocr/3.jpeg', 1)

    # grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # canny
    canned = cv2.Canny(gray, 100, 200)

    # dilate to close holes in lines
    kernel = np.ones((10,1),np.uint8) # 가로 1 세로 10
    mask = cv2.dilate(canned, kernel, iterations = 20)

    # contours 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 contours 찾기
    biggest_cntr = None
    biggest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > biggest_area:
            biggest_area = area
            biggest_cntr = contour

    # 외곽 box
    rect = cv2.minAreaRect(biggest_cntr)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # 외곽 box 그리기
    src_box = src.copy()
    cv2.drawContours(src_box, [box], 0, (0, 255, 0), 3)

    # angle 계산
    angle = rect[-1]
    if angle > 45:
        angle = -(90 - angle)

    # 기울기 조정
    rotated = src.copy()
    (h, w) = rotated.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(rotated, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 회전된 박스 좌표 찾기
    ones = np.ones(shape=(len(box), 1))
    points_ones = np.hstack([box, ones])
    transformed_box = M.dot(points_ones.T).T

    y = [transformed_box[0][1], transformed_box[1][1], transformed_box[2][1], transformed_box[3][1]]
    x = [transformed_box[0][0], transformed_box[1][0], transformed_box[2][0], transformed_box[3][0]]

    y1, y2 = int(min(y)), int(max(y))
    x1, x2 = int(min(x)), int(max(x))

    # crop
    crop = rotated[y1:y2, x1:x2]

    # 흑백처리
    gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("/home/dot/boinDOT/code/1/mask1.jpeg", mask)
    cv2.imwrite("/home/dot/boinDOT/code/1/box1.jpeg", src_box)
    cv2.imwrite("/home/dot/boinDOT/code/1/canny1.jpeg", canned)
    cv2.imwrite("/home/dot/boinDOT/code/1/rotated1.jpeg", rotated)
    cv2.imwrite("/home/dot/boinDOT/code/1/cropped1.jpeg", crop)
    cv2.imwrite("/home/dot/boinDOT/code/1/gray1.jpeg", gray2)


    img = cv2.imread('/home/dot/boinDOT/code/1/gray1.jpeg', cv2.IMREAD_COLOR)
    result = pytesseract.image_to_string(img, lang='eng+kor')
    print(">>")
    print(result)
    print("<<")
   


button.when_pressed = button_pushed
#button_pin.close()
pause()

