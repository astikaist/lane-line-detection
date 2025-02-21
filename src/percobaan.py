import cv2 
import numpy as np
from os import listdir
from os.path import isfile, join
import os as osfnc
from matplotlib import pyplot as plt

def bilateral_filter(image):
    return cv2.bilateralFilter(image, 13, 75, 75)

def blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def median_filter(image):
    return cv2.medianBlur(image, ksize=5)

def mean_filter(image):
    return cv2.blur(image, (3,3))

def dilasi(image):
    kernel = np.ones((5,5), np.uint8)
    return cv2.dilate(image, kernel, iterations =1)

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8,8)) 
    # return clahe.apply(grayscale(image))
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    return cv2.cvtColor(cv2.merge((h, s, clahe.apply(v))), cv2.COLOR_HSV2BGR)    

def create_mask(image, vertices):
    mask_color = (255,) * image.shape[2] if len(image.shape) > 2 else 255 
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.array([vertices], np.int32), mask_color)
    return mask

def create_region_of_interest_vertices(image):
    height = image.shape[0]
    width = image.shape[1]
    return [
        (0, height),
        (0, (height * 2) / 3),
        (width / 6.2, height / 2.1),
        (width / 2.5, height / 2.1),
        (width / 1.6, (height * 2) / 3),
        (width, height)
    ]

def crop(image, mask):
    return cv2.bitwise_and(image, mask)

def detect_edges(image):
    return cv2.Canny(image, 100, 300, apertureSize=3)

def detect_lines(image):
    return cv2.HoughLinesP(image, rho = 10, theta = np.pi / 180, threshold = 100, lines = np.array([]), minLineLength = 40, maxLineGap = 100) 

def draw_lines(image, lines):
    if lines is None:
        return image
    
    dst = np.copy(image)
    blank_image = np.zeros((dst.shape[0], dst.shape[1], 3), dtype = np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness = 3)

    return cv2.addWeighted(dst, 0.8, blank_image, 1, 1)

def equalize_histrogram(image, channel = None):
    if channel is not None and type(channel) is not list:
        raise Exception('channel must be a list')
    
    if channel is not None:
        dst = cv2.cvtColor(image, channel[0])
        dst[:, :, 0] = cv2.equalizeHist(dst[:, :, 0])
        return cv2.cvtColor(dst, channel[1])

    return cv2.equalizeHist(image)

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def unsharp(image, blured):
    return cv2.addWeighted(blured, 1.5, blured, -0.5, 0, image)

def similar(color):
    return color[0] == color[1] and color[1] == color[2]

# ========================================================
def binary_hsv_mask(image, color_range):
    return cv2.inRange(image, np.array(color_range[0]), np.array(color_range[1]))

def binary_gray_mask(image, color_range):
    return cv2.inRange(image, color_range[0][0], color_range[1][0])

def binary_mask_apply(image, mask):
    dst = np.zeros_like(image)

    for i in range(3):
        dst[:,:,i] = mask.copy()

    return dst

def filter_by_color_ranges(image, color_ranges):
    dst = np.zeros_like(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for color_range in color_ranges:
        color_bottom = color_range[0]
        color_top = color_range[1]

        if similar(color_range[0]) and similar(color_range[1]):
            mask = binary_gray_mask(gray_image, color_range)
        else:
            mask = binary_hsv_mask(hsv_image, color_range)

        dst = cv2.addWeighted(binary_mask_apply(image, mask), 1.0, dst, 1.0, 0.0)

    return dst

def color_threshold(image, white_value):
    return filter_by_color_ranges(image, [
        [
            [white_value, white_value, white_value],
            [255, 255, 255]
        ],
        [
            [80, 90, 90],
            [120, 255, 255]
        ]
    ])

# ----------------------------------------
def detect_hough_lines(image, white_value = 200):
    if white_value < 150:
        return None

    lines = cv2.HoughLinesP(detect_edges(color_threshold(image, white_value)), 2, np.pi / 180, 50, np.array([]), 20, 100)

    if lines is None:
        return detect_hough_lines(image, white_value - 5)

    return lines

def detect_lane_of(image):
    filtered_image = median_filter(grayscale(apply_clahe(equalize_histrogram(image, [cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR]))))
    # filtered_image = median_filter(apply_clahe(image))
    detected_edges_image = detect_edges(filtered_image)
    dst = crop(
        detected_edges_image,
        create_mask(
            detected_edges_image,
            create_region_of_interest_vertices(filtered_image)
        )
    )
    return draw_lines(image, detect_lines(dst))

def play(video):
    i = 1
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            cv2.imshow('Frame', detect_lane_of(frame))
            
            print("Frame: %d" % i)
            i += 1

            # Press ESC on keyboard to  exit
            if cv2.waitKey(1) in [27, 1048603]:
                break

        else:
            break

    video.release()
    cv2.destroyAllWindows()

def show(image, title = 'CV2'):
    cv2.imshow(title, image)

    while True:
        # Press ESC on keyboard to  exit
        if cv2.waitKey(1) in [27, 1048603]:
            cv2.destroyAllWindows()
            break
    
def detect_lane_and_save_video(video):
    # TODO: Debug the writter
    cap = cv2.VideoCapture('samples/sources/malam/night-68.avi')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    out = cv2.VideoWriter('out/N68.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, size)
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            out.write(detect_lane_of (frame))

            # Press ESC on keyboard to  exit
            if cv2.waitKey(1) in [27, 1048603]:
                break

        else:
            break

# show result video
# play(cv2.VideoCapture('samples/sources/malam/night-68.avi'))
# play(cv2.VideoCapture('samples/sources/hujan/rain-22.avi'))
# detect_lane_and_save_video(cv2.VideoCapture('samples/sources/hujan/rain-22.avi'))
# detect_lane_and_save_video(cv2.VideoCapture('samples/sources/malam/night-68.avi'))

# show result in image
# img= cv2.imread('samples/frames/hujan/rain(22)205.jpg')
# # img= cv2.imread('samples/frames/malam/malam(1)113.jpg')
# cv2.imshow('Original Image', img)
# # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
# # img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# img2 = equalize_histrogram(img, [cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR])
# cv2.imshow('CLAHE', img2)
# RGBtoGray= grayscale(img2)
# cv2.imshow('Grayscale', RGBtoGray)
# medianFilter= median_filter(RGBtoGray)
# cv2.imshow('Median Filter', medianFilter)
# Canny= detect_edges(medianFilter)
# cv2.imshow('Canny', Canny)
# RoI= crop(Canny, create_mask(Canny, create_region_of_interest_vertices(RGBtoGray)))
# cv2.imshow('RoI', RoI)
# lines= draw_lines(img, detect_lines(RoI))
# cv2.imshow('Hough Line Transform', lines)
# # show(detect_lane_of(cv2.imread(cv2.samples.findFile('samples/resize/1/000001.jpg'))))
# # detect_lane_and_save_video(cv2.VideoCapture('samples/source/6.mp4'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# save result in .jpg
# mypath = osfnc.getcwd()
# # arahkan ke folder tempat gambar (input dan output)
# # inputFrame = mypath + '\\samples\\frames\\malam'
# # outputFrame = mypath + '\\out\\frames\\malam'
# # inputFrame = mypath + '\\samples\\frames\\hujan'
# # outputFrame = mypath + '\\out\\frames\\hujan'
# # inputFrame = mypath + '\\dataset\\Rain-640x400-4fps\\frame'
# # outputFrame = mypath + '\\dataset\\Rain-640x400-4fps\\HoughTransform'
# # inputFrame = mypath + '\\dataset\\Rain-640x400-4fps\\frame\\koreksi'
# # outputFrame = mypath + '\\dataset\\Rain-640x400-4fps\\HoughTransform\\koreksi'
# # inputFrame = mypath + '\\dataset\\Night-640x400-4fps\\frame'
# # outputFrame = mypath + '\\dataset\\Night-640x400-4fps\\HoughTransform'
# # inputFrame = mypath + '\\dataset\\hujan\\frame\\R22'
# # outputFrame = mypath + '\\dataset\\hujan\\HoughTransform\\R22'
# # inputFrame = mypath + '\\dataset\\malam\\frame\\N69'
# # outputFrame = mypath + '\\dataset\\malam\\HoughTransform\\N69'
# # Buat daftar gambar
# list_img = []
# # Iterasi setiap file dalam mypath
# for f in listdir(inputFrame):
#     #Hitung seluruh direktori berkas
#     aa = join(inputFrame, f)
#     # periksa apakah itu file atau bukan
#     if isfile(aa):
#         # Jika itu sebuah file, periksa apakah itu .png/ .jpg
#         if f.endswith(".png") or f.endswith(".jpg"):
#             # Jika itu adalah .png atau .jpg lalu tambahkan ke daftar gambar
#             list_img.append(f)

# for u, nama_file in enumerate(list_img):
#     img= cv2.imread(join(inputFrame, nama_file),1)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
#     img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     RGBtoGray= grayscale(img2)
#     medianFilter= median_filter(RGBtoGray)
#     Canny= detect_edges(medianFilter)
#     RoI= crop(Canny, create_mask(Canny, create_region_of_interest_vertices(RGBtoGray)))
#     lines= draw_lines(img, detect_lines(RoI))
#     cv2.imwrite(join(outputFrame, nama_file), lines)