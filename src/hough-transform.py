import cv2 
import numpy as np
from os import listdir
from os.path import isfile, join
import os as osfnc
from matplotlib import pyplot as plt

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def median_filter(image):
    return cv2.medianBlur(image, ksize=5)

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8,8)) 
    return clahe.apply(image)    

def histeq(image):
    return cv2.equalizeHist(image)

def detect_edges(image):
    return cv2.Canny(image, 100, 300)

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

def detect_lines(image):
    return cv2.HoughLinesP(image, rho = 6, theta = np.pi / 60, threshold = 160, lines = np.array([]), minLineLength = 40, maxLineGap = 25) 

def draw_lines(image, lines):
    if lines is None:
        return image
    
    dst = np.copy(image)
    blank_image = np.zeros((dst.shape[0], dst.shape[1], 3), dtype = np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness = 3)

    return cv2.addWeighted(dst, 0.8, blank_image, 1, 1)

def detect_lane_of(image):
    filtered_image = apply_clahe((median_filter((grayscale(image)))))
    detected_edges_image = detect_edges(filtered_image)
    dst = crop(
        detected_edges_image,
        create_mask(
            detected_edges_image,
            create_region_of_interest_vertices(detected_edges_image)
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
    cap = cv2.VideoCapture('samples/sources/hujan/rain-22.avi')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    out = cv2.VideoWriter('out/rain-22.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, size)
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            out.write(detect_lane_of(frame))

            # Press ESC on keyboard to  exit
            if cv2.waitKey(1) in [27, 1048603]:
                break

        else:
            break

# show result video
# play(cv2.VideoCapture('samples/sources/malam/night-68.avi'))
# play(cv2.VideoCapture('samples/sources/hujan/rain-22.avi'))
# detect_lane_and_save_video(cv2.VideoCapture('samples/sources/hujan/rain-22.avi'))

# show result in image
# img= cv2.imread('samples/frames/hujan/rain(22)205.jpg')
img= cv2.imread('samples/frames/malam/malam(1)113.jpg')
cv2.imshow('Original Image', img)
gray_img = grayscale(img)
cv2.imshow('Grayscale', gray_img)
medianFilter= median_filter(gray_img)
cv2.imshow('Median Filter', medianFilter)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_img = clahe.apply(medianFilter)
cv2.imshow('CLAHE', clahe_img)
# equalizeHist = cv2.equalizeHist(medianFilter)
# cv2.imshow('Histeq', equalizeHist)
# Canny= detect_edges(equalizeHist)
Canny= detect_edges(clahe_img)
cv2.imshow('Canny', Canny)
# Vertices= create_region_of_interest_vertices(clahe_img)
# cv2.imshow('Vertices', Vertices)
masking = create_mask(Canny, create_region_of_interest_vertices(Canny))
cv2.imshow('Masking', masking)
RoI = crop(Canny, masking)
# RoI= crop(Canny, create_mask(Canny, create_region_of_interest_vertices(clahe_img)))
cv2.imshow('RoI', RoI)
lines= draw_lines(img, detect_lines(RoI))
cv2.imshow('Hough Line Transform', lines)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save result in .jpg
# mypath = osfnc.getcwd()
# # arahkan ke folder tempat gambar (input dan output)
# # inputFrame = mypath + '\\samples\\frames\\malam'
# # outputFrame = mypath + '\\out\\frames\\malam'
# # inputFrame = mypath + '\\samples\\frames\\hujan'
# # outputFrame = mypath + '\\out\\frames\\hujan'
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
#     gray_img = grayscale(img)
    # medianFilter= median_filter(gray_img)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    # clahe_img = clahe.apply(medianFilter)
#     Canny= detect_edges(clahe_img)
#     RoI= crop(Canny, create_mask(Canny, create_region_of_interest_vertices(clahe_img)))
#     lines= draw_lines(img, detect_lines(RoI))
#     cv2.imwrite(join(outputFrame, nama_file), lines)