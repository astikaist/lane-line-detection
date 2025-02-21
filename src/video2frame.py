import cv2

# Opens the Video file
# cap= cv2.VideoCapture('samples/sources/hujan/rain-22.avi')
# cap= cv2.VideoCapture('samples/sources/malam/640x400.avi')
# cap= cv2.VideoCapture('dataset/Rain-640x400-4fps/R23.avi')
# cap= cv2.VideoCapture('dataset/Night-640x400-4fps/N50.avi')
# cap= cv2.VideoCapture('out/N29.avi')
# cap= cv2.VideoCapture('dataset/hujan/rain-22.avi')
cap= cv2.VideoCapture('dataset/malam/night-65.avi')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    # cv2.imwrite('dataset/Rain-640x400-4fps/frame/R23-'+str(i)+'.jpg',frame)
    # cv2.imwrite('dataset/Night-640x400-4fps/frame/N50-'+str(i)+'.jpg',frame)
    # cv2.imwrite('out/frames/HoughTransform(N)/N07-'+str(i)+'.jpg',frame)
    # cv2.imwrite('samples/frames/malam/malam(1)'+str(i)+'.jpg',frame)
    # cv2.imwrite('samples/frames/hujan/rain(22)'+str(i)+'.jpg',frame)
    cv2.imwrite('dataset/malam/frame/N65/N65-'+str(i)+'.jpg',frame)
    i+=1

cap.release()
cv2.destroyAllWindows()