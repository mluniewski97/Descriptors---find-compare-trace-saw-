import cv2
import cv2 as cv
from matplotlib import pyplot as plt

video = cv2.VideoCapture('sawmovie.mp4')
img = cv.imread('saw1.jpg')
img = cv.resize(img, (640,480))

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img, None)

while video.isOpened():
    r, img2 = video.read()
    if r:
        img2 = cv.resize(img2, (640,480))
        kp2, des2 = sift.detectAndCompute(img, None)

        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

                img3 = cv.drawMatchesKnn(img, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                img3 = cv.resize(img3, (640,480))

                #plt.imshow(img3), plt.show()
                cv.imshow("sift", img3)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
