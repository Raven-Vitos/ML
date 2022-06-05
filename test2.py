import cv2

img = cv2.imread('img.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

res = faces.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)

for (x, y, w, h) in res:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    

cv2.imshow('Result', img)
cv2.waitKey(0)