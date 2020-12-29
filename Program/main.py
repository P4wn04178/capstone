import cv2

fname = '1.jpg'
img = cv2.imread(fname, cv2.IMREAD_COLOR)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()