
# importing cv2  and numpy
import cv2
import numpy as np

# Reading the image
imge = cv2.imread('./A1.jpg', cv2.IMREAD_COLOR)

# Using fastNlMeansDenoisingColored() function for denoising
img = cv2.fastNlMeansDenoisingColored(imge, None, 20, 10, 7, 21)

# Calculating the height weight and channels
h, w, c = img.shape

# Use the cvtColor() method of the cv2 module which takes the original image and the COLOR_BGR2GRAY attribute as an argument return black and white image.
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the threshold. In the threshold() method, the last argument defines the style of the threshold
_, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#Find the image contours. Use the findContours() which takes the image (we passed threshold here) and some attributes.
img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

# Sort the contours.
img_contours = sorted(img_contours, key=cv2.contourArea)
for i in img_contours:
    if cv2.contourArea(i) > 100:
        break


# Generate the mask using np.zeros:
mask = np.zeros(img.shape[:2], np.uint8)

# Draw contours:
cv2.drawContours(mask, [i], -1, 255, -1)

# Apply the bitwise_and operator:
new_img = cv2.bitwise_and(img, img, mask=mask)

# create zeros mask 2 pixels larger in each dimension
mask = np.zeros([h + 2, w + 2], np.uint8)

# do floodfill for white background
cv2.floodFill(new_img, mask, (619, 342), (255, 255, 255), (3, 151, 65), (3, 151, 65), flags=8)


# display the Oeriginal image
cv2.imshow("Original Image", img)

# display the Isolated purple shape
cv2.imshow("Isolated shape with background removed", new_img)

#display untill manually stop
cv2.waitKey(0)
