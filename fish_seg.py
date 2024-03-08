import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

# Define lower and upper bounds for orange color in HSV
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([15, 255, 255])

# Load the mask image
mask = Image.open('mask.png')
mask = np.asarray(mask)

slice_3 = mask[:, :, 3]
new_mask = np.repeat(slice_3[:, :, np.newaxis], 3, axis=2)
new_mask = new_mask / 255.0  # Normalize new_mask to have values in the range [0, 1]

image_path = 'frame_set1/frame_1.png'
img = cv2.imread(image_path)
frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
orig = frame.copy()

tl = (700,80); tr = (1460,100)
bl = (670,710); br = (1440,750)

cv2.line(frame, tl, tr, (255, 0, 0), 2)  # Draw a line between top left and top right
cv2.line(frame, tr, br, (255, 0, 0), 2)  # Draw a line between top right and bottom right
cv2.line(frame, br, bl, (255, 0, 0), 2)  # Draw a line between bottom right and bottom left
cv2.line(frame, bl, tl, (255, 0, 0), 2)  # Draw a line between bottom left and top left

s = frame.shape
pts1 = np.float32([tl, bl, tr, br])
pts2 = np.float32([[0,0],[0,s[0]],[s[1],0],[s[1],s[0]]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(frame, matrix, (s[1], s[0]))

# Multiply new_mask with result
masked_result = result * new_mask

plt.imshow(masked_result.astype(np.uint8))  # Convert to uint8 for visualization
plt.axis('off')
plt.show()
