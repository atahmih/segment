import matplotlib.pyplot as plt
import numpy as np
import cv2

# Example images as numpy arrays
img1 = cv2.imread('input_images/image2.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('output_images/image2 - stable diffusion.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.imread('output_images/image2 - openai.jpg')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

# Create a figure with 3 subplots in a row
fig, axs = plt.subplots(1, 3, figsize=(9, 3))

# Create visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img1)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img2)
plt.title('Stable Diffusion')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img3)
plt.title('OpenAI')
plt.axis('off')

plt.tight_layout()
# plt.savefig(f'{image_name}_combined.png')
# plt.show()
plt.savefig('comparison.png')