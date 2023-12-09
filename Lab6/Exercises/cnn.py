import numpy as np

from matplotlib import pyplot as plt

# Create a black image
data = np.zeros((128, 128, 3), dtype=np.uint8)

# Function to draw on the image
def draw(img, x, y, color):
    img[x, y] = [color, color, color]

def convolve(data, kernel, stride=1):
    m, n = kernel.shape
    height, width, _ = data.shape

    output_height = (height - m) // stride + 1
    output_width = (width - n) // stride + 1

    result = np.zeros((output_height, output_width))

    for i in range(0, height - m + 1, stride):
        for j in range(0, width - n + 1, stride):
            result[i // stride, j // stride] = np.sum(data[i:i+m, j:j+n, 0] * kernel)

    return result
    
# Draw on the image
draw(data, 5, 5, 100)
draw(data, 6, 6, 100)
draw(data, 5, 6, 255)
draw(data, 6, 5, 255)

# Draw additional shapes on the image
for i in range(128):
    for j in range(128):
        if (i-64)**2 + (j-64)**2 < 900:
            draw(data, i, j, 200)
        elif i > 100 and j > 100:
            draw(data, i, j, 255)
        elif (i-15)**2 + (j-110)**2 < 25:
            draw(data, i, j, 150)
        elif (i-15)**2 + (j-110)**2 == 25 or (i-15)**2 + (j-110)**2 == 26:
            draw(data, i, j, 255)

# Display the original image
plt.imshow(data, interpolation='nearest')
plt.title('Original Image')
plt.show()

# Define the vertical edge detection kernel
kernel_vertical = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

kernel_horizontal = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])

# 0*
s1 = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# 45*
s2 = np.array([
    [0, 1, 2],
    [-1, 0, 1],
    [-2, -1, 0]
])

# 90*
s3 = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

# 135*
s4 = np.array([
    [2, 1, 0],
    [1, 0, -1],
    [0, -1, -2]
])

# Perform convolution
result = convolve(data, kernel_vertical)

# Display
plt.imshow(result, cmap='gray', interpolation='nearest')
plt.title('After convolve with kernel vertical')
plt.show()

result = convolve(data, kernel_horizontal)

# Display
plt.imshow(result, cmap='gray', interpolation='nearest')
plt.title('After convolve with kernel horizontal')
plt.show()

result = convolve(data, s1, 2)

# Display
plt.imshow(result, cmap='gray', interpolation='nearest')
plt.title('After convolve with kernel vertical stride = 2')
plt.show()

result = convolve(data, kernel_horizontal, 2)

# Display
plt.imshow(result, cmap='gray', interpolation='nearest')
plt.title('After convolve with kernel horizontal stride = 2')
plt.show()

result = convolve(data, s1)

# Display
#plt.imshow(result, cmap='gray', interpolation='nearest')
#plt.title('After convolve with sobel 0*')
#plt.show()

#result = convolve(data, s2)

# Display
plt.imshow(result, cmap='gray', interpolation='nearest')
plt.title('After convolve with sobel 45* - oblique')
plt.show()

result = convolve(data, s3)

# Display
#plt.imshow(result, cmap='gray', interpolation='nearest')
#plt.title('After convolve with sobel 90*')
#plt.show()

result = convolve(data, s4)

# Display
plt.imshow(result, cmap='gray', interpolation='nearest')
plt.title('After convolve with sobel 135 - oblique*')
plt.show()