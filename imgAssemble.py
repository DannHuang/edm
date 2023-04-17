from PIL import Image
import os

image_dir = 'imgSamples'

# Create a blank image with a white background
result_width = 32*10
result_height = 32*10
result = Image.new('RGB', (result_width, result_height), (255, 255, 255))

# Open the images
for i in range(100):
    image_file = os.path.join(image_dir, f'{i:06d}.png')
    image = Image.open(image_file)

    # Paste the images onto the blank image
    result.paste(image, (i%10*32, int(i/10)*32))

# Save the result image as a PNG file
result.save("result.png")