from PIL import Image
import os

image_dir = 'imgSamples'

# Create a blank image with a white background
result_width = 64*6
result_height = 64*8
result = Image.new('RGB', (result_width, result_height), (255, 255, 255))

# Open the images
for i in range(48):
    image_file = os.path.join(image_dir, f'{i:06d}.png')
    image = Image.open(image_file)

    # Paste the images onto the blank image
    result.paste(image, (i%6*64, int(i/6)*64))

# Save the result image as a PNG file
result.save("result.png")