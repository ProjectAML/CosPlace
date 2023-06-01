from PIL import Image, ImageEnhance, ImageFilter,ImageDraw,ImageChops
from random import gauss
import numpy as np

def simulate_night(image):
    enhancer = ImageEnhance.Brightness(image)
    darkened = enhancer.enhance(0.9)

    enhancer = ImageEnhance.Color(darkened)
    desaturate = enhancer.enhance(0.95)

    for i in range(desaturate.width):
        for j in range(desaturate.height):
            r, g, b = image.getpixel((i, j))
            # Increase the blue channel by 30%
            image.putpixel((i, j), (r, g, int(b * 1.15)))

    enhancer = ImageEnhance.Contrast(image)
    contrast = enhancer.enhance(1.1)

    gradient = Image.new("RGBA", (contrast.width, contrast.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)
    for y in range(contrast.height):
        # Set the transparency of the pixel based on its position in the image
        # The top of the image will be fully opaque (255), and the bottom will be fully transparent (0)
        transparency = int(y / (contrast.height*1.2) * 255)
        draw.line((0, y, contrast.width, y), fill=(0, 0, 0, transparency))

    #rotate the gradient
    gradient = gradient.rotate(180)

    # Paste the gradient onto the image
    contrast.paste(gradient, (0, 0), gradient)

    image=dark_mask(contrast)

    image=add_gaussian_noise(image)


    return image

def dark_mask(image):
    """ Create random circular masks"""
    mask = Image.new("RGBA", (image.width, image.height), (0,0,0, 100))
    draw = ImageDraw.Draw(mask)
    for i in range(8):
      x = np.random.randint(0, image.width)
      y = np.random.randint(image.height/4, image.height)
      radius = np.random.randint(image.height/5, image.height/3)
      # Draw the circle
      draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0,0,0,0))

    # blur the mask
    mask = mask.filter(ImageFilter.GaussianBlur(radius=80))

    # Paste the mask onto the image
    image.paste(mask, (0, 0), mask)
    return image

def add_gaussian_noise(image):
    mean = 0
    stddev = 5
    for i in range(image.width):
        for j in range(image.height):
            r, g, b = image.getpixel((i, j))
            # Increase the blue channel by 30%
            r = r + int(gauss(mean, stddev))
            g = g + int(gauss(mean, stddev))
            b = b + int(gauss(mean, stddev))
            r = max(0, min(r, 255))
            g = max(0, min(g, 255))
            b = max(0, min(b, 255))

            image.putpixel((i, j), (r, g, b))
    
    return image

if __name__ == "__main__":
    image_path = '/content/codice/datasets/sf_xs/train/37.70/@0544204.32@4173406.33@10@S@037.70683@-122.49851@TYcjxIohRl--XFaR4OgdxA@@0@@@@201910@@.jpg'
    day_image = Image.open(image_path)

    # Simulazione dell'effetto notturno 
    night_image = night(day_image)

    # Visualizzazione delle immagini
    output_path = '/content/codice/datasets/night_image.jpg'
    night_image.save(output_path)
