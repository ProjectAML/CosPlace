from PIL import Image, ImageEnhance, ImageFilter,ImageDraw,ImageChops
from random import gauss
import numpy as np


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

def generate_and_apply_gradient(image, factorMin, factorMax):
    """ Add a gradient to the image """
    # Create a new image with a black to transparent vertical gradient
    gradient = Image.new("RGBA", (image.width, image.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)
    for y in range(image.height):
        # Set the transparency of the pixel based on its position in the image
        # The top of the image will be fully opaque (255), and the bottom will be fully transparent (0)
        transparency = int(y / (image.height*1.2) * 255)
        draw.line((0, y, image.width, y), fill=(0, 0, 0, transparency))

    #rotate the gradient
    gradient = gradient.rotate(180)

    # Paste the gradient onto the image
    image.paste(gradient, (0, 0), gradient)

    return image

def simulate_night(image):
    enhancer = ImageEnhance.Brightness(image)
    factor = np.random.uniform(0.8, 0.9)
    image = enhancer.enhance(factor)

    enhancer = ImageEnhance.Color(image)
    factor = np.random.uniform(0.9, 0.95)
    image = enhancer.enhance(factor)

    factor = np.random.uniform(1.05, 1.15)
    for i in range(image.width):
        for j in range(image.height):
            r, g, b = image.getpixel((i, j))
            # Increase the blue channel by 30%
            image.putpixel((i, j), (r, g, int(b * factor)))
    
    enhancer = ImageEnhance.Contrast(image)
    factor = np.random.uniform(1.05, 1.1)
    image = enhancer.enhance(factor)
 
    width, height = image.size
    mask_color = (0, 0, 0, mask_intensity)
    mask = Image.new('RGBA', (width, height), mask_color)

    # Applicazione della maschera all'immagine
    image = Image.alpha_composite(image, mask)
 
    image = generate_and_apply_gradient(image, gradient_min, gradient_max)
    image = add_gaussian_noise(image)
    
    return image



