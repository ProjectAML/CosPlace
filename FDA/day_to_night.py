from PIL import Image, ImageEnhance
from deprecated_scipy import toimage

def convert_temp(image):
    r, g, b = (180,219,255)
    matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0 )
    return image.convert('RGB', matrix)

def simulate_night(input_img):
  img = input_img
  #lower saturation
  saturation = ImageEnhance.Color(img)
  img = saturation.enhance(0.7)
  #blue tint
  img = convert_temp(img)
  toimage(img, cmin=0.0, cmax=255.0)

  return img
