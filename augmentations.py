
import torch
from typing import Tuple, Union
import torchvision.transforms as T
import random
import torchsample as TS
from PIL import Image


class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(self, brightness: float = 0., contrast: float = 0., saturation: float = 0., hue: float = 0.):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        color_jitter = super(DeviceAgnosticColorJitter, self).forward
        augmented_images = [color_jitter(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images


class DeviceAgnosticRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size: Union[int, Tuple[int, int]], scale: float):
        """This is the same as T.RandomResizedCrop but it only accepts batches of images and works on GPU"""
        super().__init__(size=size, scale=scale)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        random_resized_crop = super(DeviceAgnosticRandomResizedCrop, self).forward
        augmented_images = [random_resized_crop(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images

class DeviceAgnosticBrightness():
    def __init__(self,target_brightness=0.6):
        self.target_brightness=target_brightness
    
    def __call__(self,images: torch.Tensor)->torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        offset=random.uniform(-0.05,0.05)
        augmented_images=[]
        for img in images:
            if random.random()<0.5:
                augmented_img=T.functional.adjust_brightness(img, self.target_brightness+offset).unsqueeze(0)
                augmented_images.append(augmented_img)
            else:
                augmented_images.append(img.unsqueeze(0))
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images

class DeviceAgnostiContrast():
    def __init__(self,target_contrast=1.15):
        self.target_contrast=target_contrast
    
    def __call__(self,images: torch.Tensor)->torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        offset=random.uniform(-0.05,0.05)
        augmented_images=[]
        for img in images:
            if random.random()<0.5:
                augmented_img=T.functional.adjust_contrast(img, self.target_contrast+offset).unsqueeze(0)
                augmented_images.append(augmented_img)
            else:
                augmented_images.append(img.unsqueeze(0))
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images

class DeviceAgnostiSaturation():
    def __init__(self,target_saturation=0.8):
        self.target_saturation=target_saturation
    
    def __call__(self,images: torch.Tensor)->torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        offset=random.uniform(-0.05,0.05)
        augmented_images=[]
        for img in images:
            if random.random()<0.5:
                augmented_img=T.functional.adjust_saturation(img, self.target_saturation+offset).unsqueeze(0)
                augmented_images.append(augmented_img)
            else:
                augmented_images.append(img.unsqueeze(0))
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images


if __name__ == "__main__":
    """
    You can run this script to visualize the transformations, and verify that
    the augmentations are applied individually on each image of the batch.
    """
    import os
    from PIL import Image
    # Import skimage in here, so it is not necessary to install it unless you run this script
    from skimage import data
    
    # Initialize DeviceAgnosticRandomResizedCrop
    augs = DeviceAgnostiSaturation()
    # Create a batch with 2 astronaut images
    pil_image = Image.fromarray(data.astronaut())
    tensor_image = T.functional.to_tensor(pil_image).unsqueeze(0)
    images_batch = torch.cat([tensor_image, tensor_image])
    # Apply augmentation (individually on each of the 2 images)
    augmented_batch = augs.forward(images_batch)
    # Convert to PIL images
    augmented_image_0 = T.functional.to_pil_image(augmented_batch[0])
    augmented_image_1 = T.functional.to_pil_image(augmented_batch[1])
    # Visualize the original image, as well as the two augmented ones
    #pil_image.show()
    #augmented_image_0.show()
    #augmented_image_1.show()

    pil_image.save("original_image.jpg")
    augmented_image_0.save("augmented_image_0.jpg")
    augmented_image_1.save("augmented_image_1.jpg")

    print("Original Image:", os.path.abspath("original_image.jpg"))
    print("Augmented Image 0:", os.path.abspath("augmented_image_0.jpg"))
    print("Augmented Image 1:", os.path.abspath("augmented_image_1.jpg"))


