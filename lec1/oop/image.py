import cv2
import numpy as np
from multipledispatch import dispatch

from oop.window import Window, WindowFlags
from oop.imageFormat import ImageFormat
from oop.plotter import Plotter

class Image:
    @staticmethod
    def new(width: int, height: int, layers: int, format: ImageFormat)-> 'Image':
        img = Image(np.zeros(height, width, layers), np.uint8)
        img.format = format
        return img

    
    @staticmethod
    def from_file(path: str, color: bool = True)-> 'Image': 
        img = Image(cv2.imread(path, 1 if color else 0))
        img.format = ImageFormat.BGR if color else ImageFormat.GRAY
        return img

    @staticmethod
    def from_image(image: 'Image')-> 'Image':
        img = Image(np.copy(image.img))
        img.format = image.format
        return img

    def __init__(self, image: cv2.typing.MatLike, format: ImageFormat = ImageFormat.BGR):
        self.img = image
        self.format = format

    def height(self) -> int:
        return self.img.shape[0]

    def width(self) -> int:
        return self.img.shape[1]

    def layers(self)-> int:
        return self.img.shape[2] if len(self.img.shape) > 2 else 1
    
    def is_color(self)-> bool:
        return self.layers > 3
    
    @dispatch(str)
    def show(self, window: str)-> None:
        cv2.imshow(window, self.img)

    @dispatch(Window)
    def show(self, window: Window)-> None:
        cv2.imshow(window.name, self.img)
        window.resize(self.width(), self.height())

    @dispatch(Plotter, int, int)
    def show(self, plotter: Plotter, row: int, col: int)-> None:
        plotter.set(self.copy().convert_format(ImageFormat.RGB), row, col)

    @dispatch(str)
    def write(self, path: str)-> None:
        cv2.imwrite(path, self.img)

    @dispatch(cv2.VideoWriter)
    def write(self, writer: cv2.VideoWriter)-> None:
        writer.write(self.convert_format(ImageFormat.RGB).img)

    def copy(self) -> 'Image':
        return Image.from_image(self)

    def resize(self, width: int, height: int)-> 'Image':
        self.img = cv2.resize(self.img, (width, height))
        return self
    
    def crop(self, x: int, y: int, width: int, height: int)-> 'Image':
        self.img = self.img[y:y+height, x:x+width]
        return self

    def convert_format(self, fmt: ImageFormat)-> 'Image':
        match fmt:
            case ImageFormat.BGR:
                match self.format:
                    case ImageFormat.RGB:
                        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                    case ImageFormat.GRAY:
                        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            case ImageFormat.RGB:
                match self.format:
                    case ImageFormat.BGR:
                        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                    case ImageFormat.GRAY:
                        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
            case ImageFormat.GRAY:
                match self.format:
                    case ImageFormat.BGR:
                        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
                    case ImageFormat.RGB:
                        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.format = fmt
        return self
    
    @staticmethod
    def hconcat(images: list['Image'])-> 'Image':
        final_format = images[0].format
        max_width = max(image.width() for image in images)
        max_height = max(image.height() for image in images)

        for image in images:
            if image.format.value > final_format.value:
                final_format = image.format

        for i, image in enumerate(images):
            if image.format != final_format:
                images[i] = image.resize_and_pad(max_width, max_height).convert_format(final_format)

        return Image(cv2.hconcat([image.img for image in images]))
    
    @staticmethod
    def vconcat(images: list['Image'])-> 'Image':
        final_format = images[0].format
        max_width = max(image.width() for image in images)
        max_height = max(image.height() for image in images)

        for image in images:
            if image.format.value > final_format.value:
                final_format = image.format

        for i, image in enumerate(images):
            if image.height() < max_height & image.width() < max_width:
                images[i] = image.resize_and_pad(max_width, max_height)

            if image.format != final_format:
                img = Image.from_image(image)
                images[i] = img.convert_format(final_format)

        return Image(cv2.vconcat([image.img for image in images]))

    def resize_and_pad(self, width, height)-> 'Image':
        result = self.copy()

        # Get the original image size
        original_height = self.height()
        original_width = self.width()

        # Calculate the scaling factors for width and height
        scale_width = width / original_width
        scale_height = height / original_height

        # Choose the minimum scaling factor to maintain aspect ratio
        scale = min(scale_width, scale_height)

        # Resize the image
        resized_image = cv2.resize(self.img, (int(original_width * scale), int(original_height * scale)))

        # Calculate the borders to add
        pad_x = (width - resized_image.shape[1]) // 2
        pad_y = (height - resized_image.shape[0]) // 2

        # Create a white canvas
        print(self.img.dtype)
        if self.layers() == 1:
            padded_image = np.ones((height, width), dtype=np.uint8)
        else:
            padded_image = np.ones((height, width, self.layers()), dtype=np.uint8)

        # Place the resized image in the center of the canvas
        padded_image[pad_y:pad_y + resized_image.shape[0], pad_x:pad_x + resized_image.shape[1]] = resized_image

        result.img = padded_image
        return result
    

    def canny(self, low: int, high: int)-> 'Image':
        self.img = cv2.Canny(self.img, low, high)