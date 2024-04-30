import cv2
import numpy as np
import math
from multipledispatch import dispatch

from oop.window import Window, WindowFlags
from oop.imageFormat import ImageFormat
from oop.plotter import Plotter

class Image:
    @staticmethod
    def new(width: int, height: int, layers: int, format: ImageFormat)-> 'Image':
        img = Image(np.zeros((height, width, layers), np.uint8))
        img.format = format
        return img

    @staticmethod
    def zeros(width: int, height: int, layers: int, format: ImageFormat)-> 'Image':
        img = Image(np.zeros((height, width, layers), np.uint8))
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

    @staticmethod
    def from_object(array, format)-> 'Image':
        img = Image(np.array(array))
        img.format = format
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
    def hconcat(images: list['Image'], target: str | tuple = "max")-> 'Image':
        final_format = images[0].format

        if type(target) is tuple:
            target_width, target_height = target
        elif target == "max":
            target_width = max(image.width() for image in images)
            target_height = max(image.height() for image in images)
        elif target == "min":
            target_width = min(image.width() for image in images)
            target_height = min(image.height() for image in images)
        elif target == "average":
            target_width = sum(image.width() for image in images) // len(images)
            target_height = sum(image.height() for image in images) // len(images)
        else:
            raise ValueError("Target must be 'max', 'min' or 'average' or tuple")

        target_width = max(image.width() for image in images)
        target_height = max(image.height() for image in images)

        for image in images:
            if image.format.value > final_format.value:
                final_format = image.format

        for i, image in enumerate(images):
            if image.img.dtype != np.uint8:
                images[i] = image.to_dtype_uint8 

            if image.format != final_format:
                images[i] = image.resize_and_pad(target_width, target_height).convert_format(final_format)

            if image.format != final_format:
                img = Image.from_image(image)
                images[i] = img.convert_format(final_format)

        return Image(cv2.hconcat([image.img for image in images]), final_format)
    
    @staticmethod
    def vconcat(images: list['Image'], target: str | tuple = "max")-> 'Image':
        final_format = images[0].format

        if type(target) is tuple:
            target_width, target_height = target
        elif target == "max":
            target_width = max(image.width() for image in images)
            target_height = max(image.height() for image in images)
        elif target == "min":
            target_width = min(image.width() for image in images)
            target_height = min(image.height() for image in images)
        elif target == "average":
            target_width = sum(image.width() for image in images) // len(images)
            target_height = sum(image.height() for image in images) // len(images)
        else:
            raise ValueError("Target must be 'max', 'min' or 'average' or tuple")

        for image in images:
            if image.format.value > final_format.value:
                final_format = image.format

        for i, image in enumerate(images):
            if image.img.dtype != np.uint8:
                images[i] = image.to_dtype_uint8()

            if image.height() != target_height & image.width() != target_width:
                images[i] = image.resize_and_pad(target_width, target_height)

            if image.format != final_format:
                img = Image.from_image(image)
                images[i] = img.convert_format(final_format)
        return Image(cv2.vconcat([image.img for image in images]), final_format)

    @staticmethod
    def show_grid(images: list['Image'], target: str | tuple = "max", prefer = "column")-> 'Image':
        length = len(images)

        sqrt = math.sqrt(length)
        if sqrt == round(sqrt): # perfect grid
            rows = int(sqrt)
            cols = int(sqrt)
            rows = []
            for i in range(0, length, cols):
                rows.append(Image.hconcat(images[i:i+cols], target))
            return Image.vconcat(rows, target)
        elif prefer == "grid":
            images.append(Image.new(images[0].width(), images[0].height(), images[0].layers(), images[0].format))
            return Image.show_grid(images, target, prefer)


            


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
        padded_image = cv2.copyMakeBorder(resized_image, pad_y, height - resized_image.shape[0] - pad_y, pad_x, width - resized_image.shape[1] - pad_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        result.img = padded_image
        return result
    

    def canny(self, low: int, high: int)-> 'Image':
        self.img = cv2.Canny(self.img, low, high)

    def matchTemplate(self, template: 'Image', method: int)-> 'Image':
        result = cv2.matchTemplate(self.img, template.img, method)
        return Image.from_object(result, ImageFormat.GRAY)
    
    def to_dtype_uint8(self)-> 'Image':
        result = ((self.img + 1) * 255 / 2).astype(np.uint8)
        # result = cv2.normalize(self.img, None, 0, 255, cv2.NORM_MINMAX)
        self.img = result.astype(np.uint8)
        return self
    
    def threshold(self, a,b, format)-> 'Image':
        ret, image = cv2.threshold(self.img, a, b, format)
        self.img = image
        self.ret = ret
        return self
    
    def dilate(self, kernel)-> 'Image':
        self.img = cv2.dilate(self.img, kernel)
        return self
    
    def erode(self, kernel)-> 'Image':
        self.img = cv2.erode(self.img, kernel)
        return self
    
    def findContours(self, mode: int, method: int)-> list:
        contours, _ = cv2.findContours(self.img, mode, method)
        return contours
    
    def drawContours(self, contours: list, negat, color: tuple, thickness: int)-> 'Image':
        self.img = cv2.drawContours(self.img, contours, negat, color, thickness)
        return self
    
