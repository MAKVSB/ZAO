from matplotlib import pyplot as plt

from oop.imageFormat import ImageFormat

class Plotter:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
    def set(self, image, row: int, col: int):
        plt.subplot(self.width, self.height, row * self.width + col + 1)
        plt.imshow(image.copy().convert_format(ImageFormat.RGB).img)
    def show(self):
        plt.show()