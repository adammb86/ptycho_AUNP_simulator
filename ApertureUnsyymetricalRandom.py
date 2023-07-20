import random
import numpy as np
from matplotlib.path import Path
from skimage.draw import polygon
import matplotlib.pyplot as plt

class Opt:
    def __init__(self, dx, Npixel):
        self.dx = dx
        self.Npixel = Npixel

class Aper:
    def __init__(self, sideNumber, sideLength):
        self.sideNumber = sideNumber
        self.sideLengths = [sideLength * random.uniform(0.5, 1.5) for _ in range(sideNumber)]  # Now a list of side lengths

def calcAperture(opt, aper):
    edgeUnit = 2 * np.sin(np.pi / aper.sideNumber)
    xMags = [length / opt.dx / edgeUnit for length in aper.sideLengths]
    yMags = [length / opt.dx / edgeUnit for length in aper.sideLengths]
    aperShape = Path(np.array([(np.sin(2 * np.pi * ii / aper.sideNumber), np.cos(
        2 * np.pi * ii / aper.sideNumber)) for ii in range(aper.sideNumber)]))
    xPOS = [(aperShape.vertices[ii, 0]) * xMags[ii] + round(opt.Npixel / 2) for ii in range(aper.sideNumber)]
    yPOS = [(aperShape.vertices[ii, 1]) * yMags[ii] + round(opt.Npixel / 2) for ii in range(aper.sideNumber)]

    aperture = np.zeros((opt.Npixel, opt.Npixel), 'float')
    xFill, yFill = polygon(xPOS, yPOS, aperture.shape)
    aperture[xFill, yFill] = 1

    return aperture

# Test the function
opt = Opt(dx=0.1, Npixel=300)
aper = Aper(sideNumber=5, sideLength=5)
print(aper.sideLengths)
aperture = calcAperture(opt, aper)
plt.imshow(np.abs(aperture))
plt.show()