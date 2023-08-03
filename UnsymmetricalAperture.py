import cv2
import numpy as np
import matplotlib.pyplot as plt

class Opt:
    def __init__(self, dx, Npixel):
        self.dx = dx
        self.Npixel = Npixel

class Aper:
    def __init__(self, sideNumber, sideLengths):
        self.sideNumber = sideNumber
        self.sideLengths = sideLengths  # Now a list of side lengths

def calcAperture(opt, aper):
    edgeUnit = 2 * np.sin(np.pi / aper.sideNumber)
    xMags = [length / opt.dx / edgeUnit for length in aper.sideLengths]
    yMags = [length / opt.dx / edgeUnit for length in aper.sideLengths]
    aperShape = np.array([(np.sin(2 * np.pi * ii / aper.sideNumber), np.cos(
        2 * np.pi * ii / aper.sideNumber)) for ii in range(aper.sideNumber)])
    xPOS = [(aperShape[ii, 0]) * xMags[ii] + round(opt.Npixel / 2) for ii in range(aper.sideNumber)]
    yPOS = [(aperShape[ii, 1]) * yMags[ii] + round(opt.Npixel / 2) for ii in range(aper.sideNumber)]

    aperture = np.zeros((opt.Npixel, opt.Npixel), 'float')
    points = np.array([[[int(x), int(y)]] for x, y in zip(xPOS, yPOS)])
    cv2.fillPoly(aperture, [points], 1)

    return aperture

def calcRectangleAperture(opt, aper):
    edgeUnit = 2 * np.sin(np.pi / 4)
    xMags = [aper.width / opt.dx / edgeUnit] * 4
    yMags = [aper.height / opt.dx / edgeUnit] * 4
    aperShape = Path(np.array([(np.sin(2 * np.pi * ii / 4), np.cos(2 * np.pi * ii / 4)) for ii in range(4)]))
    xPOS = [(aperShape.vertices[ii, 0]) * xMags[ii] + round(opt.Npixel / 2) for ii in range(4)]
    yPOS = [(aperShape.vertices[ii, 1]) * yMags[ii] + round(opt.Npixel / 2) for ii in range(4)]

    aperture = np.zeros((opt.Npixel, opt.Npixel), 'float')
    xStart, yStart, xEnd, yEnd = int(min(xPOS)), int(min(yPOS)), int(max(xPOS)), int(max(yPOS))
    rr, cc = rectangle(start=(xStart, yStart), extent=(xEnd - xStart, yEnd - yStart))
    aperture[rr, cc] = 1

    return aperture
