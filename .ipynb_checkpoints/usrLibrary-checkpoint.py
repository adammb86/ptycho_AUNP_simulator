# user defined function
import numpy as np
import cv2
import tensorflow as tf
from matplotlib.path import Path
from skimage.draw import polygon


# model: 2d random walk


def calcAuNP(opt, par):

    minNpixel = np.ceil(par.diameter / opt.dx)
    if minNpixel % 2 == 1:
        minNpixel += 1

    # Define 3D coordinate
    Xw = np.arange(minNpixel) - (minNpixel / 2 - 1)
    Yw = np.arange(minNpixel) - (minNpixel / 2 - 1)
    Zw = np.arange(minNpixel) - (minNpixel / 2 - 1)
    Xw3d, Yw3d, Zw3d = np.meshgrid(Xw, Yw, Zw)
    Xw3d *= opt.dx
    Yw3d *= opt.dx
    Zw3d *= opt.dx

    # calculate binary image, inside the particle -> 1, outisde the particle -> 0
    # 3D single particle (in minNpixel x minNpixel x minNpixel)
    singleNP3dcrop = (par.diameter / 2)**2 >= (Xw3d**2 + Yw3d**2 + Zw3d**2)
    # calcuate 2d projected image
    # 2D projected single particle (in minNpixel x minNpixel)
    singleNP2dcrop = np.squeeze(opt.dx * singleNP3dcrop.sum(axis=2))
    # 2D projected single particle (in Npixel x Npixel)
    singleNP2dpad = np.pad(singleNP2dcrop, int(
        (opt.Npixel - minNpixel) / 2), "constant")

    # apply absorption and phase parameter to the 2d projected image
    # 2D projected single particle (in Npixel x Npixel)
    singleAuNPpad = np.exp(-opt.k * (par.beta_Au + 1j *
                           par.delta_Au) * singleNP2dpad)
    # 2D projected single particle (in minNpixel x minNpixel)
    singleAuNPcrop = np.exp(-opt.k * (par.beta_Au +
                            1j * par.delta_Au) * singleNP2dcrop)

    # calculate positions (1st frame)
    NPpos = np.zeros((2, int(par.number), int(par.Nframe)), dtype="float")
    for iNumber in range(par.number):  # initial position of each particle
        NPpos[0, iNumber, 0] = opt.Npixel * np.random.rand() * opt.dx
        NPpos[1, iNumber, 0] = opt.Npixel * np.random.rand() * opt.dx

    # calculate positions (from 2nd frame to Nframe-th frame)
    for iNumber in range(0, par.number):
        for iFrame in range(1, par.Nframe):
            randTheta = np.random.rand() * (2 * np.pi)
            xDelta = par.velocity * np.cos(randTheta)
            yDelta = par.velocity * np.sin(randTheta)
            NPpos[0, iNumber, iFrame] = NPpos[0, iNumber, iFrame - 1] + xDelta
            NPpos[1, iNumber, iFrame] = NPpos[1, iNumber, iFrame - 1] + yDelta

    AuNPimg = np.ones((opt.Npixel, opt.Npixel, par.Nframe), dtype="complex")
    for iFrame in range(0, par.Nframe):
        print(iFrame)
        for iNumber in range(0, par.number):
            # flags to determine the method to embed the iNumber-th particle image to the iFrame-th AuNPimg
            xEffective = np.mod(NPpos[0, iNumber, iFrame] / opt.dx, opt.Npixel)
            yEffective = np.mod(NPpos[1, iNumber, iFrame] / opt.dx, opt.Npixel)
            smallNumber = 50
            xTouchWindow = (xEffective < (singleAuNPcrop.shape[0] / 2 + smallNumber)) or (
                xEffective > (opt.Npixel - singleAuNPcrop.shape[0] / 2 - smallNumber))
            yTouchWindow = (yEffective < (singleAuNPcrop.shape[1] / 2 + smallNumber)) or (
                yEffective > (opt.Npixel - singleAuNPcrop.shape[1] / 2 - smallNumber))
            # method 1, slow but simple (shift the particle image)
            if xTouchWindow == 1 or yTouchWindow == 1:
                AuNPimg[:, :, iFrame] *= np.roll(np.roll(np.fft.fftshift(singleAuNPpad), int(np.round(
                    NPpos[0, iNumber, iFrame] / opt.dx)), axis=0), int(np.round(NPpos[1, iNumber, iFrame] / opt.dx)), axis=1)
            else:
                xStart = int(np.round(xEffective) - minNpixel / 2 + 1)
                xEnd = int(xStart + minNpixel - 1)
                yStart = int(np.round(yEffective) - minNpixel / 2 + 1)
                yEnd = int(yStart + minNpixel - 1)

                AuNPimg[(xStart-1):xEnd, (yStart-1)
                         :yEnd, iFrame] *= singleAuNPcrop

    return AuNPimg


def calcAperture(opt, aper):
    edgeUnit = 2 * np.sin(np.pi / aper.sideNumber)
    xMag = aper.sideLength / opt.dx / edgeUnit
    yMag = aper.sideLength / opt.dx / edgeUnit
    aperShape = Path(np.array([(np.sin(2 * np.pi * ii / aper.sideNumber), np.cos(
        2 * np.pi * ii / aper.sideNumber)) for ii in range(aper.sideNumber)]))
    xPOS = (aperShape.vertices[:, 0]) * xMag + round(opt.Npixel / 2)
    yPOS = (aperShape.vertices[:, 1]) * yMag + round(opt.Npixel / 2)

    aperture = np.zeros((opt.Npixel, opt.Npixel), 'float')
    xFill, yFill = polygon(xPOS, yPOS, aperture.shape)
    aperture[xFill, yFill] = 1

    return aperture


def calcAperturerounded(opt, aper, corner_radius):
    edgeUnit = 2 * np.sin(np.pi / aper.sideNumber)
    xMag = aper.sideLength / opt.dx / edgeUnit
    yMag = aper.sideLength / opt.dx / edgeUnit
    aperShape = np.array([(np.sin(2 * np.pi * ii / aper.sideNumber), np.cos(
        2 * np.pi * ii / aper.sideNumber)) for ii in range(aper.sideNumber)])
    xPOS = (aperShape[:, 0]) * xMag + round(opt.Npixel / 2)
    yPOS = (aperShape[:, 1]) * yMag + round(opt.Npixel / 2)

    aperture = np.zeros((opt.Npixel, opt.Npixel), 'float')

    # Generate the rounded corner polygon
    vertices = np.column_stack((xPOS, yPOS)).astype(np.int32)
    cv2.fillPoly(aperture, [vertices], 1, lineType=cv2.LINE_AA)

    return aperture


def angularSpectrum(aperture, opt, aper):
    if opt.Npixel % 2 == 0:
        Xw = np.arange(opt.Npixel) - (opt.Npixel / 2 - 1)
        Yw = np.arange(opt.Npixel) - (opt.Npixel / 2 - 1)
    else:
        Xw = np.arange(opt.Npixel) - (opt.Npixel - 1) / 2
        Yw = np.arange(opt.Npixel) - (opt.Npixel - 1) / 2
    Xw2d, Yw2d = np.meshgrid(Xw, Yw)
    Qx2d = Xw2d / (opt.Npixel * opt.dx)
    Qy2d = Yw2d / (opt.Npixel * opt.dx)
    w = np.emath.sqrt(opt.lamb**(-2) - Qx2d**2 - Qy2d**2)

    outwave = np.fft.fftshift(np.fft.fft2(aperture))
    outwave *= np.exp(1j * (2 * np.pi * aper.Lpropagation * w))
    outwave = np.fft.ifft2(np.fft.ifftshift(outwave))

    return outwave
