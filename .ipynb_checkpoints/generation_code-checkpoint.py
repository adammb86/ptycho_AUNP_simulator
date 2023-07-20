#!/home/adam/anaconda3/bin/python3

import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv
import numpy as np
import usrLibrary as lib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

class optical_condition:
    L:      float # distance from the sample plane to the detector plane
    p:      float # pixel size of the detector
    Npixel: int   # window size of the probe function, object functions, and diffraction patterns
    energy: float # x-ray energy of the incident beam
    h:      float # plank constant
    c:      float # velocity of the light
    lamb:   float # wavelength
    k:      float # wavenumber
    dx:     float # pixel resolution at the sample plane
    DR:     float # dynamic range of the detector, for example opt.DR = log10{1e7 (photons) / 1 (photon}
    Wnoise: bool  # flag to apply photon shot noise to each calculated diffraction pattern


opt         = optical_condition()

opt.L       = 3                                         # unit: m
opt.p       = 75e-6                                     # unit: m
opt.Npixel  = 1024                                      # unit: pixel
opt.energy  = 5000                                      # unit: eV (electronVolt)
opt.h       = 6.62607015e-34                            # unit: m^2 kg/s
opt.c       = 299792458                                 # unit: m/s
opt.lamb    = (opt.h * 6.242e18) * opt.c / opt.energy   # unit: m
opt.k       = 2 * np.pi / opt.lamb                      # unit: 1/m
opt.dx      = opt.L * opt.lamb / (opt.Npixel * opt.p)   # unit: m/pixel
opt.DR      = 7
opt.Wnoise  = True

class aperture_condition:
    sideNumber:         int     # number of the side
    sideLength:         float   # length of a side
    Lpropagation:       float   # distance from the aperture to the sample
    needPropagation:    bool    # flag whether propagation from the aperture to the sample is needed

aper = aperture_condition()

aper.sideNumber     = 3     # input: 3 --> Triangle
aper.sideLength     = 5e-6  # unit: m
aper.Lpropagation   = 5e-4     # unit: m, range: # 500-1000*10^-6

if aper.Lpropagation > 0:
    aper.needPropagation = True
else:
    aper.needPropagation = False

class particle_condition:
    beta_Au:    float # parameter related to the absorption of the wavefield
    delta_Au:   float # parameter related to the phase shift of the wavefield
    diameter:   float # diameter of the gold nanoparticle
    ratio:      float # (Area occupied with gold nanoparticles) / (Area of the window, Npixel x Npixel)
    number:     int   # total number of the gold nanoparticles within the window
    velocity:   float # displacement of the gold nanoparticles (per frame)

par = particle_condition()

par.beta_Au     = 2.633e-5
par.delta_Au    = 1.2143e-4
par.diameter    = 150e-9 #300e-9 # unit: m
par.ratio       = 100e-3
par.number      = int(np.floor((opt.Npixel * opt.dx)**2 / (np.pi * (par.diameter / 2)**2) * par.ratio))
par.Nframe      = 11
par.velocity    = 1 * opt.dx    # unit: m/frame

def generate_ptycho_array(num_loop):
    # set different seed per n frame
    np.random.seed(num_loop)

    aperture = lib.calcAperture(opt, aper)
    smooth_aperture = cv.GaussianBlur(aperture, (9,9), 0)
    if aper.needPropagation == 1:
        inwave = lib.angularSpectrum(smooth_aperture, opt, aper)
    else:
        inwave = aperture

    AuNPimg = lib.calcAuNP(opt, par)

    wavesample = AuNPimg * np.tile(inwave[:, :, np.newaxis], (1, 1, par.Nframe))
    wavedetector = np.fft.fftshift(np.fft.fft2(wavesample, axes = (0, 1)), axes = (0, 1))

    difImg       = np.abs(wavedetector)**2

    # diffraction patterns are normalized so that its dynamic range become "opt.DR"
    maxCount     = difImg.max()
    difImg       = difImg / maxCount * 10**(opt.DR)
    isCounted    = np.where(difImg >= 1, 1, 0)
    difImg       = difImg * isCounted # pixels which count lower than 1 photon is set to 0

    # 2-4-3, apply photon shot noise if "opt.Wnoise" is True
    if opt.Wnoise == 1:
        difImgWnoise = np.random.poisson(difImg)

    frame_sum = np.sum(AuNPimg,-1,keepdims=True)
    wavesample_sum = np.sum(difImg,-1,keepdims=True)
    
    arr_stack = np.stack((np.array(np.abs(AuNPimg[:,:,0])),
                      np.array(np.abs(AuNPimg[:,:,5])),
                      np.abs(frame_sum).squeeze(),
                      np.array(np.angle(AuNPimg[:,:,0])),
                      np.array(np.angle(AuNPimg[:,:,5])),
                      np.angle(frame_sum).squeeze(),
                      np.abs(wavesample_sum).squeeze()
                    ))

    filename = 'data/ptychoAUNP_'+str(num_loop)
    np.savez_compressed(filename,arr_stack)
    
    return True

def main():
    with ProcessPoolExecutor(8) as executor:
        future = []
        for i in range(1000):
            np.random.seed(i)
            future.append(executor.submit(generate_ptycho_array, i))

        for i, f in enumerate(as_completed(future)):
            _ = f.result()
            if i % 8 == 0:
                print('Finish data for seed ', i)
            
        future.clear()

if __name__ == "__main__":
    main()