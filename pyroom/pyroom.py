
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import sys

sys.path.append('../utils.py')
from enum import Enum
from utils import eval_sh
from .utility.Coordinates import Coordinates
from pyfilterbank import FractionalOctaveFilterbank


class ReceiverMode(Enum):
    IDEALSH = 1
    DISCRETE = 2


class Roomsimulator:
    '''
    Roomsimulator is a simple shoebox simulator, which uses image sources
    for the early part of the response and isotropic diffuse noise for the late part

    It returns SH domain impulse responses

    Author: Nils Meyer-Kahlen, Aalto Acoustics Lab, 2022

    '''

    def __init__(self, maxIsOrder=3, maxShOrder=4, receiverMode=ReceiverMode.IDEALSH, fs=48000, c=343.0):

        self.fs = fs
        self.c = c

        self.roomSize = Coordinates([5, 7, 3.5])
        self.sourcePosition = Coordinates([0.1, 0, -.5])
        self.receiverPosition = Coordinates([0, 0, -.25])

        self.maxIsOrder = maxIsOrder

        self.rtEquation = 'eyring'

        # Reverberation time for [  125.   250.   500.  1000.  2000.  4000.  8000. 16000.] Hz
        self.rt = np.array([0.2, 0.3, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5])
        self.convertRt2Absorption()

        self.irLength_s = np.max(self.rt) * 1.5  # ir length is 1.5 times the largest reverberation time

        # self.absorptionCoefficients = np.array([0.2, 0.2, 0.1, 0.15, 0.15, 0.2, 0.3, 0.3])

        # Filter length for FIR wall filters
        self.nfilt = 2048

        # Potentially randomize image source positions to avoid pitch glide effect
        self.randomizeIsPositions = False
        self.randomShiftStd = 0

        # isotropic (but diffuse) Late reverb
        self.lateReverb = True
        self.mixingTimeMode = 'automatic'
        self.mixingTime = 0.03  # When to blend over to the late reverb in s

        self.maxShOrder = maxShOrder

        self.receiverMode = receiverMode

        self.usePyfilterbank = False  # if false, use impule responses

        self.isList = []

        self.alignDirectSoundToStart = False

    def prepareImageSource(self):
        print("... preparing image sources")

        for xOrder in range(-self.maxIsOrder, self.maxIsOrder + 1):
            for yOrder in range(-self.maxIsOrder, self.maxIsOrder + 1):
                for zOrder in range(-self.maxIsOrder, self.maxIsOrder + 1):
                    isOrder = abs(xOrder) + abs(yOrder) + abs(zOrder)
                    if isOrder <= self.maxIsOrder:
                        self.isList.append((xOrder, yOrder, zOrder, isOrder))

        self.numIs = len(self.isList)

    def prepareWallFilter(self):
        '''
        prepareWallFilters creates a filter impulse response for each wall
        based on the absorption coefficients in octave bands

        '''

        print("... preparing wall filter impulse responses")

        reflectionCoefficients = np.sqrt(1 - self.absorptionCoefficients)

        if self.usePyfilterbank:

            filterbank = FractionalOctaveFilterbank(self.fs, order = 1, nth_oct = 1.0, start_band = -3, end_band = 4)

            filterIrs, states = filterbank.filter(np.concatenate([np.array([1]), np.zeros(self.nfilt - 1)]))
            self.filtersIrs = np.array(filterIrs)

        else:
            path = os.path.join(os.path.dirname(__file__), 'utility/octaveFilters.json')
            with open(path) as f:
                octaveFilters = json.load(f)
                self.filtersIrs = np.asarray(octaveFilters['ir'])

        filterPerOrder = [np.empty(self.nfilt * i - i) for i in range(self.maxIsOrder + 1)]
        filterPrototype = np.inner(self.filtersIrs, reflectionCoefficients)

        # filterPrototype = filterPrototype / np.sqrt(np.sum(filterPrototype**2))

        filterPerOrder[0] = np.array([1])

        for order in range(1, self.maxIsOrder + 1):
            filterPerOrder[order] = np.convolve(filterPrototype, filterPerOrder[order - 1])

        self.filterPerOrder = filterPerOrder

    def addIsotropicLateReverb(self, mixingTime):
        '''
        addIsotropicLateReverb add a diffuse reverberant tail

        '''
        print("... adding isotropic late reverb")

        timeConstants = self.rt * self.fs / 6.9078

        path = os.path.join(os.path.dirname(__file__), 'utility/tdes8.json')
        with open(path) as f:
            noise_source_positions = json.load(f)
            noise_source_azi = noise_source_positions["azi"]
            noise_source_zen = noise_source_positions["zen"]

        noise_noise_ele = [np.pi / 2 - zen for zen in noise_source_zen]

        numNoiseSources = len(noise_source_azi)

        noiseSourceAziEle = np.array([noise_source_azi, [np.pi / 2 - zen for zen in noise_source_zen]])
        noiseSourceAziEle = noiseSourceAziEle.transpose()

        y = eval_sh(self.maxShOrder, noiseSourceAziEle)

        irLength_samp = int(self.irLength_s * self.fs)
        noise = np.random.normal(0, 1, size = (irLength_samp, numNoiseSources))

        numFilterBands = self.filtersIrs.shape[1]
        noisePerSourceBand = np.zeros((irLength_samp, numNoiseSources, numFilterBands))

        tax = np.arange(irLength_samp)

        mixingTime_samp = int(mixingTime * self.fs)
        fadeLength_samp = 600

        window = np.hstack((np.zeros(mixingTime_samp),
                            np.sin(np.arange(fadeLength_samp) / fadeLength_samp * np.pi / 2) ** 2,
                            np.ones(irLength_samp - mixingTime_samp - fadeLength_samp)))

        for iBand in range(numFilterBands):
            envelope = np.exp(-tax / timeConstants[iBand])
            for iNoiseSource in range(numNoiseSources):
                noisePerSourceBand[:, iNoiseSource, iBand] = np.multiply(
                    np.convolve(self.filtersIrs[:, iBand], noise[:, iNoiseSource], mode = 'same'), envelope)

        noisePerSource = np.sum(noisePerSourceBand, axis = 2)

        noiseSh = np.zeros((irLength_samp, self.numShChannels))

        if (self.receiverMode == ReceiverMode.IDEALSH):
            noiseSh = np.matmul(noisePerSource, y)

        print("... fitting the parts together")

        # compute the rms of the simulated response and match the noise response
        # (needs sufficient IS order to deliver reasonable result)

        rmsIs = np.sqrt(np.sum(self.srir[mixingTime_samp:, 0] ** 2))
        rmsNoise = np.sqrt(np.sum(noiseSh[mixingTime_samp:, 0] ** 2))

        noiseShLate = np.multiply(noiseSh, np.tile(np.expand_dims(window, axis = 1),
                                                   (1, (self.maxShOrder + 1) ** 2))) * rmsIs / rmsNoise

        self.srir = self.srir + noiseShLate

    def simulate(self):
        '''
        simulate runs the IS simulation

        '''

        assert (self.sourcePosition.cart <= self.roomSize.cart / 2).any(), 'source is outside of the room'
        assert (self.receiverPosition.cart <= self.roomSize.cart / 2).any(), 'receiver is outside of the room'

        self.numShChannels = (self.maxShOrder + 1) ** 2

        self.convertRt2Absorption()

        if self.mixingTimeMode == 'automatic':
            self.mixingTime = self.getMixingTimeEstimateFromVolume()
            print("Mixing Time = " + str(self.mixingTime))

        self.isPositions = [np.empty(3) for i in range(self.numIs)]
        self.distance = [0 for i in range(self.numIs)]

        # Computing position and distance based on source and receiver
        for iIs in range(self.numIs):
            isPositionCartesian = np.multiply(self.isList[iIs][0:3], self.roomSize.cart) - np.multiply(
                (np.mod(self.isList[iIs][0:3], 2) * 2.0 - 1.0), self.sourcePosition.cart)

            relativeIsPositionCartesian = isPositionCartesian - self.receiverPosition.cart

            self.isPositions[iIs] = Coordinates(relativeIsPositionCartesian)

            if self.randomizeIsPositions:
                self.isPositions[iIs].x = self.isPositions[iIs].x + np.random.randn(1) * self.randomShiftStd
                self.isPositions[iIs].y = self.isPositions[iIs].y + np.random.randn(1) * self.randomShiftStd
                self.isPositions[iIs].z = self.isPositions[iIs].z + np.random.randn(1) * self.randomShiftStd

        rir = np.zeros(int(self.irLength_s * self.fs))
        srir = np.zeros((int(self.irLength_s * self.fs), (self.maxShOrder + 1) ** 2))

        # direct sound index
        for iIs in range(0, self.numIs):
            order = self.isList[iIs][3]
            # print(order)
            if order == 0:
                idxDirectSound = int(np.ceil(self.isPositions[iIs].r / self.c * self.fs))

        # assembling the rir
        for iIs in range(0, self.numIs):

            if self.alignDirectSoundToStart:
                idx = int(np.ceil(self.isPositions[iIs].r / self.c * self.fs)) - idxDirectSound
            else:
                idx = int(np.ceil(self.isPositions[iIs].r / self.c * self.fs))

            assert idx >= 0, "A reflections arrives before the direct sound, check geometry"

            filterLen = len(self.filterPerOrder[self.isList[iIs][3]])

            if self.isList[iIs][3] == 0:
                sign = 1
            else:
                sign = 1 - 2 * np.random.randint(0, 1)

            attenuation = sign * np.max((1.0 / self.isPositions[iIs].r, 1.0))
            # (f'order of image source {self.isList[iIs][3]}, distance {self.isPositions[iIs].r}')

            if (self.receiverMode == ReceiverMode.IDEALSH):
                encoding = eval_sh(self.maxShOrder, np.array([self.isPositions[iIs].azi, self.isPositions[iIs].ele]))
                shfilter = np.outer(np.expand_dims(self.filterPerOrder[self.isList[iIs][3]], axis = 1), encoding)

            srir[idx:idx + filterLen, :] = srir[idx:idx + filterLen, :] + attenuation * shfilter[0:filterLen, :]

            self.srir = srir

            srirOut = srir

        if self.lateReverb:
            self.addIsotropicLateReverb(self.mixingTime)

            srirOut = self.srir

        return srirOut

    def plotImageSources(self):

        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        x = np.array([pos.x for pos in self.isPositions])
        y = np.array([pos.y for pos in self.isPositions])
        z = np.array([pos.z for pos in self.isPositions])

        ax.scatter3D(self.sourcePosition.x, self.sourcePosition.y, self.sourcePosition.z, color = "green")
        ax.scatter3D(self.receiverPosition.x, self.receiverPosition.y, self.receiverPosition.z, color = "red")

        ax.scatter3D(x, y, z, color = "black")
        plt.savefig('imagesources.png')

    def plotWallFilters(self):

        for iOrder in range(1, len(self.filterPerOrder) - 1):
            nfft = 2 ** np.ceil(np.log2(self.filterPerOrder[iOrder].shape[0]))
            X = np.fft.fft(self.filterPerOrder[iOrder])[0: int(nfft / 2.0)]
            fax = np.arange(0, self.fs / 2.0, self.fs / nfft)
            plt.semilogx(fax, 20 * np.log10(np.abs(X)))
            plt.xlim([80, self.fs / 2])
            plt.ylim([-80, 40])

            plt.savefig('wallfilter.png')

    def convertAbsorptionToRt(self):
        if (self.rtEquation == 'sabine'):
            V = np.prod(self.roomSize)
            S = 2 * self.roomSize[0] * self.roomSize[1] + 2 * self.roomSize[0] * self.roomSize[2] + 2 * self.roomSize[
                1] * self.roomSize[2]

            if (self.rtEquation == 'sabine'):
                self.rt = 0.161 * V / (S * self.absorptionCoefficients)

            elif (self.rtEquation == 'eyring'):
                self.rt = -0.163 * V / (S * np.log(1 - self.absorptionCoefficients))

    def convertRt2Absorption(self):
        V = np.prod(self.roomSize.cart)
        S = 2 * (self.roomSize.x * self.roomSize.y +
                 self.roomSize.x * self.roomSize.z +
                 self.roomSize.y * self.roomSize.z)

        if (self.rtEquation == 'sabine'):
            self.absorptionCoefficients = 0.161 * V / (S * self.rt)

        elif (self.rtEquation == 'eyring'):
            self.absorptionCoefficients = 1 - np.exp(-0.163 * V / (S * self.rt))

    def getMixingTimeEstimateFromVolume(self):
        # Cremer, L.; Mueller, H. A. (1978): Die wissenschaftlichen Grundlagen der Raumakustik
        V = np.prod(self.roomSize.cart)
        return 2 * np.sqrt(V) / 1000.0
