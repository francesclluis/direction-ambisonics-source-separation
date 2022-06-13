import numpy as np

class Coordinates:

    def __init__(self, input, type='cartesian'):

        input = np.asarray(input)

        if type == 'cartesian':
            self.cart = input

        if type == 'sphericalDeg':
            self.cart[0] = np.cos(input[0]) * np.sin(input[1])
            self.cart[1] = np.sin(input[0]) * np.sin(input[1])
            self.cart[2] = np.cos(input[1])

        self.x = self.cart[0]
        self.y = self.cart[1]
        self.z = self.cart[2]

        self.cartNorm = self.cart / np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

        self.azi = np.arctan2(self.y, self.x)
        self.r = np.sqrt(np.sum(self.cart ** 2))
        self.zen = np.arccos(self.z / self.r)
        self.ele = np.pi / 2 - self.zen

        self.aziEle = np.hstack((self.azi, self.ele))
        self.aziZen = np.hstack((self.azi, self.zen))

    def __str__(self):
        return "azi=" + str(self.azi * 180 / np.pi) + " ele=" + str(self.ele * 180 / np.pi) + " r= " + str(self.r)

    def greatCircleDistanceTo(self, c2):

        if (self.cartNorm == c2.cartNorm).all():
            return 0

        else:
            phi = np.arccos(np.inner(self.cartNorm, c2.cartNorm))
            return phi
