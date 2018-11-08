''' Class to hold a model of a full system
'''
import starry
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import string

from .star import Star
from .planet import Planet
from . import utils
# We should have a better dependency than CT for this.
# This is just to get the NExSci data table.

import characterizethis as ct
df = ct.data()

from lightkurve import MPLSTYLE

fit_params =  {'host':[], 'planet':['rprs', 'period', 't0', 'inclination', 'eccentricity']}

class Model(object):
    '''Combined model of a star and companions.
    '''

    @property
    def _fit_labels(self):
        l = ['A.' + s for s in fit_params['host']]
        for idx in range(self.nplanets):
            for s in fit_params['planet']:
                l.append(string.ascii_lowercase[1 + idx] + '.' + s)
        return l

    def __init__(self, host=None, planets=None):
        self.host = host
        if isinstance(planets, list):
            self.planets = planets
        elif planets is None:
            self.planets = []
        else:
            self.planets = [planets]

        if (planets is None) & (host is None):
            self.system = None
        else:
            self.system = starry.kepler.System(self.host.model, *[p.model for p in self.planets])
        self.nplanets = len(self.planets)
        l = []
        for idx in range(self.nplanets):
            for f in fit_params['planet']:
                l.append(u.Quantity(np.copy(getattr(self.planets[idx], f))).value)
        self.initial_guess = l

        for planet in self.planets:
            planet._validate()
        self.host._validate()


    def __repr__(self):
        s = self.host.__repr__()
        for p in self.planets:
            s += '\n\t{}'.format(p.__repr__())
        return s


    @staticmethod
    def from_nexsci(name):
        host = Star().from_nexsci(name)
        letters = np.asarray(df.pl_letter[df.pl_hostname==name])
        planets = []
        for letter in letters:
            planets.append(Planet(host).from_nexsci(name+letter, host))
        return Model(host, planets)


    def _update_host(self, parameters, values):
        '''Update the host parameters in the model.
        '''
        if not isinstance(parameters, (list, np.ndarray)):
            raise ValueError("Parameters must be a list or numpy array.")
        if not isinstance(values, (list, np.ndarray)):
            raise ValueError("Values must be a list or numpy array.")

        # Update the class
        for p, t in zip(parameters, values):
            setattr(self.host, p, t)
        self.host._validate()


    def _update_planet(self, parameters, values):
        '''Update the planet parameters in the model.
        '''
        if not isinstance(parameters, (list, np.ndarray)):
            raise ValueError("Parameters must be a list or numpy array.")
        if not isinstance(values, (list, np.ndarray)):
            raise ValueError("Values must be a list or numpy array.")

        # Update the planet class
        for p, t in zip(parameters, values):
            planet = np.where(utils.alphabet == p.split('.')[0])[0][0] - 1
            setattr(self.planets[planet], p.split('.')[1], t)

        for planet in range(self.nplanets):
            self.planets[planet]._validate()


    def _update_model(self):
        for p in range(self.nplanets):
            self.system.secondaries[p].ecc = self.planets[p].eccentricity
            self.system.secondaries[p].w = self.planets[p].omega
            self.system.secondaries[p].r = self.planets[p].rprs
            self.system.secondaries[p].porb = u.Quantity(self.planets[p].period, u.day).value
            self.system.secondaries[p].a = self.planets[p].separation
            self.system.secondaries[p].Omega = self.planets[p].omega
            self.system.secondaries[p].tref = self.planets[p].t0
            self.system.secondaries[p].inc = self.planets[p].inclination

    def compute(self, time=np.linspace(-0.25, 3.25, 1000)):
        # Make sure all parameters are up to date
        self.host._validate()
        for planet in range(self.nplanets):
            self.planets[planet]._validate()
        self._update_model()

        self.system.compute(time, gradient=True)
        return self.system.lightcurve

    def plot(self, time, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(1)
        f = self.compute(time)
        with plt.style.context(MPLSTYLE):
            plt.plot(time, f)
        return ax

    @property
    def _gradient_labels(self):
        l = []
        for idx in range(self.nplanets):
            for f in fit_params['planet']:
                l.append(string.ascii_lowercase[1 + idx] + '.' + utils.starry_labels[f])
        return l


    @property
    def bounds(self):
        l = []
        for jdx in range(self.nplanets):
             for idx, f in enumerate(fit_params['planet']):
                 l.append(tuple(np.asarray(np.copy(getattr(self.planets[jdx], f + '_error'))) + self.initial_guess[(jdx * len(fit_params['planet'])) + idx]))
        return l


    def _draw_from_bounds(self, size=1):
        if size == 1:
            return [np.random.uniform(b[0], b[1]) for b in self.bounds]
        return [np.random.uniform(b[0], b[1], size=size) for b in self.bounds]


    def _likelihood(self, params, time, data, errors, return_model=False, gradient=True):
        # Update planet
        self._update_planet(self._fit_labels, params)
        model = self.compute(time)
        if return_model:
            return model

        chisq = np.nansum((data - model)**2 / errors**2)
        if gradient:
            dm_dy = [self.system.gradient[label] for label in self._gradient_labels]
            gradient = np.nansum(2 * (model - data) * dm_dy / errors ** 2, axis=1)
            return chisq, gradient
        else:
            return chisq

    def plot_bounds(self, time, data, errors, n=500, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        ax.errorbar(time, data, errors, label='Data')
        ax.plot(time, self._likelihood(self.initial_guess, time, data, errors, return_model=True), color='r', lw=2, label='Initial Guess')
        plt.legend()
        for i in range(n):
            ax.plot(time, self._likelihood(self._draw_from_bounds(), time, data, errors, return_model=True), color='C1', alpha=0.05)
        ax.set_title('Bounds')
        return ax
