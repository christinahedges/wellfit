''' Class to hold a model of a full system
'''
import starry
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import string
from datetime import datetime

from scipy.optimize import minimize


from .star import Star
from .planet import Planet
from . import utils
from .wellfit import log
import logging


# We should have a better dependency than CT for this.
# This is just to get the NExSci data table.
import characterizethis as ct
df = ct.data()

from lightkurve import MPLSTYLE

import celerite
from celerite import terms

#['log_sigma', 'log_rho']
fit_params =  {'host':[], 'planet':['rprs', 'period', 't0', 'inclination'], 'GP':['log_sigma', 'log_rho']}

class Model(object):
    '''Combined model of a star and companions.
    '''

    def __init__(self, host=None, planets=None, log_sigma=6, log_rho=5,
                log_sigma_error=(-0.5, 0.5), log_rho_error=(-0.5, 0.5)):

        self.fit_params = fit_params

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

        for planet in self.planets:
            planet._validate()
        self.host._validate()

        l = []
        for idx in range(self.nplanets):
            for f in self.fit_params['planet']:
                l.append(u.Quantity(np.copy(getattr(self.planets[idx], f))).value)
        self.initial_guess = l

        self.use_gps = len(self.fit_params['GP']) > 0
        if self.use_gps:
            log.info('Using Gaussian Process to fit long term trends. This will make fitting slower, but more accurate.')
            # Store GP model Parameters
            self.log_sigma = log_sigma
            self.log_rho = log_rho
            self.log_sigma_error = log_sigma_error
            self.log_rho_error = log_rho_error

            # Set up the GP model
            kernel = terms.Matern32Term(log_sigma=self.log_sigma, log_rho=self.log_rho, bounds=self.bounds[-2:])
            self.gp = celerite.GP(kernel)

            for f in self.fit_params['GP']:
                l.append(getattr(self, f))
            self.initial_guess = l


    def __repr__(self):
        s = self.host.__repr__()
        for p in self.planets:
            s += '\n\t{}'.format(p.__repr__())
        return s


    def plot(self, time, flux, flux_error, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(1)
        f = self.compute(time, flux, flux_error)
        with plt.style.context(MPLSTYLE):
            plt.plot(time, f)
        return ax


    @staticmethod
    def from_nexsci(name, **kwargs):
        host = Star().from_nexsci(name)
        letters = np.asarray(df.pl_letter[df.pl_hostname==name])
        planets = []
        for letter in letters:
            planets.append(Planet(host).from_nexsci(name+letter, host))
        return Model(host, planets, **kwargs)


    @property
    def _gradient_labels(self):
        l = []
        for idx in range(self.nplanets):
            for f in self.fit_params['planet']:
                l.append(string.ascii_lowercase[1 + idx] + '.' + utils.starry_labels[f])
        return l


    @property
    def _fit_labels(self):
        l = ['A.' + s for s in self.fit_params['host']]
        for idx in range(self.nplanets):
            for s in self.fit_params['planet']:
                l.append(string.ascii_lowercase[1 + idx] + '.' + s)
        for f in self.fit_params['GP']:
            l.append(f)
        return l


    @property
    def bounds(self):
        l = []
        for jdx in range(self.nplanets):
             for idx, f in enumerate(self.fit_params['planet']):
                 l.append(tuple(np.asarray(np.copy(getattr(self.planets[jdx], f + '_error'))) + self.initial_guess[(jdx * len(self.fit_params['planet'])) + idx]))
        for f in self.fit_params['GP']:
            l.append(tuple([getattr(self, f) + getattr(self, f + '_error')[0], getattr(self, f) + getattr(self, f + '_error')[1]]))
        return l


    def _draw_from_bounds(self, size=1):
        if size == 1:
            return [np.random.uniform(b[0], b[1]) for b in self.bounds]
        return [np.random.uniform(b[0], b[1], size=size) for b in self.bounds]


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


    def _gp_neg_log_like(self, params, y):
        '''for celerite
        '''
        self.gp.set_parameter_vector(params)
        return -self.gp.log_likelihood(y)

    def _gp_grad_neg_log_like(self, params, y):
        '''for celerite
        '''
        self.gp.set_parameter_vector(params)
        return -self.gp.grad_log_likelihood(y)[1]

    def compute(self, time, flux, flux_error, return_gradient=False):
        ''' Compute a single instance of a transit model

        THIS SHOULD RETURN A GRADIENT
        '''
        # Make sure all parameters are up to date
        self.host._validate()
        for planet in range(self.nplanets):
            self.planets[planet]._validate()
        self._update_model()

        self.system.compute(time, gradient=True)
        model_flux = self.system.lightcurve

        dm_dy = [self.system.gradient[label] for label in self._gradient_labels]
        gradient = np.nansum(2 * (model_flux - flux) * dm_dy / flux_error ** 2, axis=1)

        if self.use_gps:
            self.gp.compute(time[model_flux == 1], flux[model_flux == 1])
            mu, var = self.gp.predict(flux[model_flux == 1], time, return_var=True)
            if return_gradient:
                gp_gradient = (self._gp_grad_neg_log_like([self.log_sigma, self.log_rho], flux[model_flux == 1]))
                gradient = np.append(gradient, gp_gradient)
                return model_flux * mu, gradient
            return model_flux * mu
        if return_gradient:
            return model_flux, gradient
        return model_flux

    def _likelihood(self, params, time, flux, flux_error, return_model=False):
        # Update planet
        # Use only the planet parameters
        ok = (self.nplanets) * len(self.fit_params['planet'])
        self._update_planet(np.asarray(self._fit_labels)[0:ok], np.asarray(params)[0:ok])

        if self.use_gps:
            self.log_sigma, self.log_rho = params[ok:]
            self.gp.set_parameter_vector(params[ok:])

        model_flux, gradient = self.compute(time, flux, flux_error, return_gradient=True)
        if return_model:
            return model_flux

        chisq = np.nansum((flux - model_flux)**2 / flux_error**2)
#            dm_dy = [self.system.gradient[label] for label in self._gradient_labels]
#            gradient = np.nansum(2 * (model_flux - flux) * dm_dy / flux_error ** 2, axis=1)
        return chisq, gradient


    def fit(self, time, flux, flux_error, **kwargs):
        start = datetime.now()
        def callback(args):
            t = (datetime.now() - start).seconds / 60
            string = '    ' + ''.join(['{}'.format(np.round(f, 6)) + ''.join([' '] * (15 - len('{}'.format(np.round(f, 6))))) for f in args])
            string += ('{}'.format(np.round(t, 2)))
            log.info(string)

        string = '    ' + ''.join(['{0:15s}'.format(f) for f in self._fit_labels])
        string += ('{0:15s}'.format('Time (m)'))
        log.info('scipy.minimize fit')
        log.info('------------------\n')
        log.info(string)
        self.res = minimize(self._likelihood, self.initial_guess, args=(time, flux, flux_error),
                       jac=True, method='TNC', bounds=self.bounds, options={**kwargs}, callback=callback)


    def plot_bounds(self, time, flux, flux_error, n=500, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        ax.errorbar(time, flux, flux_error, label='Data')
        model_flux = self._likelihood(self.initial_guess, time, flux, flux_error, return_model=True)
        ax.plot(time, model_flux, color='r', lw=2, label='Initial Guess')
        plt.legend()
        for i in range(n):
            model_flux = self._likelihood(self._draw_from_bounds(), time, flux, flux_error, return_model=True)
            ax.plot(time, model_flux, color='C1', alpha=np.max([0.05, 1/(n/10)]))
        ax.set_title('Bounds')
        # Back to the initial parameters
        reset = self._likelihood(self.initial_guess, time, flux, flux_error, return_model=True)
        return ax


    def plot_results(self):
        if 'res' not in self.__dir__():
            raise ValueError("Please run wf.model.fit() before plotting results.")
        n = self.nplanets
        npar = len(self.fit_params['planet'])
        if self.use_gps:
            fig, ax = plt.subplots(n + 1, npar, figsize = (npar*5, (n + 1) * 5))
        else:
            fig, ax = plt.subplots(n, npar, figsize = (npar*5, (n) * 5))

        ax = np.atleast_2d(ax)

        for jdx in range(n):
            for idx, k in enumerate(self.fit_params['planet']):
                ax[jdx, idx].plot(self.bounds[(npar*jdx) + idx], [1,1], zorder=-1, label='Bounds', lw=1)
                ax[jdx, idx].scatter(self.initial_guess[(npar*jdx) + idx], 1, s=150, c='r', label='Initial Guess')
                ax[jdx, idx].scatter(self.res.x[(npar*jdx) + idx], 1, s=60, c='b', label='Best Fit')
                ax[jdx, idx].set_title(k)
                ax[jdx, idx].set_yticks([])
                if idx == len(self.fit_params['planet']) - 1:
                    ax[jdx, idx].legend()

        if self.use_gps:
            jdx += 1
            for idx, k in enumerate(self.fit_params['GP']):
                ax[jdx, idx].plot(self.bounds[(npar*jdx) + idx], [1,1], zorder=-1, label='Bounds', lw=1)
                ax[jdx, idx].scatter(self.initial_guess[(npar*jdx) + idx], 1, s=150, c='r', label='Initial Guess')
                ax[jdx, idx].scatter(self.res.x[(npar*jdx) + idx], 1, s=60, c='b', label='Best Fit')
                ax[jdx, idx].set_title(k)
                ax[jdx, idx].set_yticks([])
                if idx == len(self.fit_params['planet']) - 1:
                    ax[jdx, idx].legend()

            for idx in range(idx + 1, npar):
                fig.delaxes(ax[jdx, idx])
