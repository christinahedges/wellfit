''' Class to hold a model of a full system
'''
import starry
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import string
from datetime import datetime
import pandas as pd
import sys
import os
import pickle

from scipy.optimize import minimize


from .star import Star
from .planet import Planet
from . import utils
from .wellfit import log, df
import logging

from lightkurve import MPLSTYLE

import celerite
from celerite import terms

import emcee
import corner



def _prob(params, time, flux, flux_error):
    if _wellfit_toy_model is None:
        raise utils.WellFitException('You do not have a `_wellfit_toy_model` variable set.'
                                'This should not be possible. Please report this error')
    lp = _wellfit_toy_model._prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp - _wellfit_toy_model._likelihood(params, time, flux, flux_error, return_gradients=False)


#['log_sigma', 'log_rho']
fit_params =  {'host':[], 'planet':['rprs', 'period', 't0', 'inclination', 'eccentricity'], 'GP':['log_sigma', 'log_rho']}


class Model(object):
    '''Combined model of a star and companions.
    '''

    def __init__(self, host=None, planets=None, use_gps=False, log_sigma=6, log_rho=5,
                log_sigma_error=(-0.5, 0.5), log_rho_error=(-0.5, 0.5)):

        self.use_gps = use_gps
        self.fit_params = deepcopy(fit_params)
        if not self.use_gps:
            self.fit_params['GP'] = []

        # Validate everything
        if host is not None:
            host._validate()
        if planets is not None:
            for planet in planets:
                planet._validate()

        self.host = host
        if isinstance(planets, list):
            self.planets = planets
        elif planets is None:
            self.planets = []
        else:
            self.planets = [planets]

        self.nplanets = len(self.planets)

        for planet in self.planets:
            planet._validate()
        self.host._validate()

        l = []
        for idx in range(self.nplanets):
            for f in self.fit_params['planet']:
                l.append(u.Quantity(np.copy(getattr(self.planets[idx], f))).value)
        self._best_guess = l
        self.initial_guess = l

        self.system = None

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
            self._best_guess = l


        l = []
        for jdx in range(self.nplanets):
             for idx, f in enumerate(self.fit_params['planet']):
                 l.append(tuple(np.asarray(np.copy(getattr(self.planets[jdx], f + '_error'))) + self._best_guess[(jdx * len(self.fit_params['planet'])) + idx]))
        for f in self.fit_params['GP']:
            l.append(tuple([getattr(self, f) + getattr(self, f + '_error')[0], getattr(self, f) + getattr(self, f + '_error')[1]]))
        self.initial_bounds = l

        self._is_eccen = np.where([l.split('.')[-1] == 'eccentricity' for l in self._fit_labels])[0]
        self._is_inc = np.where([l.split('.')[-1] == 'inclination' for l in self._fit_labels])[0]
        self.nwalkers = 100
        self.burnin = 200
        self.nsteps = 1000
        # Work around for the starry pickle bug.
        global _wellfit_toy_model
        _wellfit_toy_model = None
        self._initialize_system()

    def _initialize_system(self):
        if (self.planets is None) & (self.host is None):
            self.system = None
        else:
            self.system = starry.kepler.System(self.host.model, *[p.model for p in self.planets])


    def __repr__(self):
        s = self.host.__repr__()
        for p in self.planets:
            s += '\n\t{}'.format(p.__repr__())
        return s


    def plot(self, time, flux=None, flux_error=None, ax=None, **kwargs):
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

    @staticmethod
    def read(fname):
        '''Read a wf.model written by the Model.write() method.
        '''
        model = pickle.load(open(fname, 'rb'))
        if not isinstance(model, Model):
            raise ValueError('{} does not contain a wellfit.model.Model'.format(fname))
        for p in model.planets:
            p._initialize_model()
        model.host._initialize_model()
        model._initialize_system()
        return model

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
                 l.append(tuple(np.asarray(np.copy(getattr(self.planets[jdx], f + '_error'))) + self._best_guess[(jdx * len(self.fit_params['planet'])) + idx]))
        for f in self.fit_params['GP']:
            l.append(tuple([getattr(self, f) + getattr(self, f + '_error')[0], getattr(self, f) + getattr(self, f + '_error')[1]]))
        return l


    def _draw_from_bounds(self, size=1):
        if size == 1:
            return [np.random.uniform(b[0], b[1]) for b in self.bounds]
        return [np.random.uniform(b[0], b[1], size=size) for b in self.bounds]


    def _update_host(self, parameters, values):
        '''Update the host parameters in the self.
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
        '''Update the planet parameters in the self.
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
        if self.system is None:
            pass
        for p in range(self.nplanets):
            self.system.secondaries[p].ecc = self.planets[p].eccentricity
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

    def compute(self, time, flux=None, flux_error=None, return_gradient=False):
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

        if return_gradient:
            if flux is None:
                raise ValueError('`flux` is None. When using gradients, compute must be passed the flux with errors.')
            if flux_error is None:
                raise ValueError('`flux_error` is None. When using gradients, compute must be passed the flux with errors.')
            dm_dy = [self.system.gradient[label] for label in self._gradient_labels]
            gradient = np.nansum(2 * (model_flux - flux) * dm_dy / flux_error ** 2, axis=1)

        if self.use_gps:
            if flux is None:
                raise ValueError('`flux` is None. When using GPs, compute must be passed the flux with errors.')
            if flux_error is None:
                raise ValueError('`flux_error` is None. When using GPs, compute must be passed the flux with errors.')

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



    def fit(self, time, flux, flux_error, **kwargs):
        start = datetime.now()
        def callback(args):
            t = (datetime.now() - start).seconds / 60
            string = '    ' + ''.join(['{}'.format(np.round(f, 6)) + ''.join([' '] * (15 - len('{}'.format(np.round(f, 6))))) for f in args])
            string += ('{}'.format(np.round(t, 2)))
            log.info(string)

        string = '    ' + ''.join(['{0:15s}'.format(f) for f in self._fit_labels])
        log.info('scipy.minimize fit')
        log.info('------------------\n')
        log.info(string)
        self.res = minimize(self._likelihood, self._best_guess, args=(time, flux, flux_error),
                       jac=True, method='TNC', bounds=self.bounds, options=kwargs, callback=callback)
        self.best_fit = self.res.x



    def _likelihood(self, params, time, flux=None, flux_error=None, return_model=False, return_gradients=True):
        # Update planet
        # Use only the planet parameters
        ok = (self.nplanets) * len(self.fit_params['planet'])
        self._update_planet(np.asarray(self._fit_labels)[0:ok], np.asarray(params)[0:ok])

        if self.use_gps:
            self.log_sigma, self.log_rho = params[ok:]
            self.gp.set_parameter_vector(params[ok:])

        model_flux, gradient = self.compute(time, flux=flux, flux_error=flux_error, return_gradient=True)
        if return_model:
            return model_flux

        chisq = np.nansum((flux - model_flux)**2 / flux_error**2)
#            dm_dy = [self.system.gradient[label] for label in self._gradient_labels]
#            gradient = np.nansum(2 * (model_flux - flux) * dm_dy / flux_error ** 2, axis=1)
        if return_gradients:
            return chisq, gradient
        return chisq

    def _prior(self, params):
        for idx, p in enumerate(params):
#            e = getattr(self.planets[0], self._fit_labels[idx].split('.')[1] + '_error')
            e = self.bounds[idx]
            if (p < e[0]) | (p > e[1]):
                return -np.inf
        if params[self._is_eccen] < 0:
            return -np.inf
        if params[self._is_inc] > 90:
            return -np.inf
        return 0.0

    @property
    def _mcmc_starting_points(self):
        ndim = len(self.best_fit)
        # steps are 0.1% of the bound
        pos = np.asarray([(((self.bounds[idx][1] - self.bounds[idx][0])/1000) * np.random.randn(self.nwalkers)/5) + self.best_fit[idx] for idx in range(ndim)]).T
        # Nothing outside of physical...
        pos[:, self._is_eccen] = np.abs(pos[:, self._is_eccen])
        pos[:, self._is_inc] = 90 - np.abs(90 - pos[:, self._is_inc])

        return pos


    def _assign(self):
        ndim = len(self.best_fit)

        ans = []
        for label, i in  zip(self._fit_labels, range(ndim)):
            med = np.median(self.sampler.chain[:, self.burnin:, i])
            low = np.percentile(self.sampler.chain[:, self.burnin:, i], 16)
            high = np.percentile(self.sampler.chain[:, self.burnin:, i], 84)
            self._update_planet([label], [med])
            self._update_planet(['{}_error'.format(label)], [(low - med, high - med)])
            ans.append(med)
        self._best_guess = ans
        self.best_fit = ans


    def fit_mcmc(self, time, flux, flux_err, threads=8):
        #self._best_guess = self.best_fit
        ndim = len(self.best_fit)
        # Work around for the starry pickle bug
        global _wellfit_toy_model
        _wellfit_toy_model = self

        nll = lambda *args: -self._liklihood(*args)
        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, _prob, args=(time, flux, flux_err), threads=threads)

        width = 50
        start = datetime.now()
        dt = 0.
        log.info('emcee fit')
        log.info('----------\n')
        for i, result in enumerate(sampler.sample(self._mcmc_starting_points, iterations=self.nsteps)):
            if (i / int(self.nsteps/10)) == (i // int(self.nsteps/10)):
                t = (datetime.now() - start).seconds / 60
                dt = t/float(i + 1)
                n = int((width+1) * float(i) / self.nsteps)
                msg = "\r[{0}{1}] \t\t {2}mins / {3}mins".format('#' * n, ' ' * (width - n), np.round(t, 2), np.round(dt * self.nsteps, 2))
                log.info(msg)
        log.info("\n")

        self.sampler = sampler
        # Set my work around back to None, users shouldn't be able to interact with this stuff.
        _wellfit_toy_model = None

        log.info('{} / {} samples burned'.format(self.burnin, self.nsteps))
        log.info('Setting results...')
        ndim = len(self.best_fit)
        samples = self.sampler.chain[:, self.burnin:, :].reshape((-1, ndim))
        self._assign()


    def to_latex(self):
        host_df = self.host.properties

        planet_df = None
        for idx in range(len(self.planets)):
            planet = self.planets[idx]
            name = '\emph{{Planet {}}}'.format(utils.alphabet[idx + 1])
            df = planet.properties
            df.columns = [name]
            if planet_df is None:
                planet_df = df
            else:
                planet_df = planet_df.join(df)

        return('{}\n{}'.format(host_df.to_latex(escape=False), planet_df.to_latex(escape=False)))


    def plot_corner(self):
        if 'sampler' not in self.__dir__():
            raise ValueError("Please run wf.self.fit_mcmc() before plotting a corner plot.")
        log.info('{} / {} samples burned'.format(self.burnin, self.nsteps))
        ndim = len(self.best_fit)
        samples = self.sampler.chain[:, self.burnin:, :].reshape((-1, ndim))
        cornerplot = corner.corner(samples, labels=self._fit_labels,
                                   truths=self.best_fit)
        return cornerplot

    def plot_burnin(self, show_bounds=True):
        ndim = len(self.best_fit)
        fig, axs = plt.subplots(ndim, figsize=(10, 16))
        for i in range(ndim):
            _ = axs[i].plot(self.sampler.chain[:, 0:, i].T, alpha=0.1, color='k');
            axs[i].set_ylabel(self._fit_labels[i])
            axs[i].axvline(self.burnin)
            if show_bounds:
                axs[i].axhline(self.bounds[i][0], ls='--')
                axs[i].axhline(self.bounds[i][1], ls='--')
                axs[i].axhline(self.best_fit[i])
        return fig


    def plot_bounds(self, time, flux, flux_error, n=500, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        ax.errorbar(time, flux, flux_error, label='Data')
        model_flux = self._likelihood(self._best_guess, time, flux, flux_error, return_model=True)
        ax.plot(time, model_flux, color='r', lw=2, label='Initial Guess')
        for i in range(n):
            model_flux = self._likelihood(self._draw_from_bounds(), time, flux, flux_error, return_model=True)
            if i==0:
                ax.plot(time, model_flux, color='C1', alpha=np.max([0.05, 1/(n/10)]), label='Random Draw From Bounds')
                leg = plt.legend()
                for lh in leg.legendHandles:
                    lh.set_alpha(1)
            ax.plot(time, model_flux, color='C1', alpha=np.max([0.05, 1/(n/10)]))
        ax.set_title('Bounds')
        # Back to the initial parameters
        reset = self._likelihood(self._best_guess, time, flux, flux_error, return_model=True)
        return ax


    def plot_results(self):
        if 'res' not in self.__dir__():
            raise ValueError("Please run wf.self.fit() before plotting results.")
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
                ax[jdx, idx].scatter(self._best_guess[(npar*jdx) + idx], 1, s=150, c='r', label='Initial Guess')
                ax[jdx, idx].scatter(self.res.x[(npar*jdx) + idx], 1, s=60, c='b', label='Best Fit')
                ax[jdx, idx].set_title(k)
                ax[jdx, idx].set_yticks([])
                if idx == len(self.fit_params['planet']) - 1:
                    ax[jdx, idx].legend()

        if self.use_gps:
            jdx += 1
            for idx, k in enumerate(self.fit_params['GP']):
                ax[jdx, idx].plot(self.bounds[(npar*jdx) + idx], [1,1], zorder=-1, label='Bounds', lw=1)
                ax[jdx, idx].scatter(self._best_guess[(npar*jdx) + idx], 1, s=150, c='r', label='Initial Guess')
                ax[jdx, idx].scatter(self.res.x[(npar*jdx) + idx], 1, s=60, c='b', label='Best Fit')
                ax[jdx, idx].set_title(k)
                ax[jdx, idx].set_yticks([])
                if idx == len(self.fit_params['planet']) - 1:
                    ax[jdx, idx].legend()

            for idx in range(idx + 1, npar):
                fig.delaxes(ax[jdx, idx])


    def write(self, fname='out.wf.model', overwrite=False):
        '''Write a planet class to a binary file. Note that since starry models cannot
        be pickled, this is the only way to write a Star. To read it back in, you
        must use the read function.'''
        if os.path.isfile(fname) & (not overwrite):
            raise ValueError('File exists. Please set overwrite to True or choose another file name.')
        self.system = None
        for p in self.planets:
            p._init_model = None
        self.host._init_model = None
        pickle.dump(self, open(fname, 'wb'))
        for p in self.planets:
            p._initialize_model()
        self.host._initialize_model()
        self._initialize_system()
