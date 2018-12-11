'''Holds all the information and model of the planet
'''
import starry
import astropy.units as u
import numpy as np
import pandas as pd
from .utils import sep

from .wellfit import log, df

units = {'period': getattr(u, 'day')}

default_bounds = {'rprs_error':(-0.1, 0.1), 'period_error':(-0.0005, 0.0005), 't0_error':(-0.00001, 0.00001), 'inclination_error':(-0.5, 0.5), 'eccentricity_error':(-0.01, 0.01), 'omega_error':(-0.01,0.01)}

class Planet(object):
    '''Companion class
    '''

    def __init__(self, host=None, rprs=0.01, period=10, t0=0, inclination=90, omega=0, eccentricity=0, multi=False, lum=0,
                     rprs_error=None, period_error=None, t0_error=None, inclination_error=None, omega_error=None, eccentricity_error=None):
        self.host = host
        self.rprs = rprs
        self.period = u.Quantity(period, u.day)
        self.t0 = t0
        self.inclination = inclination
        self._init_inclination = inclination
        self.omega = omega
        self.eccentricity = eccentricity
        self._init_eccentricity = eccentricity
        self.lum = lum

        self.rprs_error = rprs_error
        self.period_error = period_error
        self.t0_error = t0_error
        self.inclination_error = inclination_error
        self.eccentricity_error = eccentricity_error
        self.omega_error = omega_error

        self._validate()

        self._init_model = starry.kepler.Secondary(lmax=1, multi=multi)

    def _validate(self):
        '''Ensures unit convention is obeyed'''
        for k, f in units.items():
            setattr(self, k, u.Quantity(getattr(self, k), f))
        self._validate_errors()

    def _validate_errors(self):
        '''Ensure the bounds are physical'''
        for key in ['rprs_error', 'period_error', 't0_error', 'inclination_error', 'eccentricity_error']:
            if getattr(self,key) is None:
                setattr(self, key, default_bounds[key])
            if ~np.isfinite(getattr(self,key)[0]):
                setattr(self,  key, tuple([default_bounds[key][0], getattr(self, key)[1]]))
            if ~np.isfinite(getattr(self, key)[1]):
                setattr(self,  key, tuple([getattr(self, key)[0], default_bounds[key][1]]))

        if self.eccentricity + self.eccentricity_error[0] < 0:
            self.eccentricity_error = tuple([-self._init_eccentricity, self.eccentricity_error[1]])

        if self.eccentricity + self.eccentricity_error[1] > 1:
            self.eccentricity_error = tuple([self.eccentricity_error[0], 1 - self._init_eccentricity])

        if self.inclination + self.inclination_error[1] > 90.:
            self.inclination_error = tuple([self.inclination_error[0], 90. - self._init_inclination])



    @staticmethod
    def from_nexsci(name, host, sigma=10):
        # Get data
        ok = np.where(df.pl_hostname + df.pl_letter == name)[0]
        if len(ok) == 0:
            raise ValueError('No planet named {}.'.format(name))
        d = df.iloc[ok]

        # Get params
        rprs = np.asarray(d.pl_radj)/(host.radius.to(u.jupiterRad).value)
        period = np.asarray(d.pl_orbper)
        t0 = np.asarray(d.pl_tranmid)
        inc = np.asarray(d.pl_orbincl)
        if ~np.isfinite(inc):
            inc = [90]
        ecc = np.asarray(d.pl_eccen)
        if ~np.isfinite(ecc):
            ecc = [0]

        # Get errors
        rprs_error = tuple([d.iloc[0].pl_radjerr2/(host.radius.to(u.jupiterRad).value) * sigma, d.iloc[0].pl_radjerr1/(host.radius.to(u.jupiterRad).value) * sigma])
        period_error = tuple([d.iloc[0].pl_orbpererr2 * sigma, d.iloc[0].pl_orbpererr1 * sigma])
        t0_error = tuple([d.iloc[0].pl_tranmiderr2 * sigma, d.iloc[0].pl_tranmiderr1 * sigma])
        inclination_error = tuple([d.iloc[0].pl_orbinclerr2 * sigma, d.iloc[0].pl_orbinclerr1 * sigma])
        eccentricity_error = tuple([d.iloc[0].pl_eccenerr2 * sigma, d.iloc[0].pl_eccenerr1 * sigma])

        return Planet(host, rprs=rprs[0], period=period[0], t0=t0[0], inclination=inc[0], omega=0, eccentricity=ecc[0],
                      rprs_error=rprs_error, period_error=period_error, t0_error=t0_error, inclination_error=inclination_error,
                      eccentricity_error=eccentricity_error)

    @property
    def properties(self):
        df = pd.DataFrame(columns=['Value', 'Lower Bound', 'Upper Bound'])
        for idx, p in enumerate(['rprs', 'period', 't0', 'inclination', 'eccentricity']):
            df.loc[p, 'Value'] = getattr(self, p)
            df.loc[p, 'Lower Bound'] = getattr(self, p + '_error')[0]
            df.loc[p, 'Upper Bound'] = getattr(self, p + '_error')[1]
        return df

    @property
    def model(self):
        self._init_model.L = self.lum
        self._init_model.ecc = self.eccentricity
        self._init_model.r = self.rprs
        self._init_model.porb = u.Quantity(self.period, u.day).value
        self._init_model.a = self.separation
        self._init_model.Omega = self.omega
        self._init_model.tref = self.t0
        self._init_model.inc = self.inclination
        return self._init_model

    @property
    def separation(self):
        return (sep(self.period, self.host.mass)/self.host.radius).value

    @property
    def separation_error(self):
        e = []
        for idx in [0, 1]:
            p = (self.period - self.period_error[idx]*self.period.unit)
            m = (self.host.mass - self.host.mass_error[idx]*self.host.mass.unit)
            r = (self.host.radius - self.host.radius_error[idx]*self.host.radius.unit)
            e.append((sep(p, m)/r).value - self.separation)
        return tuple(e)

    @property
    def radius(self):
        return (self.rprs * self.host.radius).to(u.jupiterRad)

    @property
    def radius_error(self):
        e = []
        for idx in [0, 1]:
            er1 = self.rprs_error[idx]/self.rprs
            er2 = self.host.radius_error[idx]/self.host.radius.value
            er = (er1**2 + er2**2)**0.5
            e.append(er * self.radius.value * (-1)**(idx - 1))
        return tuple(e)

    @property
    def duration(self):
        b = self.separation * np.cos(self.inclination * np.pi/180)
        l = ((self.host.radius + self.radius.to(u.solRad))**2 + (b * self.host.radius)**2)**0.5
        l /= (self.separation*self.host.radius)
        return np.arcsin(l.value) * self.period/np.pi

    @property
    def duration_error(self):
        e = []
        for idx in [1, 0]:
            a = self.separation + self.separation_error[idx]
            inc = self.inclination + self.inclination_error[idx]
            rstar = self.host.radius + self.host.radius_error[idx]*self.host.radius.unit
            rpl = (self.radius + self.radius_error[idx]*self.radius.unit).to(u.solRad)
            p = (self.period - self.period_error[idx]*self.period.unit)

            b = a * np.cos(inc * np.pi/180)
            l = ((rstar + rpl)**2 + (b * rstar)**2)**0.5
            l /= (a * rstar)
            e.append((np.arcsin(l.value) * p/np.pi - self.duration).value)
        return tuple(e)


    def __repr__(self):
        return ('Planet: RpRs {} , P {}, t0 {}'.format(self.rprs, self.period,self.t0))
