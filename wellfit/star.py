''' Holds all the information and model of the star
'''

import starry
import astropy.units as u
import numpy as np
import pandas as pd

from astropy.constants import sigma_sb

from .wellfit import log, df
from .utils import ld_table, WellFitException

units = {'mass': getattr(u, 'solMass'),
         'radius': getattr(u, 'solRad'),
         'temperature': getattr(u, 'K'),
         'luminosity':getattr(u, 'solLum')}

default_bounds = {'radius_error':(-0.1, 0.1), 'mass_error':(-0.1, 0.1), 'temperature_error':(-500, 500), 'limb_darkening_error':(-0.1,0.1)}


class Star(object):
    '''Primary star class'''

    def __init__(self, radius=1, mass=1, temperature=5777,
                 radius_error=None, mass_error=None, temperature_error=None, luminosity=None, luminosity_error=None):
        self.radius = u.Quantity(radius, u.solRad)
        self.mass = u.Quantity(mass, u.solMass)
        self.temperature = u.Quantity(temperature, u.K)
        self.radius_error = radius_error
        self.mass_error = mass_error
        self.temperature_error = temperature_error

        if luminosity is not None:
            self.luminosity = luminosity
        else:
            self.luminosity = self._luminosity

        if luminosity is not None:
            self.luminosity_error = luminosity_error
        else:
            self.luminosity_error = self._luminosity_error


        t = ld_table[(ld_table.teff == (self.temperature.value)//250 * 250) & (ld_table.met == 0) & (ld_table.logg == 5)]
        if len(t) == 0:
            raise WellFitException('Can not find limb darkening parameters. This should not happen. Please report this error.')
        self.limb_darkening = [t.iloc[0].u, t.iloc[0].a]
        self._validate()

        self._init_model = starry.kepler.Primary()

    @property
    def _luminosity(self):
        return (4 * np.pi * self.radius**2 * sigma_sb * self.temperature**4).to(u.solLum)

    @property
    def _luminosity_error(self):
        e = []
        for idx in [0, 1]:
            r = self.radius + self.radius_error[idx] * self.radius.unit
            t = self.temperature + self.temperature_error[idx] * self.temperature.unit
            e.append(((4 * np.pi * r**2 * sigma_sb * t**4).to(u.solLum) - self.luminosity).value)
        return tuple(e)


    def __repr__(self):
        return ('Star: {}, {}, {}, {}'.format(self.radius, self.mass, self.temperature, self.luminosity))

    def _validate(self):
        '''Ensures unit convention is obeyed'''
        for k, f in units.items():
            setattr(self, k, u.Quantity(getattr(self, k), f))
        self._validate_errors()

    def _validate_errors(self):
        '''Ensure the bounds are physical'''
        for key in ['radius_error', 'mass_error', 'temperature_error']:
            if getattr(self,key) is None:
                setattr(self, key, default_bounds[key])
            if ~np.isfinite(getattr(self,key)[0]):
                setattr(self,  key, tuple([default_bounds[key][0], getattr(self, key)[1]]))
            if ~np.isfinite(getattr(self, key)[1]):
                setattr(self,  key, tuple([getattr(self, key)[0], default_bounds[key][1]]))

    @staticmethod
    def from_nexsci(name):
        ok = np.where(df.pl_hostname == name)[0]
        if len(ok) == 0:
            raise ValueError('No planet named {}'.format(name))
        ok = ok[0]
        d = df.iloc[ok]
        return Star(radius=d.st_rad, mass=d.st_mass, temperature=d.st_teff, radius_error=tuple([d.st_raderr2, d.st_raderr1]),
                    mass_error=tuple([d.st_masserr2, d.st_masserr1]), temperature_error=tuple([d.st_tefferr2, d.st_tefferr1]))

    @property
    def properties(self):

        df = pd.DataFrame(columns=['\emph{Host Star}'])
        df.loc['Radius', '\emph{Host Star}'] = '{} R$_\odot$ $\pm$_{{{}}}^{{{}}}'.format(np.round(self.radius.value, 3), np.round(self.radius_error[0], 4), np.round(self.radius_error[1], 4))
        df.loc['Mass', '\emph{Host Star}'] = '{} M$_\odot$ $\pm$_{{{}}}^{{{}}}'.format(np.round(self.mass.value, 3), np.round(self.mass_error[0], 4), np.round(self.mass_error[1], 4))
        df.loc['T_{eff}', '\emph{Host Star}'] = '{} $K$ $\pm$_{{{}}}^{{{}}}'.format(int(self.temperature.value), int(self.temperature_error[0]), int(self.temperature_error[1]))
        df.loc['Luminosity', '\emph{Host Star}'] = '{} L$_\odot$ $\pm$_{{{}}}^{{{}}}'.format(np.round(self.luminosity.value, 3), np.round(self.luminosity_error[0], 3), np.round(self.luminosity_error[1], 3))
        df.loc['Limb Darkening 1 ($u$)', '\emph{Host Star}'] = '{}'.format(np.round(self.limb_darkening[0], 2))
        df.loc['Limb Darkening 2 ($a$)', '\emph{Host Star}'] = '{}'.format(np.round(self.limb_darkening[1], 2))

        return df

    @property
    def model(self):
        self._init_model[1] = self.limb_darkening[0]
        self._init_model[2] = self.limb_darkening[1]
        return self._init_model

    def show(self):
        self.model.show()
