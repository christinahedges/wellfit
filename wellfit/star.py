''' Holds all the information and model of the star
'''

import starry
import astropy.units as u
import numpy as np
# We should have a better dependency than CT for this.
# This is just to get the NExSci data table.

import characterizethis as ct
df = ct.data()

units = {'mass': getattr(u, 'solMass'),
         'radius': getattr(u, 'solRad'),
         'temperature': getattr(u, 'K')}

default_bounds = {'radius_error':(-0.1, 0.1), 'mass_error':(-0.1, 0.1), 'temperature_error':(-500, 500), 'limb_darkening_error':(-0.1,0.1)}


class Star(object):
    '''Primary star class'''

    def __init__(self, radius=1, mass=1, temperature=5777, limb_darkening=[0.4, 0.26],
                 radius_error=None, mass_error=None, temperature_error=None, limb_darkening_error=None):
        self.limb_darkening = limb_darkening
        self.radius = u.Quantity(radius, u.solRad)
        self.mass = u.Quantity(mass, u.solMass)
        self.temperature = u.Quantity(temperature, u.K)
        self.radius_error = radius_error
        self.mass_error = mass_error
        self.temperature_error = temperature_error
        self.limb_darkening_error = limb_darkening_error

        self._validate()

        self._init_model = starry.kepler.Primary()

    def __repr__(self):
        return ('Star: {}, {}, {}'.format(self.radius, self.mass, self.temperature))

    def _validate(self):
        '''Ensures unit convention is obeyed'''
        for k, f in units.items():
            setattr(self, k, u.Quantity(getattr(self, k), f))
        self._validate_errors()

    def _validate_errors(self):
        '''Ensure the bounds are physical'''
        for key in ['radius_error', 'mass_error', 'temperature_error', 'limb_darkening_error']:
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
        s = ''
        for p in ['radius', 'mass', 'temperature', 'limb_darkening']:
            s += '{} : {} $\pm$ {} \n'.format(p, getattr(self, p), getattr(self, p + '_error'))
        return s

    @property
    def model(self):
        self._init_model[1] = self.limb_darkening[0]
        self._init_model[2] = self.limb_darkening[1]
        return self._init_model

    def show(self):
        self.model.show()
