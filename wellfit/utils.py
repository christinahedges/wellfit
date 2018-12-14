""" Basic transit functions """

import astropy.units as u
from astropy.constants import G
import numpy as np
import pandas as pd

from astropy.constants import sigma_sb

from . import PACKAGEDIR

starry_translation = {'radius':'r', 'separation':'a',
                      't0':'tref', 'porb':'period',
                       'inclination':'inc', 'eccentricity':'ecc',
                       'omega':'Omega'}

# Kepler Short Cadence and Long Cadence
SC = (58.84876*u.second.to(u.day))
LC =  30 * SC

alphabet = np.asarray(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n'])

starry_labels = {'rprs' : 'r',
                 'period':'porb',
                 'separation':'a',
                 't0':'tref',
                 'inclination':'inc',
                 'omega':'Omega',
                 'eccentricity':'ecc'}

class WellFitException(Exception):
    '''Raised when there is a really fit error.'''
    pass

def sep(period, st_mass):
    separation = (((G*st_mass/(4*np.pi**2)) * (period)**2)**(1/3)).to(u.solRad)
    return separation

def dur(period, st_radius, pl_radius, st_mass):
    b = 0
    separation = sep(period, st_mass)
    duration = period * np.arcsin(((st_radius + pl_radius.to(u.solRad))**2 - (b*st_radius)**2)**0.5/separation).value/np.pi
    return duration



def get_mass(star):
    '''ZAMS Mass
    @ARTICLE{1991Ap&SS.181..313D,
       author = {{Demircan}, O. and {Kahraman}, G.},
        title = "{Stellar mass-luminosity and mass-radius relations}",
      journal = {\apss},
     keywords = {Binary Stars, Main Sequence Stars, Mass To Light Ratios, Radii, Stellar Luminosity, Stellar Mass, Parameterization, Regression Analysis, Stellar Models},
         year = 1991,
        month = jul,
       volume = 181,
        pages = {313-322},
          doi = {10.1007/BF00639097},
       adsurl = {http://adsabs.harvard.edu/abs/1991Ap%26SS.181..313D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    '''
    mass = ((star.luminosity)/1.15)**(1/3.36)
    if hasattr(mass, 'unit'):
        return mass.value * u.solMass
    return mass * u.solMass


def get_mass_error(star):
    '''ZAMS Mass error'''
    l = star.luminosity
    if hasattr(l, 'unit'):
        l = l.value

    e = (((l + star.luminosity_error[0])/1.15)**(1/3.36) - star.mass.value,
             ((l + star.luminosity_error[1])/1.15)**(1/3.36) - star.mass.value)
    return e


def get_luminosity(star):
    return (4 * np.pi * star.radius**2 * sigma_sb * star.temperature**4).to(u.solLum)


def get_luminosity_error(star):
    e = []
    for idx in [0, 1]:
        r = star.radius + star.radius_error[idx] * star.radius.unit
        t = star.temperature + star.temperature_error[idx] * star.temperature.unit
        e.append(((4 * np.pi * r**2 * sigma_sb * t**4).to(u.solLum) - star.luminosity).value)
    return tuple(e)

ld_table = pd.read_csv('{}/data/limb_darkening.csv'.format(PACKAGEDIR), comment='#')
