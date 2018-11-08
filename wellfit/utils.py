""" Basic transit functions """

import astropy.units as u
from astropy.constants import G
import numpy as np

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


def sep(period, st_mass):
    separation = (((G*st_mass/(4*np.pi**2)) * (period)**2)**(1/3)).to(u.solRad)
    return separation

def dur(period, st_radius, pl_radius, st_mass):
    b = 0
    separation = sep(period, st_mass)
    duration = period * np.arcsin(((st_radius + pl_radius.to(u.solRad))**2 - (b*st_radius)**2)**0.5/separation).value/np.pi
    return duration
