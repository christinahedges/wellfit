

'''
What do we want?

fitter = wf.fitter('mcmc')
model = wf.model('batman')
model.add_star()
model.add_planet()
model.time(...)

model.guess_for_me(data)
model.random()
model.flux
#Plot some sort of orbit diagram/image like in starry
model.skematic


fit = fitter(model, data)
fit.corner
fit.best_params
fit.initial_params
fit.plot
fit.results
fit.to_pandas()

#PUBLISH READY TABLE:
fit.to_latex()

'''

import logging
logging.basicConfig(format='%(message)s')
log = logging.getLogger('WELLFIT')

# We should have a better dependency than CT for this.
# This is just to get the NExSci data table.
import characterizethis as ct
df = ct.get_data()

from .planet import Planet
from .star import Star
from .model import Model
