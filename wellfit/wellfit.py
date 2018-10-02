

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

class fitter(object):
    '''Something to fit transits...

    This should let us fit with MCMC or basic scipy minimize
    '''
    def __init__(self, method='mcmc'):

    def __repr__(self):
        return 'i am a fitter'



class model(object):
    '''Something to model transits...


    There are lots of different models and they have different APIs. This would be a way to collect them all
    Alternatively...we should just pick one. Maybe starry.
    '''

    def __init__(self, model_package='batman, starry, ktransit, pytransit'):
        '''Some initial parameters'''


    def __repr__(self):
        return 'i am a model'
