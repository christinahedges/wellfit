

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
