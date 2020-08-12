Hot Jupiters internal luminosity population inference
=====================================================

Hot Jupiters population inference based on coupling the observed properties of hot Jupiters to theoretical interior structure models.

This framework was developed and implemented for the paper
`Evidence of Three Mechanisms Explaining the Radius Anomaly of Hot Jupiters`
by Paula Sarkis et al. and submitted to A&A. 
The paper is being reviewed and available upon request.

About the code
--------------

The main goal of this code is to infer the internal luminosity of hot Jupiters 
based on coupling observations to theoretical models. We developed a hierarchical Bayesian model that allows us to make inferences on the population of hot Jupiters within a probabilistic framework.
This framework also allows us to account for the uncertainties on the observed parameters.

The framework is divided into two parts:

1- **Lower Level of the hierarchical model**: Infers the internal luminosity of a single planet.

2- **Upper Level of the hierarchical model**: Inference at the population level



Repo Content
------------

**Code**

The probabilistic model is implemented in `mcmc.py`. Specifically, the _lower level_ in `SinglePlanetModel` and the _upper level_ in `PlanetPopulationModel`.

Note `PlanetPopulationModel` is not yet available and will be added soon. In the meantime, it is available upon request.

**Data**

`data` contains all the required data sets to run the code. 

- `interpFn_*`: previously interpolated functions of the theoretical models
- `all_planets-ascii.txt`: the database used in our study 
- `sys_HD_209458`: an example of a dataset in order to run the `SinglePlanetModel`


**Examples**

All examples are applied to `HD_209458`.

`single_planet_demo.ipynb`: demonstrates how to use the code to infer the internal luminosity distribution of `HD_209458`.

`single_planet_choice_of_prior.ipynb`: shows how to use different priors for the internal luminosity at the _lower level_ and its effect on the inference.

More examples will be added soon. 

Dependencies
-------------

- [NumPy](http://www.numpy.org)
- [SciPy](http://www.scipy.org)
- [pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pickle](https://docs.python.org/3.8/library/pickle.html)
- [emcee](https://emcee.readthedocs.io/en/stable/)
- [corner](https://corner.readthedocs.io/en/latest/)
- [Jupyter Notebook](http://jupyter.org)

