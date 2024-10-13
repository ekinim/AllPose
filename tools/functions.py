from . import classes


supported_models = ['opencv', 'movenet']


def get_estimator(model):
    assert model in supported_models, f'Model name must be one of these: {supported_models}'

    if model == 'opencv':
        estimator = classes.OpenCVEstimator()

    elif model == 'movenet':
        estimator = classes.MovenetEstimator()

    else:
        estimator = None

    return estimator