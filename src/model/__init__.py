# We will keep keras to import on default
# This may cause problems in multiprocessing
def create_model(*args, **kwargs):
    from model.model_keras import create_model
    return create_model(*args, **kwargs)