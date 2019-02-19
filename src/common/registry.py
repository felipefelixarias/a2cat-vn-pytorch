import common.train_wrappers as wrappers

_registry = dict()
def register_trainer(id, **kwargs):
    def wrap(trainer):
        _registry[id] = dict(trainer = trainer, **kwargs)
        return trainer
    return wrap

def make_trainer(id, **kwargs):
    instance = _registry[id]['trainer'](**kwargs)

    wargs = dict(**_registry[id])
    del wargs['trainer']
    instance = wrappers.wrap(instance, **wargs).compile()
    return instance