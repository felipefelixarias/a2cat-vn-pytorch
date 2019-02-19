import common.train_wrappers as wrappers

_registry = dict()
_agent_registry = dict()
def register_trainer(id, **kwargs):
    def wrap(trainer):
        _registry[id] = dict(trainer = trainer, **kwargs)
        return trainer
    return wrap


def register_agent(id, **kwargs):
    def wrap(agent):
        _agent_registry[id] = dict(agent = agent, **kwargs)
        return agent

    return wrap

def make_trainer(id, **kwargs):
    instance = _registry[id]['trainer'](**kwargs)

    wargs = dict(**_registry[id])
    del wargs['trainer']
    instance = wrappers.wrap(instance, **wargs).compile()
    return instance

def make_agent(id, **kwargs):
    wargs = dict(**_agent_registry[id])
    del wargs['trainer']

    wargs.update(kwargs)
    instance = _agent_registry[id]['agent'](**wargs)
    return instance