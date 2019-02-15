from common.abstraction import AbstractTrainerWrapper

class SaveTrainerWrapper(AbstractTrainerWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def process(self, **kwargs):
        res = super().process(**kwargs)
        (tdiff, episode_end) = res

        return res