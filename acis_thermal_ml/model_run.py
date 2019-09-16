# These classes are designed to mock up something
# with similar attributes as a XijaModel

from .utils import pwr_states

class ModelRunComponent(object):
    def __init__(self, mvals):
        self.mvals = mvals


class ModelRun(object):
    def __init__(self, frames, msid, times, vals, inputs):
        self.msid = msid.lower()
        self.times = times
        self.comp = {self.msid: ModelRunComponent(vals)}
        for key in inputs:
            if key not in pwr_states:
                self.comp[key] = ModelRunComponent(inputs[key][frames:])
                if key == "sim_z":
                    self.comp[key].mvals *= -397.7225924607
