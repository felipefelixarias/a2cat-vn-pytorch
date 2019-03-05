from common.train import Schedule

class LinearSchedule(Schedule):
    def __init__(self, start, end, total_iterations):
        self.start = start
        self.end = end
        self.total_iterations = total_iterations

    def __call__(self):
        return self.end + (0 if self.time >= self.total_iterations else (self.end - self.start) * (self.time / self.total_iterations))