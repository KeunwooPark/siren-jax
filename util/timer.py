import time


class Timer:
    def __init__(self):
        self.timestamp = None
        pass

    def start(self):
        self.timestamp = time.perf_counter()

    def get_dt(self):

        now = time.perf_counter()
        dt = now - self.timestamp
        self.timestamp = now
        return dt
