import time

class DebugTimer:
    def __init__(self, name="snippet"):
        self.name = name
        self.start = None
        self.end = None
        self.interval = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

    def get_time_str(self):
        if self.start is None:
            return "Timer hasn't started"
        
        current_time = time.time()
        interval = current_time - self.start if self.end is None else self.interval
        
        minutes, seconds = divmod(interval, 60)
        milliseconds = (seconds - int(seconds)) * 1000
        return f"{int(minutes):02d}m {int(seconds):02d}s {int(milliseconds):03d}ms"
