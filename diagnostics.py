# ====================
# Diagnostic functions
# ====================

import time
import cProfile
import pstats
from io import StringIO

def time_this_function(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'\nTime taken by "{func.__name__}": {elapsed:.6f} seconds')
        return result
    return wrapper


def profile_this_function(func, limit=50):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        # Output profiling results
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(limit)  # Limit the output to the first `limit` lines
        print(s.getvalue())
        return result
    return wrapper