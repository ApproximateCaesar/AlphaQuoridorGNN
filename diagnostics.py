# ====================
# Diagnostic functions
# ====================

import time
import cProfile
import pstats
import snakeviz.cli as cli
from io import StringIO

from tensorflow.python.keras.utils.metrics_utils import result_wrapper


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

def speedtest(func):
    def wrapper(*args, **kwargs):
        with cProfile.Profile() as pr:
            result = func()
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        filename = "speedtest_profile.prof"
        stats.dump_stats(filename=filename)
        cli.main([filename])
        return result
    return wrapper