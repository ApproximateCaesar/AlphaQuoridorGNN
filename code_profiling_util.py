# ====================
# Diagnostic functions
# ====================

import time
import cProfile
import pstats
import snakeviz.cli as cli


def time_this_function(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'\nTime taken by "{func.__name__}": {elapsed:.6f} seconds')
        return result
    return wrapper


def profile_this_function(func):
    def wrapper(*args, **kwargs):
        with cProfile.Profile() as pr:
            result = func(*args, **kwargs)
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        filename = "speedtest_profile.prof"
        stats.dump_stats(filename=filename)
        cli.main([filename])
        return result
    return wrapper