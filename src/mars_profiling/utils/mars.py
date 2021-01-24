import functools
from mars.core import ExecutableTuple


def fetch_mars_dict_results(d):
    collectors = dict()
    executables = []

    idx = 0
    for nm, stat in d.items():
        if hasattr(stat, 'execute'):
            if isinstance(stat, ExecutableTuple):
                collectors[nm] = functools.partial(lambda l, r: tuple(results[l:r]),
                                                   idx, idx + len(stat))
                executables.extend(stat)
                idx += len(stat)
            else:
                collectors[nm] = functools.partial(lambda x: results[x], idx)
                executables.append(stat)
                idx += 1

    results = ExecutableTuple(executables).execute().fetch()
    for nm, collector in collectors.items():
        d[nm] = collector()

    return d
