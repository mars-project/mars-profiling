import functools
import pickle
import weakref

import mars.remote as mr
from mars.core import ExecutableTuple
from mars.utils import enter_mode

_result_cache = weakref.WeakKeyDictionary()


def fetch_mars_dict_results(d, ignore_value_counts=True, fetch_only=False,
        fetch_with_remote=None):
    collectors = dict()
    executables = []
    to_fetch_tileables = []

    for nm, stat in d.items():
        if isinstance(nm, tuple):
            real_name = nm[0]
        else:
            real_name = nm

        if hasattr(stat, 'execute'):
            to_be_fetched = not (ignore_value_counts and real_name.startswith('value_counts'))
            if isinstance(stat, ExecutableTuple):
                executables.extend(stat)
                if to_be_fetched:
                    collectors[nm] = functools.partial(lambda x: tuple(_result_cache[v.data] for v in x), stat)
                    to_fetch_tileables.extend(stat)
            else:
                executables.append(stat)
                if to_be_fetched:
                    collectors[nm] = functools.partial(lambda x: _result_cache[x.data], stat)
                    to_fetch_tileables.append(stat)

    results = ExecutableTuple(executables)
    if not fetch_only:
        results = ExecutableTuple(executables).execute()

    if fetch_with_remote or (fetch_with_remote is None and len(to_fetch_tileables) > 10):
        results = pickle.loads(mr.spawn(lambda *args: pickle.dumps(tuple(a.fetch() for a in args)),
                               args=tuple(to_fetch_tileables)).execute().fetch())
    else:
        results = ExecutableTuple(to_fetch_tileables)
        results = results.fetch()

    with enter_mode(build=True):
        for x, r in zip(to_fetch_tileables, results):
            _result_cache[x.data] = r
        for nm, collector in collectors.items():
            d[nm] = collector()

    return d
