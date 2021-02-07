from typing import Dict, Sequence

from mars_profiling.utils.mars import fetch_mars_dict_results


def freq_table(freqtable, n: int, max_number_to_print: int) -> Sequence[Dict]:
    """Render the rows for a frequency table (value, count).

    Args:
      freqtable: The frequency table.
      n: The total number of values.
      max_number_to_print: The maximum number of observations to print.

    Returns:
        The rows of the frequency table.
    """
    freq_table_stats = {
        'agg_results': freqtable.agg(['sum', 'size', 'count']),
        'head_table': freqtable.iloc[0:max_number_to_print],
        'first_value': freqtable.iloc[0],
    }
    fetch_mars_dict_results(freq_table_stats)

    tail_table = freqtable.iloc[max_number_to_print:]

    # TODO: replace '' by '(Empty)' ?

    if max_number_to_print > n:
        max_number_to_print = n

    if max_number_to_print < freq_table_stats['agg_results']['size']:
        freq_other_stats = {
            'freq_other': tail_table.sum(),
            'min_freq': tail_table.iloc[0],
        }
        fetch_mars_dict_results(freq_other_stats)
        freq_other = freq_other_stats['freq_other']
        min_freq = freq_other_stats['min_freq']
    else:
        freq_other = 0
        min_freq = 0

    freq_missing = n - freq_table_stats['agg_results']['sum']
    # No values
    if freq_table_stats['agg_results']['size'] == 0:
        return []

    max_freq = max(freq_table_stats['first_value'], freq_other, freq_missing)

    # TODO: Correctly sort missing and other
    # No values
    if max_freq == 0:
        return []

    rows = []
    for label, freq in freq_table_stats['head_table'].items():
        rows.append(
            {
                "label": label,
                "width": freq / max_freq,
                "count": freq,
                "percentage": float(freq) / n,
                "n": n,
                "extra_class": "",
            }
        )

    if freq_other > min_freq:
        other_count = str(freq_table_stats['agg_results']['count'] - max_number_to_print)
        rows.append(
            {
                "label": f"Other values ({other_count})",
                "width": freq_other / max_freq,
                "count": freq_other,
                # Hack for tables with combined...
                "percentage": min(float(freq_other) / n, 1.0),
                "n": n,
                "extra_class": "other",
            }
        )

    if freq_missing > min_freq:
        rows.append(
            {
                "label": "(Missing)",
                "width": freq_missing / max_freq,
                "count": freq_missing,
                "percentage": float(freq_missing) / n,
                "n": n,
                "extra_class": "missing",
            }
        )

    return rows


def extreme_obs_table(freqtable, number_to_print, n, ascending=True) -> list:
    """Similar to the frequency table, for extreme observations.

    Args:
      freqtable: The frequency table.
      number_to_print: The number of observations to print.
      n: The total number of observations.
      ascending: The ordering of the observations (Default value = True)

    Returns:
        The HTML rendering of the extreme observation table.
    """
    # If it's mixed between base types (str, int) convert to str. Pure "mixed" types are filtered during type
    # discovery
    # TODO: should be in cast?
    # if "mixed" in freqtable.index.inferred_type:
    #     freqtable.index = freqtable.index.astype(str)

    sorted_freqtable = freqtable.sort_index(ascending=ascending)
    obs_to_print = sorted_freqtable.iloc[:number_to_print]
    freq_dict = {
        'obs_to_print': obs_to_print,
        'max_freq': obs_to_print.max(),
    }
    fetch_mars_dict_results(freq_dict)

    obs_to_print = freq_dict['obs_to_print']
    max_freq = freq_dict['max_freq']

    rows = []
    for label, freq in obs_to_print.items():
        rows.append(
            {
                "label": label,
                "width": freq / max_freq if max_freq != 0 else 0,
                "count": freq,
                "percentage": float(freq) / n,
                "extra_class": "",
                "n": n,
            }
        )

    return rows
