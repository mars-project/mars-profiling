import numpy as np
import mars.dataframe as md

from mars_profiling import ProfileReport


def test_mars_simple():
    df = md.DataFrame({
        'a': np.random.rand(100),
        'b': np.random.choice(['a', 'b', 'c']),
    }, chunk_size=13)

    profile = ProfileReport(df, title="Pandas Profiling Report", minimal=True)
    profile.to_html()
