import pandas as pd

from mars_profiling import ProfileReport
from mars_profiling.model.base import Variable


def test_issue351():
    data = pd.DataFrame(["Jan", 1]).set_index(0)

    profile = ProfileReport(data)
    assert (
        profile.get_description()["variables"]["0"]["type"]
        == Variable.S_TYPE_UNSUPPORTED
    )
