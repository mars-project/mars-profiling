from pathlib import Path

import pandas as pd

from mars_profiling import ProfileReport
from mars_profiling.utils.cache import cache_file

if __name__ == "__main__":
    file_name = cache_file(
        "websites.csv",
        "https://raw.githubusercontent.com/berkmancenter/url-lists/master/lists/et.csv",
    )

    df = pd.read_csv(file_name, parse_dates=["date_added"])
    profile = ProfileReport(
        df,
        title="Website Inaccessibility Test Lists",
        correlations={"cramers": {"calculate": False}},
    )
    profile.to_file(Path("./website_inaccessibility_report.html"))
