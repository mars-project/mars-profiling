from mars_profiling.report.presentation.core import Collapse
from mars_profiling.report.presentation.flavours.html import templates


class HTMLCollapse(Collapse):
    def render(self):
        return templates.template("collapse.html").render(**self.content)
