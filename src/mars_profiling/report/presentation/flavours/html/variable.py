from mars_profiling.report.presentation.core import Variable
from mars_profiling.report.presentation.flavours.html import templates


class HTMLVariable(Variable):
    def render(self):
        return templates.template("variable.html").render(**self.content)
