from ipywidgets import widgets

from mars_profiling.report.presentation.core.root import Root


class WidgetRoot(Root):
    def render(self):
        return widgets.VBox(
            [self.content["body"].render(), self.content["footer"].render()]
        )
