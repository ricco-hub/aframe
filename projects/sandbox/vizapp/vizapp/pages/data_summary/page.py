from vizapp.pages.data_summary.param_hist import OneDHistogram
from vizapp.pages.page import Page


class DataSummaryPage(Page):
    def __init__(self, app):
        super().__init__(app)
        self.param_hist = OneDHistogram(
            self.app.foreground, self.app.rejected_params
        )

    def get_layout(self):
        return self.param_hist.get_layout()
