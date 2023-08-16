from pathlib import Path

import h5py
import ipywidgets as widgets
import numpy as np
from bokeh.layouts import row
from bokeh.plotting import figure, show
from IPython.display import clear_output
from ipywidgets import Layout

label = [
    "mass_1",
    "mass_2",
    "redshift",
    "psi",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "ra",
    "dec",
    "theta_jn",
    "snr",
]
output_figure = widgets.Output()
fig = []


class OneDHistogram:
    def __init__(self, foreground: Path, rejected: Path):
        self.foreground = self.load_path(foreground)
        self.rejected = self.load_path(rejected)
        self.all_params = {
            k: np.concatenate(((self.foreground[k], self.rejected[k])))
            for k in self.rejected.keys()
        }

    def load_path(self, path: Path) -> dict[float]:
        with h5py.File(path, "r") as f:
            file = {
                k: f["parameters"][k][:] for k in list(f["parameters"].keys())
            }

        return file

    def initialize_sources(self):
        self.hist_dict_f = self.create_dict(self.foreground)
        self.hist_dict_all = self.create_dict(self.all_params)
        self.hist_dict_r = self.create_dict(self.rejected)

    def create_dict(self, data: dict[float]) -> dict[float]:
        """
        Format: hist_dict[keyname][hist]
                hist_dict[keyname][edges]
        """
        hist_dict = {
            k: np.histogram(data[k], bins=100, density=True) for k in label
        }

        return hist_dict

    def dropdown(self):
        p_f = self.get_layout_fore(
            self.x_dropdown_f.children[0].value, self.foreground
        )
        p_all = self.get_layout_all(
            self.x_dropdown_all.children[0].value, self.all_params
        )
        p_r = self.get_layout_rej(
            self.x_dropdown_r.children[0].value, self.rejected
        )
        fig[0] = p_f
        fig[1] = p_all
        fig[2] = p_r

        with output_figure:
            clear_output(True)
            show(row(fig[0], fig[1], fig[2]))

    def get_layout_fore(self, x_var: str, data: dict[float]):
        title = "Normalized Foreground Distribution of {}".format(x_var)
        hist_dict = self.hist_dict_f

        hist = hist_dict[x_var][0]
        edges = hist_dict[x_var][1]

        fig_fore = figure(
            title=title, x_axis_label=x_var, width=350, height=250
        )
        fig_fore.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color="lightskyblue",
            line_color="white",
        )
        fig_fore.y_range.start = 0
        fig_fore.outline_line_width = 2
        fig_fore.outline_line_color = "black"
        fig_fore.title.text_font_size = "8pt"

    def get_layout_all(self, x_var: str, data: dict[float]):
        title = "Normalized Foreground + Rejected Distribution of {}".format(
            x_var
        )
        hist_dict = self.initialize_sources().hist_dict_all

        hist = hist_dict[x_var][0]
        edges = hist_dict[x_var][1]

        fig_all = figure(
            title=title, x_axis_label=x_var, width=350, height=250
        )
        fig_all.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color="lightskyblue",
            line_color="white",
        )
        fig_all.y_range.start = 0
        fig_all.outline_line_width = 2
        fig_all.outline_line_color = "black"
        fig_all.title.text_font_size = "8pt"

    def get_layout_rej(self, x_var: str, data: dict[float]):
        title = "Normalized Rejected Distribution of {}".format(x_var)
        hist_dict = self.initialize_sources().hist_dict_r

        hist = hist_dict[x_var][0]
        edges = hist_dict[x_var][1]

        fig_rej = figure(
            title=title, x_axis_label=x_var, width=350, height=250
        )
        fig_rej.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color="lightskyblue",
            line_color="white",
        )
        fig_rej.y_range.start = 0
        fig_rej.outline_line_width = 2
        fig_rej.outline_line_color = "black"
        fig_rej.title.text_font_size = "8pt"

    def get_layout(self):
        x_dropdown_f = widgets.interactive(self.dropdown, x=label)
        x_dropdown_f.children[0].description = "Foreground"
        x_dropdown_f.children[0].value = "mass_1"
        x_dropdown_f.children[0].layout = Layout(
            width="200px", margin="0px 150px 0px 0px"
        )

        x_dropdown_all = widgets.interactive(self.dropdown, x=label)
        x_dropdown_all.children[0].description = "Fore + Reject"
        x_dropdown_all.children[0].value = "mass_1"
        x_dropdown_all.children[0].layout = Layout(width="200px")
        x_dropdown_all.children[0].layout = Layout(
            width="200px", margin="0px 150px 0px 0px"
        )

        x_dropdown_r = widgets.interactive(self.dropdown, x=label)
        x_dropdown_r.children[0].description = "Reject"
        x_dropdown_r.children[0].value = "mass_1"
        x_dropdown_r.children[0].layout = Layout(width="200px")

        menu = widgets.HBox(
            [x_dropdown_f, x_dropdown_all, x_dropdown_r],
            layout=widgets.Layout(margin="0px 0px 0px 37px"),
        )
        app_layout = widgets.Layout(
            display="flex",
            flex_direction="column",
            align_items="stretch",
            border="solid",
            width="100%",
            margin="5px 5px 5px 5px",
        )
        app = widgets.VBox(
            [
                self.get_layout_fore,
                self.get_layout_all,
                self.get_layout_rej,
                menu,
            ],
            layout=app_layout,
        )

        return app
