"""Pair viewer Dash app."""

import base64
import re
from pathlib import Path

from dash import Dash, Input, Output, State, ctx, html


class PairViewer:
    """Dash app for inspecting raw/graded/ungraded image pairs."""

    def __init__(
        self,
        raw_dir: str,
        graded_dir: str,
        ungraded_dir: str,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.graded_dir = Path(graded_dir)
        self.ungraded_dir = Path(ungraded_dir)

    @staticmethod
    def _natural_key(filename: str) -> tuple:
        """Sort key that extracts numbers for natural ordering."""
        parts = re.split(r"(\d+)", filename)
        return tuple(int(p) if p.isdigit() else p for p in parts)

    def list_pairs(self) -> list[str]:
        """Return naturally-sorted filenames present in all three directories."""
        raw_files = {f.name for f in self.raw_dir.iterdir() if f.is_file()}
        graded_files = {f.name for f in self.graded_dir.iterdir() if f.is_file()}
        ungraded_files = {f.name for f in self.ungraded_dir.iterdir() if f.is_file()}
        common = raw_files & graded_files & ungraded_files
        return sorted(common, key=self._natural_key)

    def _encode_image(self, path: Path) -> str:
        """Read an image file and return a base64 data URI."""
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _build_panels(self, filename: str) -> list:
        """Build the three image panels for a given pair filename."""
        raw_src = self._encode_image(self.raw_dir / filename)
        graded_src = self._encode_image(self.graded_dir / filename)
        ungraded_src = self._encode_image(self.ungraded_dir / filename)

        font_family = (
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', "
            "Roboto, Oxygen, Ubuntu, sans-serif"
        )
        panel_style = {"flex": "1", "padding": "16px", "textAlign": "center"}
        label_style = {
            "color": "#e8e8e8",
            "fontFamily": font_family,
            "fontWeight": "500",
            "letterSpacing": "0.05em",
            "textTransform": "uppercase",
            "fontSize": "14px",
        }
        image_style = {"width": "100%", "borderRadius": "4px"}

        return [
            html.Div(
                style=panel_style,
                children=[
                    html.H4("Raw", style=label_style),
                    html.Img(src=raw_src, style=image_style),
                ],
            ),
            html.Div(
                style=panel_style,
                children=[
                    html.H4("Graded", style=label_style),
                    html.Img(src=graded_src, style=image_style),
                ],
            ),
            html.Div(
                style=panel_style,
                children=[
                    html.H4("Ungraded", style=label_style),
                    html.Img(src=ungraded_src, style=image_style),
                ],
            ),
        ]

    def run(self) -> None:
        """Start the Dash server."""
        pairs = self.list_pairs()
        total = len(pairs)
        first = pairs[0]

        bg_color = "#1a1a1a"
        text_color = "#e8e8e8"
        font_family = (
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', "
            "Roboto, Oxygen, Ubuntu, sans-serif"
        )
        button_style = {
            "backgroundColor": "#2a2a2a",
            "color": text_color,
            "border": "1px solid #3a3a3a",
            "borderRadius": "4px",
            "padding": "8px 20px",
            "fontFamily": font_family,
            "fontSize": "14px",
            "cursor": "pointer",
            "margin": "0 8px",
        }

        app = Dash(__name__)
        app.layout = html.Div(
            style={
                "backgroundColor": bg_color,
                "color": text_color,
                "fontFamily": font_family,
                "minHeight": "100vh",
                "padding": "32px",
            },
            children=[
                html.H1(
                    "Pair Inspection",
                    style={
                        "textAlign": "center",
                        "fontWeight": "300",
                        "letterSpacing": "0.02em",
                    },
                ),
                html.H3(
                    id="filename-label",
                    children=f"Showing: {first}  (1 of {total})",
                    style={
                        "textAlign": "center",
                        "fontWeight": "400",
                        "color": "#a0a0a0",
                    },
                ),
                html.Div(
                    style={"textAlign": "center", "marginTop": "16px"},
                    children=[
                        html.Button("← Previous", id="prev-btn", style=button_style),
                        html.Button("Next →", id="next-btn", style=button_style),
                    ],
                ),
                html.Div(
                    id="panels",
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "marginTop": "24px",
                    },
                    children=self._build_panels(first),
                ),
                html.Div(id="current-index", children="0", style={"display": "none"}),
            ],
        )

        @app.callback(
            Output("panels", "children"),
            Output("filename-label", "children"),
            Output("current-index", "children"),
            Input("prev-btn", "n_clicks"),
            Input("next-btn", "n_clicks"),
            State("current-index", "children"),
            prevent_initial_call=True,
        )
        def navigate(prev_clicks, next_clicks, current_index):
            idx = int(current_index)
            triggered = ctx.triggered_id

            if triggered == "prev-btn":
                idx = (idx - 1) % total
            elif triggered == "next-btn":
                idx = (idx + 1) % total

            filename = pairs[idx]
            return (
                self._build_panels(filename),
                f"Showing: {filename}  ({idx + 1} of {total})",
                str(idx),
            )

        app.run(debug=True)