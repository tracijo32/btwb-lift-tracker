from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from dataclasses import dataclass

@dataclass(frozen=True)
class ChunkTheme:
    """
    Colors for the Plotly theme used by this project.

    This is intentionally a dataclass (not a dict) so all colors can be overridden
    by constructing a new instance.
    """

    # Semantic layout colors
    bg: str = "#FDFBE4"
    text: str = "#657b83"
    grid: str = "#d8d4b8"

    # Transparent/overlay colors
    transparent_bg: str = "rgba(0,0,0,0)"
    hoverlabel_bg_transparent: str = "#fffef5"

    # Chunk / Pygments-inspired palette
    blue: str = "#4070A0"
    orange: str = "#C65D09"
    green: str = "#007020"
    purple: str = "#BB60D5"
    cyan: str = "#0E84B5"
    brown: str = "#902000"
    dark_blue: str = "#06287E"
    sea_green: str = "#208050"

    # Optional explicit overrides for Plotly sequences (otherwise derived)
    colorway: tuple[str, ...] | None = None
    colorscale: tuple[tuple[float, str], ...] | None = None

    @property
    def resolved_colorway(self) -> tuple[str, ...]:
        if self.colorway is not None:
            return self.colorway
        return (
            self.blue,
            self.orange,
            self.green,
            self.purple,
            self.cyan,
            self.brown,
            self.dark_blue,
            self.sea_green,
        )

    @property
    def resolved_colorscale(self) -> tuple[tuple[float, str], ...]:
        if self.colorscale is not None:
            return self.colorscale
        return (
            (0.00, self.bg),
            (0.20, self.blue),
            (0.40, self.cyan),
            (0.60, self.green),
            (0.80, self.sea_green),
            (1.00, self.dark_blue),
        )

    # Backwards-compatible aliases (and common spelling variants)
    @property
    def light_blue(self) -> str:
        return self.blue

    @property
    def light_grey(self) -> str:
        return self.grid

    @property
    def dark_grey(self) -> str:
        return self.text

    def __getitem__(self, key: str):
        """
        Allow legacy code to keep using CHUNK_THEME["text"]-style access.
        """
        if key == "colorway":
            return list(self.resolved_colorway)
        if key == "colorscale":
            return [[p, c] for p, c in self.resolved_colorscale]
        return getattr(self, key)


CHUNK_THEME = ChunkTheme()


def build_chunk_template(
    transparent: bool = True,
    font_family: str = (
        'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    ),
    theme: ChunkTheme = CHUNK_THEME,
) -> go.layout.Template:
    """
    Build a Plotly template that matches the Pelican Chunk theme.

    Parameters
    ----------
    transparent
        If True, use transparent chart backgrounds so the page background shows
        through. This is usually best for website embeds.
        If False, use the Chunk code-block-like cream background.
    font_family
        CSS-style font family string for Plotly text.
    """
    bg = theme.transparent_bg if transparent else theme.bg

    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            colorway=list(theme.resolved_colorway),
            font=dict(
                family=font_family,
                color=theme.text,
                size=14,
            ),
            title=dict(
                font=dict(
                    color=theme.text,
                    size=22,
                ),
                x=0.02,
                xanchor="left",
            ),
            margin=dict(l=60, r=30, t=60, b=50),
            hovermode="x unified",
            legend=dict(
                bgcolor=theme.transparent_bg,
                borderwidth=0,
                font=dict(color=theme.text),
                title=dict(font=dict(color=theme.text)),
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor=theme.text,
                tickcolor=theme.text,
                ticks="outside",
                tickfont=dict(color=theme.text),
                title=dict(font=dict(color=theme.text)),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=theme.grid,
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor=theme.text,
                tickcolor=theme.text,
                ticks="outside",
                tickfont=dict(color=theme.text),
                title=dict(font=dict(color=theme.text)),
            ),
            hoverlabel=dict(
                bgcolor=theme.hoverlabel_bg_transparent if transparent else theme.bg,
                bordercolor=theme.grid,
                font=dict(color=theme.text),
            ),
        )
    )


def register_chunk_template(
    name: str = "chunk",
    *,
    transparent: bool = True,
    set_default: bool = True,
) -> go.layout.Template:
    """
    Register the Chunk template with Plotly.

    Parameters
    ----------
    name
        Template name to register under.
    transparent
        Whether to use transparent backgrounds.
    set_default
        If True, make this the default Plotly template for the session.
    """
    template = build_chunk_template(transparent=transparent)
    pio.templates[name] = template
    if set_default:
        pio.templates.default = name
    return template


def apply_chunk_styling(fig: go.Figure) -> go.Figure:
    """
    Apply a few extra per-figure settings that help charts feel cleaner on a site.
    """
    fig.update_layout(template="chunk")
    fig.update_traces(marker_line_width=0)
    return fig