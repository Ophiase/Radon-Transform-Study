from typing import Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np

###################################################################################

MEDICAL_GRAYSCALE = [[0, 'rgb(0,0,0)'], [1, 'rgb(255,255,255)']]
LAYOUT_CONFIG = {
    'height': 800,
    'width': 1200,
    'margin': dict(t=40, b=20),
    'plot_bgcolor': 'rgba(0,0,0,0.9)',
    'paper_bgcolor': 'rgba(0,0,0,0.9)',
    'font': dict(color='white', size=12),
    'dragmode': 'pan'  # Validated Plotly parameter
}
AXIS_CONFIG = {
    'showgrid': False,
    'zeroline': False,
    'ticks': '',
    'showticklabels': False,
    'scaleanchor': "x",
    'scaleratio': 1
}
WINDOW_PRESETS = {
    'CT Window': (0, 1),
    'Bone Window': (0.6, 1.0),
    'Lung Window': (0.0, 0.3)
}

###################################################################################


def plot_results_matplotlib(original: np.ndarray, sinogram: np.ndarray,
                            fbp: np.ndarray, bp: np.ndarray,
                            metrics_fbp: Tuple[float, float, float],
                            metrics_bp: Tuple[float, float, float],
                            title_suffix: str = "") -> None:
    """Enhanced visualization with custom titles"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 14))

    images = [
        (original, f"Original Phantom\n{title_suffix}"),
        (sinogram, "Sinogram (Radon Transform)"),
        (fbp,
         f"Filtered Back Projection\nMSE: {metrics_fbp[0]:.4f}, SSIM: {metrics_fbp[2]:.3f}"),
        (bp,
         f"Simple Back Projection\nMSE: {metrics_bp[0]:.4f}, SSIM: {metrics_bp[2]:.3f}")
    ]

    for ax, (img, title) in zip(axs.flat, images):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

###################################################################################


def _create_figure(title_suffix: str) -> go.Figure:
    """Initialize subplot grid with validated parameters"""
    return make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Original Phantom<br>{title_suffix}",
            "Sinogram (Radon Transform)",
            f"Filtered Back Projection<br>",
            f"Simple Back Projection<br>"
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
               [{'type': 'heatmap'}, {'type': 'heatmap'}]]
    )


def _add_heatmap_trace(fig: go.Figure, image: np.ndarray,
                       row: int, col: int, name: str,
                       clamped: bool = True
                       ) -> None:
    """Add validated heatmap trace to figure"""
    fig.add_trace(
        go.Heatmap(
            z=image,
            colorscale=MEDICAL_GRAYSCALE,
            showscale=False,
            hoverinfo="x+y+z",
            zmin=0 if clamped else None,
            zmax=1 if clamped else None,
            name=name
        ),
        row=row,
        col=col
    )


def _configure_axes(fig: go.Figure) -> None:
    """Apply consistent axis configuration"""
    fig.update_xaxes(AXIS_CONFIG)
    fig.update_yaxes(AXIS_CONFIG)


def _add_metrics_annotations(fig: go.Figure,
                             metrics_fbp: Tuple[float, float, float] = None,
                             metrics_bp: Tuple[float, float, float] = None) -> None:
    """Dynamically update subplot titles with metrics"""
    if metrics_fbp is not None:
        fig.layout.annotations[2].text += f"MSE: {metrics_fbp[0]:.4f}, SSIM: {metrics_fbp[2]:.3f}"
    if metrics_bp is not None:
        fig.layout.annotations[3].text += f"MSE: {metrics_bp[0]:.4f}, SSIM: {metrics_bp[2]:.3f}"


def create_windowing_buttons() -> list:
    """Generate validated windowing control buttons"""
    return [dict(
        label=name,
        method='update',
        args=[{'zmin': z_range[0], 'zmax': z_range[1]}]
    ) for name, z_range in WINDOW_PRESETS.items()]


def _add_controls(fig: go.Figure) -> None:
    """Add interactive controls with validated parameters"""
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=1.0,
            y=1.15,
            buttons=[
                dict(
                    label="Reset Views",
                    method="relayout",
                    args=["autosize", True]
                )
            ] + create_windowing_buttons()
        )]
    )


def plot_results(
        original: np.ndarray, sinogram: np.ndarray,
        fbp: np.ndarray, bp: np.ndarray,
        metrics_fbp: Tuple[float, float, float] = None,
        metrics_bp: Tuple[float, float, float] = None,
        title_suffix: str = "",
        show: bool = True) -> Optional[go.Figure]:
    """
    Robust medical imaging visualization with Plotly
    Features validated parameters and modular components
    """
    try:
        # Initialize figure
        fig = _create_figure(title_suffix)

        # Add image data
        _add_heatmap_trace(fig, original, 1, 1, "Original")
        _add_heatmap_trace(fig, sinogram, 1, 2, "Sinogram", clamped=False)
        _add_heatmap_trace(fig, fbp, 2, 1, "FBP")
        _add_heatmap_trace(fig, bp, 2, 2, "BP", clamped=False)

        # Add metrics to titles
        _add_metrics_annotations(fig, metrics_fbp, metrics_bp)

        # Configure layout
        fig.update_layout(LAYOUT_CONFIG)
        _configure_axes(fig)

        # Add controls
        _add_controls(fig)

        if show:
            fig.show()
            return None
        return fig

    except Exception as e:
        print(f"Visualization error: {str(e)}")
        if not show:
            return go.Figure()  # Return empty figure for error handling
        return None
