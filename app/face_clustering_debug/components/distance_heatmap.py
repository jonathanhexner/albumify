"""Distance matrix heatmap component."""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def render_distance_heatmap(
    matrix: np.ndarray,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    title: str = "Distance Matrix",
) -> None:
    """Render a colour-coded distance heatmap.

    Args:
        matrix: 2-D distance matrix.
        row_labels: Y-axis tick labels (falls back to indices if length mismatches).
        col_labels: X-axis tick labels (falls back to indices if length mismatches).
        title: Chart title.
    """
    n_rows, n_cols = matrix.shape

    # Always derive from matrix shape; ignore provided labels if they don't match
    eff_row = (row_labels if row_labels and len(row_labels) == n_rows
               else [str(i) for i in range(n_rows)])
    eff_col = (col_labels if col_labels and len(col_labels) == n_cols
               else [str(i) for i in range(n_cols)])

    fig, ax = plt.subplots(figsize=(max(4, n_cols * 0.5), max(3, n_rows * 0.5)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax.set_title(title)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(eff_col, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(eff_row, fontsize=7)

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=6, color="black")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
