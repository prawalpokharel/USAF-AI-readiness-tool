from __future__ import annotations
from typing import Dict, List, Tuple
import math
import matplotlib.pyplot as plt


def radar_chart(domain_labels: List[str], domain_scores: List[int]):
    """Return a matplotlib Figure for a radar chart."""
    if len(domain_labels) != len(domain_scores):
        raise ValueError("labels and scores must match length")

    # close the loop
    labels = domain_labels + [domain_labels[0]]
    scores = domain_scores + [domain_scores[0]]

    angles = [n / float(len(domain_labels)) * 2 * math.pi for n in range(len(domain_labels))]
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]], domain_labels)
    ax.set_ylim(0, 100)
    ax.plot(angles, scores, linewidth=2)
    ax.fill(angles, scores, alpha=0.2)
    ax.grid(True)
    return fig
