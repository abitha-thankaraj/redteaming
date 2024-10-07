import matplotlib
from typing import Optional


class IrisColormap(matplotlib.colors.ListedColormap):
    """Official IRIS lab plotting color palette. Palette author: Chelsea Finn.

    Recommended usage:
      - Index 0: Your proposed method.
      - Index 1/2: Main comparisons 1/2.
      - Index 3: Main comparison 3 or ablation.
      - Index 4: Main comparison 4 or oracle.
      - Index 5: Main comparison 5. Think about if you really need 6 colors in a
          plot.
    """

    def __init__(self, N: Optional[int] = None):
        """See matplotlib.colors.Colormap for N argument docs."""
        hex_colors = ["#FF6150", "#134E6F", "#1AC0C6", "#FFA822", "#DEE0E6", "#091A29"]

        rgb_colors = [matplotlib.colors.to_rgb(c) for c in hex_colors]
        super().__init__(rgb_colors, name="iris", N=N)
