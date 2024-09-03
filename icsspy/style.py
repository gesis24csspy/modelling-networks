import importlib.resources as resources

import matplotlib.pyplot as plt


def set_style(style="statistical_rethinking"):
    if style == "statistical_rethinking":
        style_path = resources.files("icsspy.mplstyles") / "mcelreath.mplstyle"
        plt.style.use(str(style_path))
    else:
        plt.style.use("fivethirtyeight")
