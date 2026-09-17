# UI package
from .dashboard import render_dashboard
from .comparison import render_comparison_page
from .recommendations import render_recommendations_page
from .how_it_works import render_how_it_works_page
from .styles import DARK_THEME_CSS

__all__ = [
    "render_dashboard",
    "render_comparison_page",
    "render_recommendations_page",
    "render_how_it_works_page",
    "DARK_THEME_CSS"
]
