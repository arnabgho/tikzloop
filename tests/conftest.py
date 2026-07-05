import pytest

from tikzloop.sandbox import compile_tikz


@pytest.fixture(scope="session", autouse=True)
def warm_tectonic_cache():
    # First-ever compile downloads LaTeX packages and can take minutes;
    # warm the cache with a generous timeout so tests measure compile
    # behavior, not network.
    compile_tikz(r"\begin{tikzpicture}\node{warm};\end{tikzpicture}", timeout=300)
