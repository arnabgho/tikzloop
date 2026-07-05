from tikzloop.sandbox import compile_tikz, ensure_standalone

VALID = r"""
\begin{tikzpicture}
  \node[draw, circle, fill=red!30] (a) at (0,0) {A};
  \node[draw, rectangle] (b) at (2,0) {B};
  \draw[->, thick] (a) -- (b);
\end{tikzpicture}
"""

BROKEN = r"""
\begin{tikzpicture}
  \node[draw (a) at (0,0) {A};
\end{tikzpicture}
"""

INFINITE = r"""
\begin{tikzpicture}
\end{tikzpicture}
\loop\iftrue\repeat
"""

SHELL_ESCAPE = r"""
\begin{tikzpicture}
  \node {x};
\end{tikzpicture}
\immediate\write18{touch /tmp/pwned}
"""


def test_valid_compiles():
    result = compile_tikz(VALID)
    assert result.ok, result.log
    assert result.image is not None
    assert result.image.width > 10


def test_broken_fails_with_log():
    result = compile_tikz(BROKEN)
    assert not result.ok
    assert result.log.strip()


def test_infinite_loop_times_out():
    result = compile_tikz(INFINITE, timeout=5)
    assert not result.ok
    assert "TIMEOUT" in result.log


def test_shell_escape_rejected_before_compile():
    result = compile_tikz(SHELL_ESCAPE)
    assert not result.ok
    assert "REJECTED" in result.log


def test_ensure_standalone_wraps_and_detects_packages():
    wrapped = ensure_standalone(r"\begin{tikzcd} A \arrow[r] & B \end{tikzcd}")
    assert wrapped.startswith("\\documentclass[tikz]{standalone}")
    assert "\\usepackage{tikz-cd}" in wrapped
    full_doc = "\\documentclass{article}\nx"
    assert ensure_standalone(full_doc) == full_doc


def test_preamble_lines_hoisted_and_libraries_detected():
    code = (
        "\\usetikzlibrary{arrows.meta}\n"
        "\\begin{tikzpicture}\n"
        "  \\node[draw, diamond, below=1cm of a] {D};\n"
        "\\end{tikzpicture}"
    )
    wrapped = ensure_standalone(code)
    preamble = wrapped.split("\\begin{document}")[0]
    assert "\\usetikzlibrary{arrows.meta}" in preamble
    assert "shapes.geometric" in preamble
    assert "positioning" in preamble


def test_diamond_node_compiles_via_dynamic_library():
    result = compile_tikz(
        r"\begin{tikzpicture}\node[draw, diamond, fill=blue!20]{V};\end{tikzpicture}"
    )
    assert result.ok, result.log
