"""Safe TikZ compile + render sandbox.

The host VM/process is treated as disposable; what we defend against is
model-generated LaTeX taking down the *loop*: infinite loops (timeout),
shell access (tectonic --untrusted + a pre-compile denylist), and
resource blowups (memory rlimit on Linux, capped render resolution).
"""

from __future__ import annotations

import re
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

STANDALONE_TEMPLATE = """\\documentclass[tikz]{{standalone}}
{packages}
\\begin{{document}}
{body}
\\end{{document}}
"""

# Packages auto-included when their environments/commands appear in the body
# (mirrors TikZilla's "dynamic package inclusion").
DYNAMIC_PACKAGES = {
    r"\\begin\{circuitikz\}": "\\usepackage{circuitikz}",
    r"\\begin\{tikzcd\}": "\\usepackage{tikz-cd}",
    r"\\begin\{axis\}": "\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}",
}

# TikZ libraries auto-loaded when their keys/commands appear in the body.
DYNAMIC_LIBRARIES = {
    r"\b(diamond|ellipse callout|regular polygon|star,|trapezium)\b": "shapes.geometric",
    r"\\matrix|\bmatrix of\b": "matrix",
    r"-\{?(Latex|Stealth|Straight Barb)": "arrows.meta",
    r"\b(above|below|left|right)=.*\bof\b": "positioning",
    r"\bdecorate\b|decoration=": "decorations.pathmorphing",
    r"\bcalc\b|\(\$": "calc",
}

# Preamble-only commands that models often emit alongside the tikzpicture;
# hoisted out of the body when wrapping in a standalone document.
PREAMBLE_LINE = re.compile(
    r"^[ \t]*\\(usetikzlibrary|usepackage|pgfplotsset|tikzset)\b.*$", re.MULTILINE
)

# Commands that can touch the filesystem or shell. Tectonic --untrusted
# already blocks shell escape; this is belt-and-suspenders and also keeps
# generations self-contained (no external file deps), as in DaTikZ filtering.
FORBIDDEN = re.compile(
    r"\\(write18|input\b|include\b|openout|openin|read\b|ShellEscape|directlua)"
)

MAX_LOG_CHARS = 2000


@dataclass
class CompileResult:
    ok: bool
    log: str = ""
    png_path: Path | None = None
    image: Image.Image | None = field(default=None, repr=False)
    seconds: float = 0.0


def ensure_standalone(code: str) -> str:
    """Wrap a bare tikzpicture in a standalone document if needed.

    Preamble-only commands (\\usetikzlibrary, \\usepackage, ...) found in the
    body are hoisted into the preamble; common packages and TikZ libraries
    are detected from the body and added automatically.
    """
    if "\\documentclass" in code:
        return code

    hoisted = [m.group(0).strip() for m in PREAMBLE_LINE.finditer(code)]
    body = PREAMBLE_LINE.sub("", code).strip()

    preamble = list(hoisted)
    preamble += [
        pkg for pat, pkg in DYNAMIC_PACKAGES.items() if re.search(pat, body)
    ]
    libs = sorted(
        lib for pat, lib in DYNAMIC_LIBRARIES.items()
        if re.search(pat, body) and not any(lib in h for h in hoisted)
    )
    if libs:
        preamble.append("\\usetikzlibrary{" + ",".join(libs) + "}")

    return STANDALONE_TEMPLATE.format(packages="\n".join(preamble), body=body)


def _limit_memory():
    # RLIMIT_AS is enforced on Linux (Colab); macOS ignores it, where the
    # timeout is the effective backstop.
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_AS, (2**31, 2**31))


def compile_tikz(
    code: str,
    workdir: Path | str | None = None,
    timeout: float = 25.0,
    dpi: int = 150,
    load_image: bool = True,
) -> CompileResult:
    """Compile TikZ source to PDF with tectonic, render page 1 to PNG.

    Returns CompileResult with ok=False and a truncated log on any failure
    (forbidden command, compile error, timeout, render error).
    """
    import time

    if m := FORBIDDEN.search(code):
        return CompileResult(ok=False, log=f"REJECTED: forbidden command \\{m.group(1)}")

    code = ensure_standalone(code)
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory() as td:
        out = Path(workdir) if workdir else Path(td)
        out.mkdir(parents=True, exist_ok=True)
        tex = out / "fig.tex"
        tex.write_text(code)
        try:
            proc = subprocess.run(
                ["tectonic", "--untrusted", "--chatter", "minimal",
                 "--outdir", str(out), str(tex)],
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limit_memory,
            )
        except subprocess.TimeoutExpired:
            return CompileResult(ok=False, log="TIMEOUT: compilation exceeded limit",
                                 seconds=time.monotonic() - t0)
        except FileNotFoundError:
            raise RuntimeError("tectonic not found — `brew install tectonic`") from None

        pdf = out / "fig.pdf"
        if proc.returncode != 0 or not pdf.exists():
            log = (proc.stderr or "") + (proc.stdout or "")
            return CompileResult(ok=False, log=log[-MAX_LOG_CHARS:],
                                 seconds=time.monotonic() - t0)

        png = out / "render.png"
        try:
            subprocess.run(
                ["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1",
                 "-singlefile", str(pdf), str(out / "render")],
                capture_output=True, timeout=15, check=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            return CompileResult(ok=False, log=f"RENDER FAILED: {e}",
                                 seconds=time.monotonic() - t0)

        image = None
        if load_image:
            with Image.open(png) as im:
                image = im.convert("RGB").copy()
        persistent = workdir is not None
        return CompileResult(
            ok=True,
            png_path=png if persistent else None,
            image=image,
            seconds=time.monotonic() - t0,
        )
