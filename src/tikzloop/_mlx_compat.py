"""Import-time workaround for mlx-lm <= 0.31.3 with transformers >= 4.5x.

mlx_lm.tokenizer_utils calls AutoTokenizer.register("NewlineTokenizer", ...)
with a string config key, which current transformers rejects with
AttributeError ('str' has no __module__). The registration is only needed
for mlx-lm's own NewlineTokenizer models, none of which we use; swallow the
failure so `import mlx_lm` succeeds. Remove once upstream fixes it.

Must be imported BEFORE mlx_lm / mlx_vlm.
"""

from transformers import AutoTokenizer

_orig_register = AutoTokenizer.register


def _tolerant_register(config_class, *args, **kwargs):
    try:
        return _orig_register(config_class, *args, **kwargs)
    except AttributeError:
        if isinstance(config_class, str):
            return None
        raise


AutoTokenizer.register = staticmethod(_tolerant_register)
