"""B<N> - <name> (<authors, year>, arXiv:<id>). See README.md."""
SYSTEM = "mymethod"                  # value of --system (lowercase)
LABEL = "B<N> MyMethod"
DEFAULT_UNIT = "sentence"            # or "chunk"


def pipeline_class():
    from .mymethod import MyMethod   # imported lazily so this baseline's dependencies never affect the others
    return MyMethod


def variants(generator, coder=None, openai=False):
    """Runs launched by scripts/run_baselines.py: (label, extra CLI args). Use --method-arg key=value for settings."""
    return [("B<N> MyMethod", [])]
