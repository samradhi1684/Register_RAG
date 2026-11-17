"""B7b - LycheeMemory (Chen et al., ACL 2026, arXiv:2602.08382): authors' released 7B checkpoint, inference only.
Reference row: it does NOT use the frozen qwen2.5 stack (the paper's model is its own 7B). See README.md."""
SYSTEM = "lycheeacl"
LABEL = "B7b LycheeMemory (ACL 2026, released 7B checkpoint)"
DEFAULT_UNIT = "chunk"


def pipeline_class():
    from .lychee_acl import LycheeACL
    return LycheeACL


def variants(generator, coder=None, openai=False):
    # the --generator flag is irrelevant for this row (the checkpoint is its own generator); run it once per dataset
    return [("B7b LycheeMemory ACL (own 7B, recurrent memory)", [])]
