"""B9 - LiveLedger (paper: "When Is Enough Not Enough? Illusory Completion in Search Agents", arXiv:2602.07549). See README.md."""
SYSTEM = "liveledger"
LABEL = "B9 LiveLedger"
DEFAULT_UNIT = "sentence"            # small observations keep a multi-step agent inside the context window


def pipeline_class():
    from .liveledger import LiveLedgerAgent
    return LiveLedgerAgent


def variants(generator, coder=None, openai=False):
    return [("B9 search agent + LiveLedger", []),
            ("B9 search agent, no ledger (the paper's 'w/o LiveLedger' control)", ["--method-arg", "ledger=false"])]
