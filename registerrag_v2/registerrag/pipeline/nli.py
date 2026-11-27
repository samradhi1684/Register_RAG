"""
NLI binding verifier (v3 candidate, 2026-11-25).

The ms-marco cross-encoder scores *relevance*: a passage about the right
entity scores high whether or not it states the extracted value. An NLI model
scores *entailment*: does the passage state "the answer to <step> is <value>"?

  premise    = the passage the value was extracted from
  hypothesis = 'The answer to "<step, placeholders filled>" is <value>.'

Used as a soft signal in RegisterManager.add (see config.nli_*):
  contradiction -> the value is rejected for that passage; the next passage is tried
  entailment    -> accepted
  neutral       -> accepted unless a passage within the look-ahead entails a value
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

_MODEL = {}


def _model(name: str):
    if name not in _MODEL:
        from sentence_transformers import CrossEncoder
        m = CrossEncoder(name)
        labels = {v.lower(): int(k) for k, v in m.config.id2label.items()}
        _MODEL[name] = (m, labels)
    return _MODEL[name]


def hypothesis(need: str, value: str) -> str:
    need = " ".join(need.split())
    return f'The answer to "{need}" is {value}.'


def judge(premise: str, need: str, value: str, model_name: str) -> tuple[str, float, float]:
    """-> (label, p_entailment, p_contradiction)."""
    m, lab = _model(model_name)
    p = m.predict([(premise[:1500], hypothesis(need, value))], apply_softmax=True)[0]
    pe, pc, pn = float(p[lab["entailment"]]), float(p[lab["contradiction"]]), float(p[lab["neutral"]])
    label = max((("entailment", pe), ("contradiction", pc), ("neutral", pn)), key=lambda x: x[1])[0]
    return label, pe, pc
