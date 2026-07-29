"""Gradio demo for Vietnamese diacritic restoration.

Type toneless Vietnamese in, get it back with diacritics restored. Pick
between the n-gram+dictionary baseline and the neural (Transformer)
model. See CLAUDE.md for project context and README.md for eval numbers
and known limitations (both models share the same dictionary-coverage
ceiling, and the neural model currently generalizes slightly *worse*
than the baseline on cross-domain text -- see examples/hard_examples.md).

    python app/app.py
"""

import sys
from pathlib import Path

import gradio as gr

_HERE = Path(__file__).resolve().parent
# Dev repo layout: app.py lives in app/, so data/ and src/ are one level up.
# Space deployment layout (see app/deploy_space.py): app.py, data/, and
# src/ all sit flat at the Space repo root, alongside each other.
REPO_ROOT = _HERE if (_HERE / "data").exists() else _HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from baseline import BaselineRestorer  # noqa: E402

BASELINE = "Baseline (n-gram + Viterbi)"
NEURAL = "Neural (Transformer)"

baseline_restorer = BaselineRestorer(
    REPO_ROOT / "data" / "dictionary.json",
    REPO_ROOT / "data" / "ngram_counts.json",
)

neural_checkpoint = REPO_ROOT / "data" / "neural" / "checkpoint.pt"
neural_restorer = None
if neural_checkpoint.exists():
    # Deferred import: the app should still run with just the baseline
    # (and without torch installed) if the neural checkpoint isn't there.
    from model import NeuralRestorer

    neural_restorer = NeuralRestorer(REPO_ROOT / "data" / "dictionary.json", neural_checkpoint)

MODEL_CHOICES = [BASELINE, NEURAL] if neural_restorer else [BASELINE]

EXAMPLE_SENTENCES = [
    "toi di hoc o Ha Noi hom nay troi dep qua",
    "Chao ban, ban khoe khong?",
    "Viet Nam la mot dat nuoc xinh dep, toi rat yeu que huong minh",
    "hello the weather is 25 do C hom nay",
]
EXAMPLES = [[text, BASELINE] for text in EXAMPLE_SENTENCES]


def restore_text(text: str, model_choice: str) -> str:
    if model_choice == NEURAL and neural_restorer is not None:
        return neural_restorer.restore(text)
    return baseline_restorer.restore(text)


demo = gr.Interface(
    fn=restore_text,
    inputs=[
        gr.Textbox(
            label="Toneless Vietnamese",
            placeholder="toi di hoc o Ha Noi hom nay troi dep qua",
            lines=4,
        ),
        gr.Radio(MODEL_CHOICES, value=BASELINE, label="Model"),
    ],
    outputs=gr.Textbox(label="Restored"),
    examples=EXAMPLES,
    title="Vietnamese Diacritic Restoration",
    description=(
        "Type Vietnamese without diacritics (dấu) and get them restored. "
        "See the README for eval numbers and known failure cases (e.g. "
        "casual words losing to formal homographs, and code-switched "
        "English being out of scope for either model)."
    ),
)

if __name__ == "__main__":
    demo.launch()
