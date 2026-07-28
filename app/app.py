"""Gradio demo for the Vietnamese diacritic-restoration baseline.

Type toneless Vietnamese in, get it back with diacritics restored.
See CLAUDE.md for project context and README.md for eval numbers and
known limitations.

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

restorer = BaselineRestorer(
    REPO_ROOT / "data" / "dictionary.json",
    REPO_ROOT / "data" / "ngram_counts.json",
)

EXAMPLES = [
    "toi di hoc o Ha Noi hom nay troi dep qua",
    "Chao ban, ban khoe khong?",
    "Viet Nam la mot dat nuoc xinh dep, toi rat yeu que huong minh",
    "hello the weather is 25 do C hom nay",
]

demo = gr.Interface(
    fn=restorer.restore,
    inputs=gr.Textbox(
        label="Toneless Vietnamese",
        placeholder="toi di hoc o Ha Noi hom nay troi dep qua",
        lines=4,
    ),
    outputs=gr.Textbox(label="Restored"),
    examples=EXAMPLES,
    title="Vietnamese Diacritic Restoration",
    description=(
        "Type Vietnamese without diacritics (dấu) and get them restored. "
        "n-gram + dictionary baseline — see the README for eval numbers "
        "and known failure cases (e.g. casual words losing to formal "
        "homographs, and code-switched English being out of scope)."
    ),
)

if __name__ == "__main__":
    demo.launch()
