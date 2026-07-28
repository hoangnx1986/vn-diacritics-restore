"""Gradio demo for the Vietnamese diacritic-restoration baseline.

Type toneless Vietnamese in, get it back with diacritics restored.
See CLAUDE.md for project context and README.md for eval numbers and
known limitations.

    python app/app.py
"""

import sys
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent.parent
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
