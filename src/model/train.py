"""Train the neural diacritic restoration model on data/corpus.txt.

Only ever reads data/corpus.txt -- data/test_corpus.txt is reserved for
src/eval.py's final held-out number and must never be used for training.

If you hit an unsupported-op error on Apple Silicon, re-run with
PYTORCH_ENABLE_MPS_FALLBACK=1 (falls back to CPU for that one op instead
of crashing); the core Transformer/embedding ops used here have already
been validated on MPS directly.

Usage:
    python src/model/train.py --corpus data/corpus.txt \
        --dictionary data/dictionary.json --output data/neural/checkpoint.pt

    # quick trial run before committing to the full corpus/epoch count:
    python src/model/train.py --max-sentences 2000 --epochs 2 \
        --output data/neural/trial_checkpoint.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from architecture import NeuralDiacriticModel  # noqa: E402
from dataset import (  # noqa: E402
    IGNORE_INDEX,
    DiacriticDataset,
    build_examples,
    load_sentences,
    make_collate_fn,
    train_val_split,
)
from vocab import CharVocab, ClassVocab  # noqa: E402


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    n_batches = 0

    with torch.enable_grad() if training else torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            scores = model(
                batch["char_ids"], batch["padding_mask"], batch["candidate_ids"], batch["candidate_valid"]
            )
            loss = F.cross_entropy(scores.permute(0, 2, 1), batch["targets"], ignore_index=IGNORE_INDEX)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            mask = batch["targets"] != IGNORE_INDEX
            preds = scores.argmax(dim=-1)
            total_correct += ((preds == batch["targets"]) & mask).sum().item()
            total_count += mask.sum().item()
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1), total_correct / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus.txt"))
    parser.add_argument("--dictionary", type=Path, default=Path("data/dictionary.json"))
    parser.add_argument("--output", type=Path, default=Path("data/neural/checkpoint.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-sentences", type=int, default=None, help="use only the first N sentences (for a quick trial run)"
    )
    args = parser.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}")

    dictionary = json.loads(args.dictionary.read_text(encoding="utf-8"))
    class_vocab = ClassVocab.from_dictionary(dictionary)
    char_vocab = CharVocab()
    collate_fn = make_collate_fn(char_vocab)

    sentences = load_sentences(args.corpus)
    if args.max_sentences:
        sentences = sentences[: args.max_sentences]
    train_sents, val_sents = train_val_split(sentences, seed=args.seed)
    print(f"train sentences: {len(train_sents)}, val sentences: {len(val_sents)}")

    train_loader = DataLoader(
        DiacriticDataset(build_examples(train_sents, dictionary, class_vocab)),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        DiacriticDataset(build_examples(val_sents, dictionary, class_vocab)),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = NeuralDiacriticModel(char_vocab_size=len(char_vocab), num_classes=len(class_vocab)).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_steps, max(total_steps // 2, 1))

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / warmup_steps
        remaining = max(total_steps - step, 0)
        return remaining / max(total_steps - warmup_steps, 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, scheduler)
        val_loss, val_acc = run_epoch(model, val_loader, device)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4%} "
            f"val_loss={val_loss:.4f} val_syllable_acc={val_acc:.4%} ({elapsed:.0f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model.config,
                    "class_vocab": class_vocab.to_list(),
                },
                args.output,
            )
            print(f"  -> saved best checkpoint to {args.output} (val_syllable_acc={val_acc:.4%})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping after {epoch} epochs (patience={args.patience})")
                break

    print(f"done. best val_syllable_acc={best_val_acc:.4%}")


if __name__ == "__main__":
    main()
