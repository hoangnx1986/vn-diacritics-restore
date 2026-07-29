# Hard / ambiguous examples

A small, hand-picked set of real failures from both the n-gram baseline
and the neural model, mined from the held-out news test set
(`data/test_corpus.txt`, a different domain than the Wikipedia+colloquial
training corpus — see `src/eval.py`). Kept for qualitative spot-checks
and for the README's honest failure-case section, per CLAUDE.md.

These are not bugs — the code is doing exactly what it's designed to do
(pick the sequence the trigram LM scores highest). They're corpus
limitations: training data that doesn't have enough signal in the right
register to resolve every ambiguity correctly.

**Update after broadening the corpus:** we added ~11.4K colloquial
sentences from a student-feedback dataset (UIT-VSFC) alongside the
original ~98K Wikipedia sentences and retrained (see `data/corpus.txt`,
`src/baseline/train.py`). Held-out syllable accuracy barely moved
(89.0% → 89.2%) but one specific miss below (`quá`) got fixed, while the
`bé`/`trẻ`/`bạn` cases did not. That's itself an informative result: a
colloquial source only helps the ambiguities it actually contains
examples of. Student course-feedback text uses casual intensifiers like
"quá" often, but has little reason to talk about children ("bé", "trẻ")
or address someone as "bạn" — so those specific gaps need a source in
*that* register (parenting/education content, casual dialogue) to close,
not just "more colloquial text" in general.

## Systematic: `be` → child (bé) vs. faction/pedestal/side (bè/bệ/bề)

The baseline resolves `be` to `bè`, `bệ`, or `bề` far more often than
the correct `bé` ("child"), even in an article that is *about children*:

| Input | Expected | Baseline output |
|---|---|---|
| `Trong khi do, nhieu be du cham chi bao nhieu cung chang cai thien duoc diem so.` | `...nhiều **bé** dù chăm chỉ...` | `...nhiều **bè** dù chăm chỉ...` |
| `Cau tra loi la nhung be thuoc nhom "thieu so" nam duoc bi quyet tap trung.` | `...những **bé** thuộc nhóm...` | `...những **bệ** thuộc nhóm...` |
| `Ban cung nen chi dan tre de tap, sach, tai lieu vao mot cho gon gang.` | `**Bạn** cũng nên...` | `**Bản** cũng nên...` |

This looks like a corpus-frequency artifact: the training corpus (formal
Wikipedia prose) rarely uses `bé` in the "child" sense, so the LM's
priors for that syllable are dominated by other, rarer-but-present
spellings once context is thin.

## Systematic: `tre` → child (trẻ) vs. late (trễ) vs. bamboo (tre)

| Input | Expected | Baseline output |
|---|---|---|
| `Don gian la do tre co the bi ap luc tinh than va thoi gian.` | `...do **trẻ** có thể bị áp lực...` | `...do **trễ** có thể bị áp lực...` |
| `Dieu nay se giup be hoc tot hon, dong thoi cho tre thoi gian de chinh sua bai tap ve nha.` | `...cho **trẻ** thời gian...` | `...cho **trễ** thời gian...` |

## Register mismatch: casual `bạn`/`quá` lose to formal homographs

Found during manual testing (not the news set): Wikipedia's formal
register underrepresents casual/spoken words relative to their formal
homographs.

| Input | Expected | Baseline output (Wikipedia-only) | Baseline output (+ colloquial corpus) |
|---|---|---|---|
| `Chao ban, ban khoe khong?` | `Chào bạn, bạn khỏe không?` | `Chào bán, bản khỏe không?` | `Chào bán, bản khỏe không?` — **still wrong** |
| `...troi dep qua` | `...trời đẹp quá` (intensifier "so") | `...trời đẹp qua` (qua = pass by) | `...trời đẹp quá` — **fixed** |

`quá` improved because the added student-feedback text actually uses
casual intensifiers; `bạn` didn't, because that dataset has little
reason to address someone as "bạn" either — see the "Update after
broadening the corpus" note above for what this implies about corpus
selection.

## Neural model: much higher in-domain accuracy, but worse cross-domain generalization

Phase 2 (`src/model/`) trains a ~6.8M-parameter Transformer that
classifies each syllable using full-sentence bidirectional attention
instead of n-gram+Viterbi. On its own validation split (held out from
the *same* training corpus), it reaches a striking **94.4%** syllable
accuracy — far above anything the n-gram baseline achieves in-domain.
But on `data/test_corpus.txt` (the news-domain held-out set — a genuinely
different domain than training, same set used for the baseline's
headline number), the comparison inverts:

| model | syllable acc | char acc | exact match |
|---|---|---|---|
| Naive (most-frequent candidate, no context) | 52.5% | 89.2% | 0.2% |
| N-gram + Viterbi baseline | **89.2%** | **97.5%** | **16.4%** |
| Neural (Transformer) | 86.3% | 96.8% | 10.7% |

The neural model is *worse* than the simpler n-gram baseline at
generalizing to unseen domain/register, despite learning its own
training distribution far better. This is a textbook overfitting/
generalization story, not a bug: a 6.8M-parameter model has much more
capacity to pick up training-domain-specific associations than a
counting-based n-gram model with backoff smoothing, and some of what it
learned from Wikipedia+colloquial text apparently doesn't transfer as
cleanly to news-register text.

Side-by-side on the exact hard examples already documented above:

| Input (abbreviated) | Baseline | Neural |
|---|---|---|
| "...nhiều bé dù chăm chỉ..." | `bè` (wrong) | `be`, plus **new** errors elsewhere in the same sentence (`đủ`, `chấm`, `cùng`, `chăng` all wrong) |
| "...những bé thuộc nhóm..." | `bệ` (wrong) | `bé` — **correct**, but `nắm`→`năm` and `bí`→`bị` newly wrong |
| "Ban cũng nên chỉ dẫn trẻ..." | `Bản` (wrong) | `Ban` (also wrong, a different miss) |
| "...do trẻ có thể bị áp lực..." | `trễ` (wrong) | `trễ` (same wrong answer) |
| "Chào bạn, bạn khỏe không?" | `bán, bản` (both wrong) | `bán, bán` (both wrong, different miss) |
| canonical CLAUDE.md sentence | ✓ correct | ✓ correct |

Not uniformly worse — it does fix cases the baseline misses (the "bé"
example) — but on this specific cross-domain set it introduces more
scattered new errors than it resolves, netting out slightly behind. Both
models still share the exact same dictionary-coverage ceiling (same
`data/dictionary.json`), so this gap is purely about the scoring
mechanism's generalization, not candidate legality.

## Non-goal, by design: code-switched English

CLAUDE.md explicitly scopes out Vietnamese–English code-switching. This
is what that limitation looks like in practice — any English word that
happens to collide with a valid Vietnamese toneless syllable gets
"corrected":

| Input | Baseline output |
|---|---|
| `hello the weather is 25 do C hom nay` | `hello **thể** weather is 25 độ C hôm nay` |

`the` is a legal Vietnamese syllable (thế/thể/...), and there's no
language-ID step, so it's treated as Vietnamese and restored — exactly
the tradeoff CLAUDE.md accepts by declaring code-switching out of scope.
