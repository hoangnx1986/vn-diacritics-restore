# Hard / ambiguous examples

A small, hand-picked set of real failures from the n-gram baseline,
mined from the held-out news test set (`data/test_corpus.txt`, a
different domain than the Wikipedia training corpus — see
`src/eval.py`). Kept for qualitative spot-checks and for the README's
honest failure-case section, per CLAUDE.md.

These are not bugs — the code is doing exactly what it's designed to do
(pick the sequence the trigram LM scores highest). They're corpus
limitations: a ~98K-sentence, Wikipedia-only training set doesn't have
enough signal to resolve every ambiguity correctly. A bigger/more
diverse corpus (CLAUDE.md's own next step) should fix most of these.

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

| Input | Expected | Baseline output |
|---|---|---|
| `Chao ban, ban khoe khong?` | `Chào bạn, bạn khỏe không?` | `Chào bán, bản khỏe không?` (bán = sell, bản = version) |
| `...troi dep qua` | `...trời đẹp quá` (quá = intensifier "so") | `...trời đẹp qua` (qua = pass by) |

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
