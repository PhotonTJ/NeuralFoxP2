#!/usr/bin/env python
"""Print the lm-evaluation-harness tasks to run for one language, filtered to tasks your
installed lm-eval actually has.

    python scripts/lmeval_tasks.py hi target ll     -> belebele_hin_Deva,xnli_hi
    python scripts/lmeval_tasks.py hi english ll    -> belebele_eng_Latn,xnli_en
    python scripts/lmeval_tasks.py bn target gen    -> mgsm_direct_bn

ll  = log-likelihood (multiple-choice) tasks -> run with FOXP2_MODE=all
gen = generative tasks                       -> run with the default decode_window mode
Belebele covers all 11 target languages, so it is the one benchmark comparable across them.
"""
import sys

BELEBELE = {"hi": "hin_Deva", "zh": "zho_Hans", "bn": "ben_Beng", "te": "tel_Telu", "es": "spa_Latn",
            "sw": "swh_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "mr": "mar_Deva", "de": "deu_Latn",
            "fr": "fra_Latn", "en": "eng_Latn"}
XNLI = {"hi", "zh", "es", "sw", "ar", "de", "fr", "en"}
MGSM = {"bn", "de", "es", "fr", "sw", "te", "zh", "en"}


def wanted(lang, kind):
    l = "en" if kind == "english" else lang
    ll = [f"belebele_{BELEBELE[l]}"] + ([f"xnli_{l}"] if l in XNLI else [])
    gen = [f"mgsm_direct_{l}"] if l in MGSM else []
    return ll, gen


def main():
    lang, kind, typ = sys.argv[1], sys.argv[2], sys.argv[3]
    ll, gen = wanted(lang, kind)
    tasks = ll if typ == "ll" else gen
    try:
        from lm_eval.tasks import TaskManager
        have = set(TaskManager().all_tasks)
        missing = [t for t in tasks if t not in have]
        if missing:
            print(f"[lmeval_tasks] not in this lm-eval install, skipped: {missing}", file=sys.stderr)
        tasks = [t for t in tasks if t in have]
    except Exception as e:  # pragma: no cover
        print(f"[lmeval_tasks] could not list tasks ({e}); using names unchecked", file=sys.stderr)
    print(",".join(tasks))


if __name__ == "__main__":
    main()
