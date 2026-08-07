"""Phase 1 driver (no LLM calls).

1. Print, in full, the contradictory duplicate groups BEFORE dropping them, and
   save them to contradictions.txt (deck material).
2. Print, in full, the cross-speaker near-duplicate groups, flagging any that
   look like false matches from over-aggressive normalization.
3. Run the cleaning stage and report dropped rows.
4. Build both splits and report realized test fraction + class balance.
5. Assert split integrity (no speaker crosses; no near-dup group crosses).
"""

from __future__ import annotations

from truthclf import data

DATA_PATH = "data.csv"
CONTRA_PATH = "contradictions.txt"
SCHEME = "primary"
TEST_FRAC = 0.2
SEED = 0


def hr(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def fmt_contradiction(groups):
    lines = []
    for gi, (key, members) in enumerate(groups, 1):
        lines.append(f"\nCONTRADICTION GROUP {gi}  (normalized key: {key!r})")
        for m in members:
            lines.append(
                f"  row_id={m.row_id}  label={m.label}  "
                f"binary={m.y(SCHEME)}  speaker={m.speaker_name!r}  "
                f"affiliation={m.speaker_affiliation!r}"
            )
            lines.append(f"    context : {m.statement_context!r}")
            lines.append(f"    text    : {m.statement!r}")
    return "\n".join(lines)


def main():
    rows = data.load(DATA_PATH)
    print(f"Loaded {len(rows)} rows from {DATA_PATH}")

    # 1. Contradictions in full (before dropping) + artifact -------------
    hr("CONTRADICTORY DUPLICATE GROUPS (printed BEFORE removal)")
    contra = data.contradiction_groups(rows, SCHEME)
    report_text = fmt_contradiction(contra)
    print(report_text)
    with open(CONTRA_PATH, "w", encoding="utf-8") as f:
        f.write(f"Contradictory duplicate groups (scheme={SCHEME})\n")
        f.write(f"{len(contra)} groups; these are dropped from the dataset.\n")
        f.write(report_text + "\n")
    print(f"\n  -> saved to {CONTRA_PATH}  ({len(contra)} groups)")

    # 2. Cross-speaker near-duplicate groups in full --------------------
    hr("CROSS-SPEAKER NEAR-DUPLICATE GROUPS (leak risk; verify genuineness)")
    cross = data.cross_speaker_repeat_groups(rows)
    n_flagged = 0
    for gi, (key, members, suspicious) in enumerate(cross, 1):
        flag = "  [FLAG: possible false match]" if suspicious else ""
        if suspicious:
            n_flagged += 1
        print(f"\nNEAR-DUP GROUP {gi}  (key: {key!r}){flag}")
        for m in members:
            print(f"  row_id={m.row_id}  speaker={m.speaker_name!r}  "
                  f"label={m.label}  binary={m.y(SCHEME)}")
            print(f"    text: {m.statement!r}")
    print(f"\n  {len(cross)} cross-speaker near-dup groups; "
          f"{n_flagged} flagged for manual review.")

    # 3. Cleaning stage --------------------------------------------------
    hr("CLEANING STAGE (drop contradictory groups)")
    clean, rep = data.clean_dataset(rows, SCHEME)
    print(rep.summary())

    # 4. Both splits: realized size + class balance ---------------------
    hr("SPLITS — realized test fraction & class balance")
    for name, splitter in [
        ("speaker-disjoint (PRIMARY)", data.speaker_disjoint_split),
        ("stratified-random", data.stratified_random_split),
    ]:
        train, test = splitter(clean, test_frac=TEST_FRAC, seed=SEED, scheme=SCHEME)
        n = len(train) + len(test)
        ntr_t, ntr_f, tr_frac = data.class_balance(train, SCHEME)
        nte_t, nte_f, te_frac = data.class_balance(test, SCHEME)
        print(f"\n  {name}")
        print(f"    train={len(train)}  test={len(test)}  "
              f"realized test fraction={len(test)/n:.3f}")
        print(f"    train True/False = {ntr_t}/{ntr_f}  (True frac {tr_frac:.3f})")
        print(f"    test  True/False = {nte_t}/{nte_f}  (True frac {te_frac:.3f})")

    # 5. Integrity assertions -------------------------------------------
    hr("SPLIT-INTEGRITY ASSERTIONS")
    tr_sd, te_sd = data.speaker_disjoint_split(clean, test_frac=TEST_FRAC, seed=SEED, scheme=SCHEME)
    tr_sr, te_sr = data.stratified_random_split(clean, test_frac=TEST_FRAC, seed=SEED, scheme=SCHEME)

    checks = [
        ("speaker-disjoint: no speaker crosses", data.speakers_cross(tr_sd, te_sd)),
        ("speaker-disjoint: no near-dup group crosses", data.normkeys_cross(tr_sd, te_sd)),
        ("stratified: no near-dup group crosses", data.normkeys_cross(tr_sr, te_sr)),
    ]
    all_ok = True
    for label, violations in checks:
        ok = len(violations) == 0
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
              + ("" if ok else f"  ({len(violations)} violations: {list(violations)[:5]})"))
    # sanity: stratified intentionally allows speakers to cross
    n_spk_cross_sr = len(data.speakers_cross(tr_sr, te_sr))
    print(f"  [info] stratified split speakers crossing (expected >0): {n_spk_cross_sr}")
    print(f"\n  ALL INTEGRITY CHECKS {'PASSED' if all_ok else 'FAILED'}")


if __name__ == "__main__":
    main()
