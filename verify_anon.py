"""본문과 동일한 검증 세트를 실명(JSON v3)·익명(JSON anon) 두 조건에 같은 방식으로 계산.
- McNemar: 연속성 보정 χ² (본문 p값 .005/.030/.029를 재현하는 방식), Bonferroni α=0.05/3
- 기법 간 쌍 비교(조건별), 실명↔익명 대응 비교(기법별)
- 혼동행렬 기반 팀A/팀B 재현율 (위치 편향 점검, 본문 RQ2와 동일 정의)
사용법: python verify_anon.py experiment_results_json_v3.csv experiment_results_json_anon.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import chi2

ALPHA = 0.05 / 3
TECH = ["FewShot", "CoT", "Role"]


def load(p):
    return pd.read_csv(p, dtype=str, encoding="utf-8-sig").set_index("Scenario ID")


def mcnemar_cc(a, b):
    """a, b: 0/1 정답 배열. 반환 (a만 정답, b만 정답, chi2, p)"""
    x = int(((a == 1) & (b == 0)).sum())
    y = int(((a == 0) & (b == 1)).sum())
    if x + y == 0:
        return x, y, 0.0, 1.0
    s = (abs(x - y) - 1) ** 2 / (x + y)
    return x, y, s, float(1 - chi2.cdf(s, 1))


def correct(df, t):
    return (df[f"Correct_{t}"] == "1").astype(int).values


def recall_ab(df, t, label_a):
    """팀A 승 시나리오에서의 재현율, 팀B 승 시나리오에서의 재현율"""
    win_a = (df["Winner"] == label_a(df)).values
    pred_a = (df[f"Result_{t}"] == label_a(df)).values
    ra = (pred_a & win_a).sum() / win_a.sum()
    rb = (~pred_a & ~win_a & (df[f"Correct_{t}"] == "1").values).sum() / (~win_a).sum()
    return ra, rb, int(win_a.sum()), int((~win_a).sum())


named, anon = load(sys.argv[1]), load(sys.argv[2])
anon = anon.loc[named.index]
n = len(named)
conds = {"실명": (named, lambda d: d["Team A"]), "익명": (anon, lambda d: d["Team A"])}

print(f"n={n}, Bonferroni α={ALPHA:.4f} (연속성 보정 McNemar)\n")

print("[1] 정확도 / 파싱 상태")
for cname, (df, _) in conds.items():
    for t in TECH:
        acc = correct(df, t).sum()
        err = (df[f"Correct_{t}"] == "-1").sum()
        unc = (df[f"Result_{t}"] == "UNCLEAR").sum()
        print(f"  {cname} {t:<8}{acc:>4}/{n} ({acc/n*100:.1f}%)  ERROR={err}  UNCLEAR={unc}")

print("\n[2] 기법 간 쌍 비교 (조건별)")
for cname, (df, _) in conds.items():
    for x, y in [("CoT", "FewShot"), ("Role", "FewShot"), ("CoT", "Role")]:
        a, b, s, p = mcnemar_cc(correct(df, x), correct(df, y))
        print(f"  {cname} {x} vs {y}: {x}만정답={a}, {y}만정답={b}, χ²={s:.3f}, "
              f"p={p:.4f} → {'유의' if p < ALPHA else '유의하지 않음'}")

print("\n[3] 실명 ↔ 익명 (기법별 대응 비교)")
for t in TECH:
    a, b = correct(named, t), correct(anon, t)
    x, y, s, p = mcnemar_cc(a, b)
    same = (a == b).mean() * 100
    print(f"  {t:<8}{a.mean()*100:.1f}% → {b.mean()*100:.1f}% ({(b.mean()-a.mean())*100:+.1f}%p)  "
          f"익명만오답={x}, 익명만정답={y}, χ²={s:.3f}, p={p:.4f} → "
          f"{'유의' if p < ALPHA else '유의하지 않음'}  | 정오 일치율 {same:.1f}%")
a_all = np.concatenate([correct(named, t) for t in TECH])
b_all = np.concatenate([correct(anon, t) for t in TECH])
print(f"  합산: 실명 {a_all.sum()} / 익명 {b_all.sum()} (of {len(a_all)}), "
      f"익명만오답={int(((a_all==1)&(b_all==0)).sum())}, 익명만정답={int(((a_all==0)&(b_all==1)).sum())}")

print("\n[4] 위치 편향 점검: 팀A 재현율 vs 팀B 재현율")
for cname, (df, la) in conds.items():
    for t in TECH:
        ra, rb, na, nb = recall_ab(df, t, la)
        print(f"  {cname} {t:<8}팀A 재현율 {ra:.3f} (n={na}) | 팀B 재현율 {rb:.3f} (n={nb})")