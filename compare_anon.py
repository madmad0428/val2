"""실명(JSON v3) vs 익명(JSON anon) 비교: 기법별 정확도 + 대응 McNemar (Bonferroni α=0.05/3)
사용법: python compare_anon.py experiment_results_json_v3.csv experiment_results_json_anon.csv
"""
import sys
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

named = pd.read_csv(sys.argv[1], dtype=str).set_index("Scenario ID")
anon = pd.read_csv(sys.argv[2], dtype=str, encoding="utf-8-sig").set_index("Scenario ID")
anon = anon.loc[named.index]
ALPHA = 0.05 / 3
N = len(named)

print(f"{'기법':<8}{'실명':>10}{'익명':>10}{'차이(%p)':>10}  익명만오답 익명만정답  p(exact)  판정")
for tech in ["FewShot", "CoT", "Role"]:
    c = f"Correct_{tech}"
    err = (anon[c] == "-1").sum()
    a = (named[c] == "1").astype(int)
    b = (anon[c] == "1").astype(int)
    only_named = int(((a == 1) & (b == 0)).sum())   # 익명화로 틀리게 된 것
    only_anon = int(((a == 0) & (b == 1)).sum())    # 익명화로 맞게 된 것
    table = [[int(((a == 1) & (b == 1)).sum()), only_named],
             [only_anon, int(((a == 0) & (b == 0)).sum())]]
    p = mcnemar(table, exact=True).pvalue
    print(f"{tech:<8}{a.mean()*100:>9.1f}%{b.mean()*100:>9.1f}%{(b.mean()-a.mean())*100:>+10.1f}"
          f"  {only_named:>9} {only_anon:>9}  {p:>8.4f}  {'유의' if p < ALPHA else '차이 없음'}"
          + (f"  (익명 ERROR {err}건 포함됨)" if err else ""))
print(f"\nn={N}, Bonferroni α={ALPHA:.4f}")