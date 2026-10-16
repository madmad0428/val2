import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

orig = pd.read_csv("experiment_results_json_v3.csv")
swap = pd.read_csv("results_swap_150.csv")
rerun = pd.read_csv("results_rerun_150.csv")

print("=== 접전 경기(n=75) swap vs rerun 불일치율 McNemar 검정 ===\n")

for tech in ["FewShot", "CoT", "Role"]:
    res_col = f"Result_{tech}"

    m_s = swap[["Scenario ID", res_col, "Score Diff"]].merge(
        orig[["Scenario ID", res_col]], on="Scenario ID", suffixes=("_swap", "_orig"))
    m_r = rerun[["Scenario ID", res_col, "Score Diff"]].merge(
        orig[["Scenario ID", res_col]], on="Scenario ID", suffixes=("_rerun", "_orig"))

    m_s["type"] = m_s["Score Diff"].abs().apply(lambda x: "close" if x <= 5 else "dominant")
    m_r["type"] = m_r["Score Diff"].abs().apply(lambda x: "close" if x <= 5 else "dominant")

    m_s_close = m_s[m_s["type"] == "close"].copy()
    m_r_close = m_r[m_r["type"] == "close"].copy()

    m_s_close["swap_wrong"] = (m_s_close[f"{res_col}_swap"] != m_s_close[f"{res_col}_orig"]).astype(int)
    m_r_close["rerun_wrong"] = (m_r_close[f"{res_col}_rerun"] != m_r_close[f"{res_col}_orig"]).astype(int)

    merged = m_s_close[["Scenario ID", "swap_wrong"]].merge(
        m_r_close[["Scenario ID", "rerun_wrong"]], on="Scenario ID")

    both_wrong = ((merged["swap_wrong"]==1) & (merged["rerun_wrong"]==1)).sum()
    swap_only  = ((merged["swap_wrong"]==1) & (merged["rerun_wrong"]==0)).sum()
    rerun_only = ((merged["swap_wrong"]==0) & (merged["rerun_wrong"]==1)).sum()
    both_ok    = ((merged["swap_wrong"]==0) & (merged["rerun_wrong"]==0)).sum()

    table = [[both_wrong, swap_only], [rerun_only, both_ok]]
    result = mcnemar(table, exact=True)

    swap_rate = merged["swap_wrong"].mean()
    rerun_rate = merged["rerun_wrong"].mean()

    print(f"{tech}:")
    print(f"  swap 불일치율={swap_rate:.1%} | rerun 불일치율={rerun_rate:.1%}")
    print(f"  2x2 표: both_wrong={both_wrong}, swap_only={swap_only}, rerun_only={rerun_only}, both_ok={both_ok}")
    print(f"  McNemar p={result.pvalue:.4f} ({'유의함' if result.pvalue<0.05 else '유의하지 않음'})\n")