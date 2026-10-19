"""완전 익명 결과 CSV 생성: CoT·Role은 기존 익명 결과, Few-shot은 예시까지 익명화한 재실험 결과.
사용법: python merge_full_anon.py experiment_results_json_anon.csv experiment_results_json_anon_fs.csv experiment_results_json_anon_full.csv
"""
import sys
import pandas as pd

base = pd.read_csv(sys.argv[1], dtype=str, encoding="utf-8-sig").set_index("Scenario ID")
fs = pd.read_csv(sys.argv[2], dtype=str, encoding="utf-8-sig").set_index("Scenario ID").loc[base.index]
assert fs["Correct_FewShot"].isin(["0", "1"]).all(), "Few-shot 재실험에 ERROR/빈칸이 남아 있음"
for c in ["Result_FewShot", "Raw_FewShot", "Correct_FewShot", "Prompt_FewShot"]:
    base[c] = fs[c]
base.reset_index().to_csv(sys.argv[3], index=False, encoding="utf-8-sig")
print("저장:", sys.argv[3])