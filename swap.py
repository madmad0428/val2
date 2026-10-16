import pandas as pd

df = pd.read_csv("dataset_440_json_v3.csv")

def swap_scenario(scenario):
    idx_a = scenario.find("팀A:")
    idx_b = scenario.find("팀B:")
    header = scenario[:idx_a]
    block_a = scenario[idx_a:idx_b]
    block_b = scenario[idx_b:]
    new_a = block_b.replace("팀B:", "팀A:", 1)
    new_b = block_a.replace("팀A:", "팀B:", 1)
    if not new_a.endswith("\n"):
        new_a += "\n"
    new_b = new_b.rstrip("\n")
    return header + new_a + new_b

def swap_prompt(prompt, scenario, team_a, team_b):
    prompt, scenario = str(prompt), str(scenario)
    if scenario not in prompt:
        return None
    new_prompt = prompt.replace(scenario, swap_scenario(scenario))
    orig_instr = f'{{"winner": "{team_a}" 또는 "{team_b}"}}'
    swap_instr = f'{{"winner": "{team_b}" 또는 "{team_a}"}}'
    if orig_instr not in new_prompt:
        return None
    return new_prompt.replace(orig_instr, swap_instr)

# 1. 전수 치환 가능 확인
fail = 0
for idx, row in df.iterrows():
    for col in ["Prompt_FewShot", "Prompt_CoT", "Prompt_Role"]:
        if swap_prompt(row[col], row["Scenario Text"], row["Team A"], row["Team B"]) is None:
            fail += 1
print(f"치환 실패: {fail}건 (0이어야 함)")

# 2. 층화 표본: 접전(≤5) / 일방(≥6) 각 75개 — 논문 3.1절 기준
df["type"] = df["Score Diff"].apply(lambda x: "close" if abs(x) <= 5 else "dominant")
sample = pd.concat([
    df[df["type"] == "close"].sample(n=75, random_state=42),
    df[df["type"] == "dominant"].sample(n=75, random_state=42),
])
print(f"표본: {len(sample)}개 (접전 {(sample['type']=='close').sum()} / 일방 {(sample['type']=='dominant').sum()})")

# 3. swap 데이터셋 생성
rows = []
for idx, row in sample.iterrows():
    new_row = row.copy()
    new_row["Team A"], new_row["Team B"] = row["Team B"], row["Team A"]  # 채점 기준용 뒤집기
    new_row["Scenario Text"] = swap_scenario(str(row["Scenario Text"]))
    for col in ["Prompt_FewShot", "Prompt_CoT", "Prompt_Role"]:
        new_row[col] = swap_prompt(row[col], row["Scenario Text"], row["Team A"], row["Team B"])
    # 결과 컬럼 초기화
    for col in ["Result_FewShot", "Result_CoT", "Result_Role",
                "Correct_FewShot", "Correct_CoT", "Correct_Role",
                "Raw_FewShot", "Raw_CoT", "Raw_Role"]:
        new_row[col] = ""
    rows.append(new_row)

swap_df = pd.DataFrame(rows)
swap_df.to_csv("dataset_swap_150.csv", index=False)

# 원본 재실행용(기저 변동 측정) 표본도 저장 — 같은 150개, 순서 원본 그대로
rerun_df = sample.copy()
for col in ["Result_FewShot", "Result_CoT", "Result_Role",
            "Correct_FewShot", "Correct_CoT", "Correct_Role",
            "Raw_FewShot", "Raw_CoT", "Raw_Role"]:
    rerun_df[col] = ""
rerun_df.to_csv("dataset_rerun_150.csv", index=False)

print("저장 완료: dataset_swap_150.csv, dataset_rerun_150.csv")

import pandas as pd

orig = pd.read_csv("experiment_results_json_v3.csv")
swap = pd.read_csv("results_swap_150.csv")
rerun = pd.read_csv("results_rerun_150.csv")

print(f"{'기법':<10} {'swap 일관성':<15} {'rerun 일관성':<15} {'차이':<10}")
print("-" * 50)

for tech in ["FewShot", "CoT", "Role"]:
    res_col = f"Result_{tech}"

    merged_s = swap[["Scenario ID", res_col]].merge(
        orig[["Scenario ID", res_col]], on="Scenario ID", suffixes=("_swap", "_orig"))
    merged_r = rerun[["Scenario ID", res_col]].merge(
        orig[["Scenario ID", res_col]], on="Scenario ID", suffixes=("_rerun", "_orig"))

    consist_swap = (merged_s[f"{res_col}_swap"] == merged_s[f"{res_col}_orig"]).mean()
    consist_rerun = (merged_r[f"{res_col}_rerun"] == merged_r[f"{res_col}_orig"]).mean()

    print(f"{tech:<10} {consist_swap:<15.1%} {consist_rerun:<15.1%} {consist_swap-consist_rerun:+.1%}")

# 접전/일방 그룹별 분해
print("\n=== 경기 유형별 swap 일관성 ===")
for tech in ["FewShot", "CoT", "Role"]:
    res_col = f"Result_{tech}"
    merged_s = swap[["Scenario ID", res_col, "Score Diff"]].merge(
        orig[["Scenario ID", res_col]], on="Scenario ID", suffixes=("_swap", "_orig"))
    merged_s["type"] = merged_s["Score Diff"].abs().apply(lambda x: "접전" if x <= 5 else "일방")
    for t in ["접전", "일방"]:
        sub = merged_s[merged_s["type"] == t]
        consist = (sub[f"{res_col}_swap"] == sub[f"{res_col}_orig"]).mean()
        print(f"{tech} - {t}: {consist:.1%} (n={len(sub)})")

import pandas as pd

orig = pd.read_csv("experiment_results_json_v3.csv")
rerun = pd.read_csv("results_rerun_150.csv")

print("=== 경기 유형별 rerun 일관성 (대조군) ===")
for tech in ["FewShot", "CoT", "Role"]:
    res_col = f"Result_{tech}"
    merged = rerun[["Scenario ID", res_col, "Score Diff"]].merge(
        orig[["Scenario ID", res_col]], on="Scenario ID", suffixes=("_rerun", "_orig"))
    merged["type"] = merged["Score Diff"].abs().apply(lambda x: "접전" if x <= 5 else "일방")
    for t in ["접전", "일방"]:
        sub = merged[merged["type"] == t]
        consist = (sub[f"{res_col}_rerun"] == sub[f"{res_col}_orig"]).mean()
        print(f"{tech} - {t}: {consist:.1%} (n={len(sub)})")