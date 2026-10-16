import pandas as pd

df2 = pd.read_csv("experiment_results_json_v3.csv")

# 1. Raw 응답이 실제로 JSON 형태인지 확인
print("=== Raw_CoT 샘플 (전체) ===")
print(str(df2.iloc[0]["Raw_CoT"])[:500])
print("\n... (끝부분) ...")
print(str(df2.iloc[0]["Raw_CoT"])[-300:])

# 2. Result 값 분포 확인
print("\n=== Result_CoT 고유값 상위 10 ===")
print(df2["Result_CoT"].value_counts().head(10))

# 3. UNCLEAR/에러 케이스 확인
for col in ["Result_FewShot", "Result_CoT", "Result_Role"]:
    unclear = (~df2.apply(lambda r: r[col] in [r["Team A"], r["Team B"]], axis=1)).sum()
    print(f"{col}: 팀명 외 값 {unclear}건")

# 4. 현재 기록된 정확도 확인
print()
for col in ["Correct_FewShot", "Correct_CoT", "Correct_Role"]:
    acc = (df2[col].astype(str) == "1").sum()
    print(f"{col}: {acc}/440 ({round(acc/440*100,1)}%)")

import re, json

def get_json_winner(text):
    matches = re.findall(r'\{[^{}]*"winner"[^{}]*\}', str(text))
    if not matches:
        return None
    try:
        return json.loads(matches[-1]).get("winner")
    except:
        return None

print("=== JSON 블록 vs Result 컬럼 일치 검증 ===")
for res_col, raw_col in [("Result_FewShot", "Raw_FewShot"),
                          ("Result_CoT", "Raw_CoT"),
                          ("Result_Role", "Raw_Role")]:
    json_winners = df2[raw_col].apply(get_json_winner)
    no_json = json_winners.isna().sum()
    mismatch = ((json_winners != df2[res_col]) & json_winners.notna()).sum()
    print(f"{res_col}: JSON 블록 없음 {no_json}건 | Result와 불일치 {mismatch}건")

import re, json
import pandas as pd

df2 = pd.read_csv("experiment_results_json_v3.csv")

def get_json_winner_v2(text, team_a, team_b):
    """vct_run.py의 실제 로직을 그대로 재현"""
    clean = re.sub(r'```json|```', '', str(text)).strip()
    matches = re.findall(r'\{[^}]+\}', clean)
    results = []
    for m in reversed(matches):
        try:
            parsed = json.loads(m)
            winner = parsed.get('winner', '').strip()
            if str(team_a).lower() in winner.lower():
                results.append(('A', team_a, winner))
            elif str(team_b).lower() in winner.lower():
                results.append(('B', team_b, winner))
        except:
            continue
    return results

print("=== 불일치 케이스 상세 ===")
for res_col, raw_col in [("Result_FewShot", "Raw_FewShot"),
                          ("Result_CoT", "Raw_CoT"),
                          ("Result_Role", "Raw_Role")]:
    for idx, row in df2.iterrows():
        matches_found = re.findall(r'\{[^}]+\}', re.sub(r'```json|```', '', str(row[raw_col])))
        if len(matches_found) > 1:
            # 여러 JSON 블록이 있는 경우만 우선 확인
            print(f"\n[{row['Scenario ID']}] {res_col} - JSON 블록 {len(matches_found)}개 발견")
            print(f"블록들: {matches_found}")
            print(f"저장된 Result: {row[res_col]} / 실제 승자: {row['Winner']}")

def extract_winner_json_v2(text, team_a, team_b):
    clean = re.sub(r'```json|```', '', str(text)).strip()
    matches = re.findall(r'\{[^}]+\}', clean)
    for m in reversed(matches):
        try:
            parsed = json.loads(m)
            winner = parsed.get('winner', '').strip()
            if str(team_a).lower() in winner.lower():
                return team_a
            if str(team_b).lower() in winner.lower():
                return team_b
        except:
            continue
    return "UNCLEAR"

print("=== 원본 로직 재현 검증 ===")
for res_col, raw_col in [("Result_FewShot", "Raw_FewShot"),
                          ("Result_CoT", "Raw_CoT"),
                          ("Result_Role", "Raw_Role")]:
    reproduced = df2.apply(lambda r: extract_winner_json_v2(r[raw_col], r["Team A"], r["Team B"]), axis=1)
    mismatch = (reproduced != df2[res_col]).sum()
    print(f"{res_col}: 재현 결과와 저장값 불일치 {mismatch}건")

reproduced_fs = df2.apply(lambda r: extract_winner_json_v2(r["Raw_FewShot"], r["Team A"], r["Team B"]), axis=1)
mask = reproduced_fs != df2["Result_FewShot"]

for idx, row in df2[mask].iterrows():
    print(f"--- {row['Scenario ID']} ---")
    print(f"Team A: {row['Team A']} / Team B: {row['Team B']} / 실제 승자: {row['Winner']}")
    print(f"저장된 Result: {row['Result_FewShot']}")
    print(f"재현된 Result: {reproduced_fs[idx]}")
    print(f"Raw 마지막 200자: {str(row['Raw_FewShot'])[-200:]}\n")

import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv("experiment_results_440_reparsed.csv")
df["actual"] = (df["Winner"] == df["Team A"]).astype(int)

print("=== v3 교정 후 Team A/B Recall ===")
for col in ["Result_Fewshot_v3", "Result_CoT_v3", "Result_Role_v3"]:
    pred = (df[col] == df["Team A"]).astype(int)
    print(f"\n{col}")
    print(classification_report(df["actual"], pred, labels=[1, 0],
                                 target_names=["Team A win", "Team B win"], digits=3))

import pandas as pd

df = pd.read_csv("dataset_440_final.csv")
print("=== Scenario Text 샘플 ===")
print(str(df.iloc[0]["Scenario Text"])[:800])
print("\n=== Prompt_FewShot 구조 (앞 500자) ===")
print(str(df.iloc[0]["Prompt_FewShot"])[:500])

import pandas as pd

df = pd.read_csv("dataset_440_json_v3.csv")
row = df.iloc[0]

# 1. Scenario Text가 프롬프트 안에 그대로 포함돼 있는지
for col in ["Prompt_FewShot", "Prompt_CoT", "Prompt_Role"]:
    contained = str(row["Scenario Text"]) in str(row[col])
    print(f"{col}: Scenario Text 그대로 포함? {contained}")

# 2. 프롬프트 끝부분 (지시문 확인)
print("\n=== Prompt_CoT 마지막 500자 ===")
print(str(row["Prompt_CoT"])[-500:])