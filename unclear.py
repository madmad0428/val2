import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("experiment_results_440.csv")

techniques = {
    "Few-shot": "Result_FewShot",
    "CoT": "Result_CoT",
    "Role": "Result_Role",
}

# Positive = Team A 승리, Negative = Team B 승리로 정의
df["actual"] = (df["Winner"] == df["Team A"]).astype(int)

for name, col in techniques.items():
    print(f"\n{'='*40}\n{name}\n{'='*40}")

    # 예측이 Team A/Team B 둘 중 무엇도 아닌 경우 = 파싱 실패(UNCLEAR 등)
    is_team_a = df[col] == df["Team A"]
    is_team_b = df[col] == df["Team B"]
    is_unclear = ~(is_team_a | is_team_b)

    print(f"파싱 실패/UNCLEAR 건수: {is_unclear.sum()} / {len(df)}")

    # 방식: UNCLEAR을 '오답으로 강제 처리'하되, Positive/Negative 지정 필요
    # → UNCLEAR는 Team A도 아니므로 자동으로 predicted=0(Team B 승리 예측)으로 처리됨
    #   이는 기존 정확도 계산과 동일한 효과(무조건 오답 카운트는 아님 주의, 아래 설명 참고)
    df["predicted"] = is_team_a.astype(int)

    print(confusion_matrix(df["actual"], df["predicted"], labels=[1, 0]))
    print(classification_report(df["actual"], df["predicted"],
                                 labels=[1, 0], target_names=["Team A win", "Team B win"],
                                 digits=3))


print(f"\n{'='*40}\nTeam A 실제 승리 비율\n{'='*40}")
print(df["actual"].value_counts(normalize=True))

import pandas as pd

df = pd.read_csv("experiment_results_440.csv")

# CoT에서 Team A로 판정된 케이스 중, Raw 텍스트에 Team B도 언급된 경우 확인
both_mentioned = df.apply(
    lambda r: str(r["Team B"]).lower() in str(r["Raw_CoT"]).lower()
              and str(r["Team A"]).lower() in str(r["Raw_CoT"]).lower(),
    axis=1
)

print(f"CoT에서 두 팀 다 언급된 케이스: {both_mentioned.sum()} / 440")

# 그 중 몇 개 실제로 열어서 눈으로 확인
sample = df[both_mentioned].head(5)
for idx, row in sample.iterrows():
    print(f"\n--- {row['Scenario ID']} ---")
    print(f"Team A: {row['Team A']} / Team B: {row['Team B']} / 실제 승자: {row['Winner']}")
    print(f"파싱된 결과: {row['Result_CoT']}")
    print(f"Raw 마지막 300자:\n{str(row['Raw_CoT'])[-300:]}")

import pandas as pd

df = pd.read_csv("experiment_results_440.csv")

for col_result, col_raw in [
    ("Result_FewShot", "Raw_FewShot"),
    ("Result_CoT", "Raw_CoT"),
    ("Result_Role", "Raw_Role"),
]:
    # 두 팀 이름이 raw 텍스트 마지막 300자에 다 등장하는 케이스
    def both_in_last(row):
        last = str(row[col_raw])[-300:].lower()
        return str(row["Team A"]).lower() in last and str(row["Team B"]).lower() in last

    mask = df.apply(both_in_last, axis=1)
    # 그 중 파싱 결과가 Team A로 나온 비율 (버그 의심 케이스)
    suspect = df[mask & (df[col_result] == df["Team A"])]

    print(f"\n{col_result}")
    print(f"두 팀 다 언급된 케이스: {mask.sum()}")
    print(f"그 중 Team A로 파싱됨(버그 의심): {len(suspect)}")

import pandas as pd

df = pd.read_csv("experiment_results_440.csv")

for col_result, col_raw, col_correct in [
    ("Result_FewShot", "Raw_FewShot", "Correct_FewShot"),
    ("Result_CoT", "Raw_CoT", "Correct_CoT"),
    ("Result_Role", "Raw_Role", "Correct_Role"),
]:
    def both_in_last(row):
        last = str(row[col_raw])[-300:].lower()
        return str(row["Team A"]).lower() in last and str(row["Team B"]).lower() in last

    mask = df.apply(both_in_last, axis=1)
    suspect = df[mask & (df[col_result] == df["Team A"])]

    # 이 중 실제로 Team A가 승자가 아닌 경우 = 진짜 오분류(버그로 인한 피해)
    actually_wrong = suspect[suspect["Winner"] != suspect["Team A"]]
    # 우연히 Team A가 진짜 승자였던 경우 = 버그가 있었지만 결과적으로 맞음
    coincidentally_right = suspect[suspect["Winner"] == suspect["Team A"]]

    print(f"\n{col_result}")
    print(f"버그 의심 케이스: {len(suspect)}")
    print(f"  → 실제 오분류(버그로 인한 손실): {len(actually_wrong)}")
    print(f"  → 우연히 정답(Team A가 실제 승자): {len(coincidentally_right)}")

import re

def extract_winner_v2(text, team_a, team_b):
    conclusion_markers = ["최종 승자", "승자:", "예상 최종 스코어", "🏆", "결론"]
    best_marker_idx = -1
    for marker in conclusion_markers:
        idx = text.rfind(marker)
        if idx > best_marker_idx:
            best_marker_idx = idx
    if best_marker_idx != -1:
        snippet = text[best_marker_idx:best_marker_idx+150]
        pos_a = snippet.lower().find(str(team_a).lower())
        pos_b = snippet.lower().find(str(team_b).lower())
        if pos_a != -1 and (pos_b == -1 or pos_a < pos_b):
            return team_a
        if pos_b != -1 and (pos_a == -1 or pos_b < pos_a):
            return team_b
    last = text[-300:]
    for team in [team_a, team_b]:
        if str(team).lower() in last.lower():
            return team
    return "UNCLEAR"

print(f"\n{'='*40}\n재파싱 결과 (v2)\n{'='*40}")

for tech, raw_col in [("Few-shot", "Raw_FewShot"), ("CoT", "Raw_CoT"), ("Role", "Raw_Role")]:
    new_results = df.apply(
        lambda r: extract_winner_v2(str(r[raw_col]), r["Team A"], r["Team B"]), axis=1
    )
    new_correct = (new_results == df["Winner"]).sum()
    unclear_count = (new_results == "UNCLEAR").sum()
    print(f"{tech}: {new_correct}/440 ({round(new_correct/440*100,1)}%) | UNCLEAR: {unclear_count}")

    df[f"Result_{tech.replace('-','')}_v2"] = new_results

df.to_csv("experiment_results_440_reparsed.csv", index=False)
print("\n저장 완료: experiment_results_440_reparsed.csv")

import pandas as pd

df = pd.read_csv("experiment_results_440_reparsed.csv")

# Role의 UNCLEAR 케이스 샘플 확인
mask = df["Result_Role_v2"] == "UNCLEAR"
print(f"Role UNCLEAR: {mask.sum()}건\n")

for idx, row in df[mask].head(3).iterrows():
    print(f"--- {row['Scenario ID']} ---")
    print(f"Team A: {row['Team A']} / Team B: {row['Team B']} / 승자: {row['Winner']}")
    print(f"Raw 처음 200자:\n{str(row['Raw_Role'])[:200]}")
    print(f"\nRaw 마지막 200자:\n{str(row['Raw_Role'])[-200:]}\n")

def extract_winner_v3(text, team_a, team_b):
    text = str(text)
    ta, tb = str(team_a), str(team_b)

    # 1순위: 명시적 승자 선언 (Role/CoT 공통 패턴) — 문서 전체에서 '처음' 등장하는 것
    for marker in ["승리팀:", "승리팀 :", "최종 승자:", "승자:", "🏆 최종 승자:"]:
        idx = text.find(marker)
        if idx != -1:
            snippet = text[idx:idx+80].lower()
            pos_a = snippet.find(ta.lower())
            pos_b = snippet.find(tb.lower())
            if pos_a != -1 and (pos_b == -1 or pos_a < pos_b):
                return ta
            if pos_b != -1 and (pos_a == -1 or pos_b < pos_a):
                return tb

    # 2순위: 텍스트 첫 150자에 팀 이름 단독 선언 (예: "**KRÜ Esports**\n\n분석 근거")
    head = text[:150].lower()
    pos_a, pos_b = head.find(ta.lower()), head.find(tb.lower())
    if pos_a != -1 and pos_b == -1:
        return ta
    if pos_b != -1 and pos_a == -1:
        return tb

    # 3순위: 마지막 300자에서 '더 마지막에' 언급된 팀 (결론 문장에 가까운 쪽)
    last = text[-300:].lower()
    pos_a, pos_b = last.rfind(ta.lower()), last.rfind(tb.lower())
    if pos_a > pos_b:
        return ta
    if pos_b > pos_a:
        return tb

    return "UNCLEAR"

print(f"\n{'='*40}\n재파싱 결과 (v3)\n{'='*40}")

for tech, raw_col in [("Few-shot", "Raw_FewShot"), ("CoT", "Raw_CoT"), ("Role", "Raw_Role")]:
    new_results = df.apply(
        lambda r: extract_winner_v3(r[raw_col], r["Team A"], r["Team B"]), axis=1
    )
    new_correct = (new_results == df["Winner"]).sum()
    unclear_count = (new_results == "UNCLEAR").sum()
    print(f"{tech}: {new_correct}/440 ({round(new_correct/440*100,1)}%) | UNCLEAR: {unclear_count}")
    df[f"Result_{tech.replace('-','')}_v3"] = new_results

df.to_csv("experiment_results_440_reparsed.csv", index=False)
print("\n저장 완료")

import pandas as pd
df = pd.read_csv("experiment_results_440_reparsed.csv")

# v3 컬럼 실제 이름 확인
print([c for c in df.columns if "_v3" in c])

for res_col, raw_col in [
    ("Result_Fewshot_v3", "Raw_FewShot"),
    ("Result_CoT_v3", "Raw_CoT"),
    ("Result_Role_v3", "Raw_Role"),
]:
    mask = df[res_col] == "UNCLEAR"
    for idx, row in df[mask].iterrows():
        print(f"--- {res_col} {row['Scenario ID']} ---")
        print(f"A: {row['Team A']} / B: {row['Team B']} / 승자: {row['Winner']}")
        print(f"끝 200자: {str(row[raw_col])[-200:]}\n")

# 수동 검증용: UNCLEAR 10건의 원문 전체를 텍스트 파일로 추출
targets = [
    ("S377", "Raw_FewShot"),
    ("S038", "Raw_CoT"),
    ("S120", "Raw_CoT"),
    ("S144", "Raw_CoT"),
    ("S198", "Raw_CoT"),
    ("S209", "Raw_CoT"),
    ("S244", "Raw_CoT"),
    ("S148", "Raw_Role"),
    ("S316", "Raw_Role"),
    ("S414", "Raw_Role"),
]

with open("manual_check.txt", "w", encoding="utf-8") as f:
    for sid, raw_col in targets:
        row = df[df["Scenario ID"] == sid].iloc[0]
        f.write(f"{'='*60}\n")
        f.write(f"[{sid}] ({raw_col})\n")
        f.write(f"Team A: {row['Team A']} / Team B: {row['Team B']} / 실제 승자: {row['Winner']}\n")
        f.write(f"{'='*60}\n")
        f.write(str(row[raw_col]))
        f.write("\n\n\n")

print("manual_check.txt 생성 완료")

manual_overrides = {
    ("S377", "Result_Fewshot_v3"): "Natus Vincere",
    ("S038", "Result_CoT_v3"): "Mega Minors",
    ("S120", "Result_CoT_v3"): "Team Heretics",
    ("S144", "Result_CoT_v3"): "Team Heretics",
    ("S198", "Result_CoT_v3"): "Rex Regum Qeon",
    ("S209", "Result_CoT_v3"): "Kiwoom DRX",
    ("S244", "Result_CoT_v3"): "BBL Esports",
    ("S148", "Result_Role_v3"): "FUT Esports",
    ("S316", "Result_Role_v3"): "G2 Esports",
    ("S414", "Result_Role_v3"): "Rex Regum Qeon",
}

for (sid, col), value in manual_overrides.items():
    df.loc[df["Scenario ID"] == sid, col] = value

print(f"\n{'='*40}\n최종 결과 (v3 + 수동 판독 10건)\n{'='*40}")
for col in ["Result_Fewshot_v3", "Result_CoT_v3", "Result_Role_v3"]:
    correct = (df[col] == df["Winner"]).sum()
    unclear = (df[col] == "UNCLEAR").sum()
    print(f"{col}: {correct}/440 ({round(correct/440*100,1)}%) | UNCLEAR: {unclear}")

df.to_csv("experiment_results_440_reparsed.csv", index=False)
print("저장 완료")

from statsmodels.stats.contingency_tables import mcnemar

print(f"\n{'='*40}\nMcNemar 검정 (v3 최종 결과)\n{'='*40}")

correct = {}
for tech, col in [("Few-shot", "Result_Fewshot_v3"),
                  ("CoT", "Result_CoT_v3"),
                  ("Role", "Result_Role_v3")]:
    correct[tech] = (df[col] == df["Winner"]).astype(int)

pairs = [("Few-shot", "CoT"), ("Few-shot", "Role"), ("CoT", "Role")]
alpha_bonferroni = 0.05 / 3  # 기존 논문과 동일하게 Bonferroni 보정

for a, b in pairs:
    # 분할표: [둘 다 정답, a만 정답 / b만 정답, 둘 다 오답]
    both = ((correct[a] == 1) & (correct[b] == 1)).sum()
    only_a = ((correct[a] == 1) & (correct[b] == 0)).sum()
    only_b = ((correct[a] == 0) & (correct[b] == 1)).sum()
    neither = ((correct[a] == 0) & (correct[b] == 0)).sum()

    table = [[both, only_a], [only_b, neither]]
    result = mcnemar(table, exact=True)
    sig = "유의함" if result.pvalue < alpha_bonferroni else "유의하지 않음"
    print(f"{a} vs {b}: p={result.pvalue:.4f} ({sig}, α={alpha_bonferroni:.4f})")