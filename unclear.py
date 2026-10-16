import pandas as pd

df = pd.read_csv("experiment_results_440.csv")
print(df.columns.tolist())
print(df.head(3))
print(f"\n총 행 수: {len(df)}")

for tech in ["few_shot", "cot", "role"]:
    sub = df[df["technique"] == tech]
    
    # UNCLEAR 처리 방식 명시 필요 — 두 가지 중 논문과 일관되게
    # 방식 A: UNCLEAR을 오답으로 처리 (기존 정확도 계산과 동일하다면)
    # 방식 B: UNCLEAR 제외 후 계산 (별도 보고)
    valid = sub[sub["predicted"].isin(["win", "loss"])]
    
    print(f"\n=== {tech} ===")
    print(f"유효 응답: {len(valid)}/{len(sub)}")
    print(confusion_matrix(valid["actual"], valid["predicted"], 
                           labels=["win", "loss"]))
    print(classification_report(valid["actual"], valid["predicted"],
                                labels=["win", "loss"], digits=3))
    
    print(df["actual"].value_counts(normalize=True))