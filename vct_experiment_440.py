import anthropic
import pandas as pd
import time
import os

client = anthropic.Anthropic(api_key="   ")

INPUT_FILE = 'dataset_440_final.csv'
OUTPUT_FILE = 'experiment_results_440.csv'

df = pd.read_csv(INPUT_FILE)

# 컬럼 타입 강제로 string으로 변환
for col in ['Result_FewShot','Result_CoT','Result_Role',
            'Raw_FewShot','Raw_CoT','Raw_Role',
            'Correct_FewShot','Correct_CoT','Correct_Role']:
    df[col] = df[col].astype(str)

if os.path.exists(OUTPUT_FILE):
    done = pd.read_csv(OUTPUT_FILE, dtype=str)
    done_ids = set(done[done['Correct_FewShot'].notna() & (done['Correct_FewShot'] != '') & (done['Correct_FewShot'] != 'nan')]['Scenario ID'].tolist())
    df = done.copy()
    for col in ['Result_FewShot','Result_CoT','Result_Role',
                'Raw_FewShot','Raw_CoT','Raw_Role',
                'Correct_FewShot','Correct_CoT','Correct_Role']:
        df[col] = df[col].astype(str)
    print(f"이미 처리된 시나리오: {len(done_ids)}개")
else:
    done_ids = set()

def extract_winner(text, team_a, team_b):
    last = text[-300:]
    for team in [team_a, team_b]:
        if team.lower() in last.lower():
            return team
    for team in [team_a, team_b]:
        if team.lower() in text.lower():
            return team
    return "UNCLEAR"

total = len(df)
for i, idx in enumerate(df.index):
    row = df.loc[idx]
    sid = row['Scenario ID']

    if sid in done_ids:
        continue

    print(f"[{i+1}/{total}] {sid} 처리 중...")

    for prompt_col, result_col, correct_col, raw_col in [
        ('Prompt_FewShot','Result_FewShot','Correct_FewShot','Raw_FewShot'),
        ('Prompt_CoT','Result_CoT','Correct_CoT','Raw_CoT'),
        ('Prompt_Role','Result_Role','Correct_Role','Raw_Role'),
    ]:
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=3000,
                messages=[{"role": "user", "content": str(row[prompt_col])}]
            )
            raw = response.content[0].text
            result = extract_winner(raw, str(row['Team A']), str(row['Team B']))
            correct = '1' if result == str(row['Winner']) else '0'

            df.at[idx, raw_col] = raw
            df.at[idx, result_col] = result
            df.at[idx, correct_col] = correct
            time.sleep(0.3)

        except Exception as e:
            print(f"  ERROR ({prompt_col}): {e}")
            df.at[idx, result_col] = 'ERROR'
            df.at[idx, correct_col] = '-1'

    fs = df.at[idx, 'Correct_FewShot']
    cot = df.at[idx, 'Correct_CoT']
    role = df.at[idx, 'Correct_Role']
    print(f"  FewShot={'정답' if fs=='1' else '오답'} | CoT={'정답' if cot=='1' else '오답'} | Role={'정답' if role=='1' else '오답'}")

    if (i + 1) % 20 == 0:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"  >>> {i+1}개 중간 저장 완료")

    time.sleep(0.5)

df.to_csv(OUTPUT_FILE, index=False)

print("\n=== 실험 완료 ===")
for col in ['Correct_FewShot','Correct_CoT','Correct_Role']:
    acc = (df[col] == '1').sum()
    unclear = (df[col] == '-1').sum()
    print(f"{col}: {acc}/440 ({round(acc/440*100,1)}%) | UNCLEAR: {unclear}")