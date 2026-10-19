"""Few-shot만 재실험 (예시 팀명까지 익명화). 설정은 vct_run_anon.py와 동일.
사용법(PowerShell): $env:ANTHROPIC_API_KEY="sk-ant-..." ; python vct_run_anon_fs.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import anthropic
import pandas as pd
import time
import json
import os
import re

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

INPUT_FILE = 'dataset_440_json_anon_fs.csv'
OUTPUT_FILE = 'experiment_results_json_anon_fs.csv'

if os.path.exists(OUTPUT_FILE):
    df = pd.read_csv(OUTPUT_FILE, dtype=str, encoding='utf-8-sig')
else:
    df = pd.read_csv(INPUT_FILE, dtype=str, encoding='utf-8-sig')
for col in ['Result_FewShot', 'Raw_FewShot', 'Correct_FewShot']:
    df[col] = df[col].fillna('').astype(str)
done_ids = set(df[df['Correct_FewShot'].isin(['0', '1'])]['Scenario ID'])
print(f"이미 처리된 시나리오: {len(done_ids)}개")


def extract_winner_json(text, team_a, team_b):
    try:
        clean = re.sub(r'```json|```', '', text).strip()
        for m in reversed(re.findall(r'\{[^}]+\}', clean)):
            try:
                winner = json.loads(m).get('winner', '').strip()
                if team_a.lower() in winner.lower():
                    return team_a
                if team_b.lower() in winner.lower():
                    return team_b
            except Exception:
                continue
    except Exception:
        pass
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    for line in reversed(lines[-5:]):
        for team in [team_a, team_b]:
            if team.lower() in line.lower():
                return team
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
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=3400,
            messages=[{"role": "user", "content": str(row['Prompt_FewShot'])}]
        )
        raw = response.content[0].text
        result = extract_winner_json(raw, str(row['Team A']), str(row['Team B']))
        df.at[idx, 'Raw_FewShot'] = raw
        df.at[idx, 'Result_FewShot'] = result
        df.at[idx, 'Correct_FewShot'] = '1' if result == str(row['Winner']) else '0'
        time.sleep(0.3)
    except Exception as e:
        if 'credit balance' in str(e):
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            print(f"\n!!! 크레딧 부족으로 중단 ({sid}). 충전 후 다시 실행하면 이어서 진행됩니다.")
            sys.exit(1)
        print(f"  ERROR: {repr(e)}")
        df.at[idx, 'Result_FewShot'] = 'ERROR'
        df.at[idx, 'Correct_FewShot'] = '-1'
    v = df.at[idx, 'Correct_FewShot']
    print(f"  FewShot={'정답' if v == '1' else ('ERROR' if v == '-1' else '오답')}")
    if (i + 1) % 20 == 0:
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"  >>> {i+1}개 중간 저장 완료")
    time.sleep(0.5)

df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
acc = (df['Correct_FewShot'] == '1').sum()
err = (df['Correct_FewShot'] == '-1').sum()
print(f"\n=== 최종 결과 ===\nCorrect_FewShot: {acc}/{total} ({round(acc/total*100,1)}%) | API ERROR(-1): {err}")
if err:
    print("  ※ ERROR가 남아 있음 → 다시 실행해서 이어서 돌릴 것")