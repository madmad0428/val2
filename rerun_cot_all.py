import anthropic
import pandas as pd
import time

client = anthropic.Anthropic(api_key="  ")
df = pd.read_csv('experiment_results_440.csv', dtype=str)

def extract_winner(text, team_a, team_b):
    last = text[-300:]
    for team in [team_a, team_b]:
        if team.lower() in last.lower():
            return team
    for team in [team_a, team_b]:
        if team.lower() in text.lower():
            return team
    return "UNCLEAR"

# UNCLEAR(-1)인 CoT만 재실행
targets = df[df['Correct_CoT'] == '-1'].index.tolist()
print(f"재실행 대상: {len(targets)}개")

for idx in targets:
    row = df.loc[idx]
    sid = row['Scenario ID']
    print(f"처리 중: {sid}")

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=3200,
            messages=[{"role": "user", "content": str(row['Prompt_CoT'])}]
        )
        raw = response.content[0].text
        result = extract_winner(raw, str(row['Team A']), str(row['Team B']))
        correct = '1' if result == str(row['Winner']) else '0'

        df.at[idx, 'Raw_CoT'] = raw
        df.at[idx, 'Result_CoT'] = result
        df.at[idx, 'Correct_CoT'] = correct
        print(f"  결과: {result} | 정답: {row['Winner']}")
        time.sleep(0.5)

    except Exception as e:
        print(f"  ERROR: {e}")

df.to_csv('experiment_results_440.csv', index=False)
print("저장 완료!")

for col in ['Correct_FewShot','Correct_CoT','Correct_Role']:
    acc = (df[col] == '1').sum()
    unclear = (df[col] == '-1').sum()
    print(f"{col}: {acc}/440 ({round(acc/440*100,1)}%) | UNCLEAR: {unclear}")