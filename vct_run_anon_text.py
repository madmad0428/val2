"""익명 텍스트 파싱 재실험 호출 (본실험 vct_experiment_440.py와 같은 조건).

- 입력:  dataset_440_text_anon.csv  (build_anon_text_dataset.py로 생성)
- 출력:  experiment_results_text_anon.csv  (원문 응답만 저장, 파싱은 별도)
- 조건:  claude-haiku-4-5-20251001, max_tokens=3000, temperature 미지정(API 기본값)
         -> 기존 실험이 temperature를 지정하지 않았으므로 일부러 맞춤. 바꾸지 말 것.
- 본실험은 응답이 3000 토큰에서 잘린 CoT를 max_tokens=3400으로 다시 불렀음.
  여기서는 글자 수 대신 stop_reason == 'max_tokens'로 잘림을 판정해 같은 방식으로 3400 재호출하고
  Rerun3400_* 컬럼에 표시한다.
- 이어서 하기: 이미 응답이 있는 칸은 건너뜀. 중단 후 다시 실행하면 이어서 진행.
- 소규모 시험: python vct_run_anon_text.py --limit 3   (시나리오 3개 = 최대 9회 호출)

필요: pip install anthropic pandas / 환경변수 ANTHROPIC_API_KEY (키를 코드에 쓰지 말 것)
"""
import os
import sys
import time
import pandas as pd
import anthropic

INPUT_FILE = 'dataset_440_text_anon.csv'
OUTPUT_FILE = 'experiment_results_text_anon.csv'
MODEL = 'claude-haiku-4-5-20251001'
MAX_TOKENS = 3000
MAX_TOKENS_RERUN = 3400
TECHS = [('FewShot', 'Prompt_FewShot'), ('CoT', 'Prompt_CoT'), ('Role', 'Prompt_Role')]

limit = None
if '--limit' in sys.argv:
    limit = int(sys.argv[sys.argv.index('--limit') + 1])

client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

if os.path.exists(OUTPUT_FILE):
    df = pd.read_csv(OUTPUT_FILE, dtype=str, encoding='utf-8-sig').fillna('')
    print(f'기존 결과 이어서 진행: {OUTPUT_FILE}')
else:
    df = pd.read_csv(INPUT_FILE, dtype=str, encoding='utf-8-sig').fillna('')
for t, _ in TECHS:
    for prefix in ['Raw_', 'Stop_', 'Rerun3400_']:
        if prefix + t not in df.columns:
            df[prefix + t] = ''
if 'Model_Used' not in df.columns:
    df['Model_Used'] = ''


def call(prompt, max_tokens):
    """일시 오류는 3회까지 재시도. 크레딧 부족이면 저장하고 종료."""
    for attempt in range(3):
        try:
            return client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                messages=[{'role': 'user', 'content': prompt}],
            )
        except Exception as e:
            if 'credit balance' in str(e):
                df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
                print('\n!!! 크레딧 부족으로 중단. 충전 후 다시 실행하면 이어서 진행됩니다.')
                sys.exit(1)
            print(f'  재시도 {attempt + 1}/3: {repr(e)}')
            time.sleep(5 * (attempt + 1))
    return None


def done(idx, t):
    raw = df.at[idx, 'Raw_' + t]
    return raw != '' and raw != 'ERROR'


total = len(df) if limit is None else min(limit, len(df))
for i, idx in enumerate(df.index[:total]):
    sid = df.at[idx, 'Scenario ID']
    todo = [(t, c) for t, c in TECHS if not done(idx, t)]
    if not todo:
        continue
    print(f'[{i + 1}/{total}] {sid} 처리 중...')
    for t, prompt_col in todo:
        prompt = df.at[idx, prompt_col]
        resp = call(prompt, MAX_TOKENS)
        if resp is not None and resp.stop_reason == 'max_tokens':
            resp2 = call(prompt, MAX_TOKENS_RERUN)
            if resp2 is not None:
                resp = resp2
                df.at[idx, 'Rerun3400_' + t] = '1'
        if resp is None:
            df.at[idx, 'Raw_' + t] = 'ERROR'
            df.at[idx, 'Stop_' + t] = 'ERROR'
        else:
            df.at[idx, 'Raw_' + t] = resp.content[0].text
            df.at[idx, 'Stop_' + t] = str(resp.stop_reason)
            df.at[idx, 'Model_Used'] = str(resp.model)
        time.sleep(0.3)
    time.sleep(0.5)
    if (i + 1) % 20 == 0:
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f'  >>> {i + 1}개 중간 저장 완료')

df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print('\n=== 호출 완료 ===')
for t, _ in TECHS:
    got = sum(1 for idx in df.index[:total] if done(idx, t))
    err = int((df['Raw_' + t] == 'ERROR').sum())
    cut = int((df['Stop_' + t] == 'max_tokens').sum())
    rr = int((df['Rerun3400_' + t] == '1').sum())
    print(f'{t}: 응답 {got}/{total} | ERROR {err} | 3400 재호출 {rr} | 재호출 후에도 잘림 {cut}')
print('사용 모델:', sorted(set(df['Model_Used']) - {''}))
if any((df['Raw_' + t] == 'ERROR').any() for t, _ in TECHS):
    print('※ ERROR가 남아 있음 -> 다시 실행하면 해당 칸만 재호출합니다.')
print('다음 단계: 응답 파싱(교정 파서)은 별도로 수행. 이 파일에는 원문 응답만 있음.')