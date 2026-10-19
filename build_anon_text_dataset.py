"""텍스트 파싱 조건(본실험)용 익명 데이터셋 생성.

dataset_440_final.csv (실명) -> dataset_440_text_anon.csv (익명)

바꾸는 것 (JSON 익명 데이터셋과 동일한 규칙):
  - 대상 경기: [경기] 줄, 팀A:/팀B: 줄의 팀명 -> Team A / Team B
  - 응답 지시문: "X 또는 Y로(만) 답해" -> "Team A 또는 Team B로(만) 답해"
  - Few-shot 예시: "팀A: FURIA |" -> "팀A: Team A |" (예시 3개의 팀명 자리만)
  - Team A / Team B / Winner / Match Name 컬럼 (원본은 Orig_* 컬럼에 보존)
안 바꾸는 것: 스탯 줄, 에이전트 구성, 대회명, 맵, 예시 정답("→ 팀B 승" 등)과 그 순서.

사용법: python build_anon_text_dataset.py [입력.csv] [출력.csv]
"""
import re
import sys
import pandas as pd

INPUT = sys.argv[1] if len(sys.argv) > 1 else 'dataset_440_final.csv'
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else 'dataset_440_text_anon.csv'
EX_SPLIT = '이제 아래 경기를 예측해.'
EXAMPLE_NAMES = ['FURIA', 'Paper Rex', 'T1', 'G2 Esports', 'Team Liquid', 'Sentinels']


def has_name(text, name):
    return re.search(r'(?<![A-Za-z0-9])' + re.escape(name) + r'(?![A-Za-z0-9])', text) is not None


def anon_scenario(s):
    s2 = re.sub(r'^\[경기\] .*$', '[경기] Team A vs Team B', s, flags=re.M)
    s2 = re.sub(r'^팀A: .*$', '팀A: Team A', s2, flags=re.M)
    s2 = re.sub(r'^팀B: .*$', '팀B: Team B', s2, flags=re.M)
    changed = sum(1 for a, b in zip(s.split('\n'), s2.split('\n')) if a != b)
    assert changed == 3, f'시나리오에서 바뀐 줄이 3개가 아님: {changed}'
    return s2


def anon_prompt(prompt, scen, scen_anon, a, b, kind):
    assert scen in prompt, '프롬프트 안에 Scenario Text가 없음'
    p = prompt.replace(scen, scen_anon)
    instr = f'{a} 또는 {b}로'
    assert instr in p, f'응답 지시문을 찾지 못함: {instr}'
    p = p.replace(instr, 'Team A 또는 Team B로')
    if kind == 'FewShot':
        assert EX_SPLIT in p
        head, tail = p.split(EX_SPLIT, 1)
        head = re.sub(r'^(팀A: )[^|\n]+?( \|)', r'\1Team A\2', head, flags=re.M)
        head = re.sub(r'^(팀B: )[^|\n]+?( \|)', r'\1Team B\2', head, flags=re.M)
        p = head + EX_SPLIT + tail
    return p


df = pd.read_csv(INPUT, dtype=str, encoding='utf-8-sig')
rows = []
for _, r in df.iterrows():
    a, b, w = r['Team A'], r['Team B'], r['Winner']
    assert w in (a, b), f"{r['Scenario ID']}: Winner가 Team A/B 중 하나가 아님"
    scen_anon = anon_scenario(r['Scenario Text'])
    n = r.copy()
    for col, kind in [('Prompt_FewShot', 'FewShot'), ('Prompt_CoT', 'CoT'), ('Prompt_Role', 'Role')]:
        n[col] = anon_prompt(r[col], r['Scenario Text'], scen_anon, a, b, kind)
        for name in {a, b} | set(EXAMPLE_NAMES):
            assert not has_name(n[col], name), f"{r['Scenario ID']} {col}: 실명 '{name}'이 남아 있음"
    n['Scenario Text'] = scen_anon
    n['Orig_Team A'], n['Orig_Team B'], n['Orig_Winner'], n['Orig_Match Name'] = a, b, w, r['Match Name']
    n['Team A'], n['Team B'] = 'Team A', 'Team B'
    n['Winner'] = 'Team A' if w == a else 'Team B'
    n['Match Name'] = 'Team A vs Team B'
    for col in ['Result_FewShot', 'Result_CoT', 'Result_Role', 'Correct_FewShot', 'Correct_CoT',
                'Correct_Role', 'Raw_FewShot', 'Raw_CoT', 'Raw_Role']:
        n[col] = ''
    rows.append(n)

out = pd.DataFrame(rows)
out.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
print(f'저장 완료: {OUTPUT} ({len(out)}행)')
print('Winner 분포:', out['Winner'].value_counts().to_dict())
print('자체 검사 통과: 모든 프롬프트에서 실명 0건, 시나리오 변경 줄 수 3줄')