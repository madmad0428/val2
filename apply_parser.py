"""익명 텍스트 파싱 응답에 파서를 적용하고, 사람이 읽어야 할 행(review)을 뽑는다.

입력 (레포 폴더에 있어야 함)
  - experiment_results_text_anon.csv        익명 응답 (vct_run_anon_text.py 결과)
  - experiment_results_440_reparsed.csv     실명 응답 + 저장된 교정 결과(v3)
  - corrected_parser.py                     교정 파서
출력
  - parsed_text_anon.csv     익명: 기법별 ①원래 파서, ②교정 파서 결과
  - parsed_text_named.csv    실명: 같은 파서로 다시 파싱한 결과 (비교용)
  - review_text.csv          사람이 읽고 Human_pick을 채울 행 (Scenario ID, 조건, 기법, 응답 끝부분 등)

사용법: python apply_parser.py
"""
import re
import pandas as pd
from corrected_parser import parse_winner_ex, _aliases, _find_all

TECH = ['FewShot', 'CoT', 'Role']
V3 = {'FewShot': 'Result_Fewshot_v3', 'CoT': 'Result_CoT_v3', 'Role': 'Result_Role_v3'}


def extract_winner_orig(text, team_a, team_b):
    """본실험 파서(vct_experiment_440.py의 extract_winner)와 동일 — ① 텍스트 파싱용"""
    last = text[-300:]
    for team in [team_a, team_b]:
        if team.lower() in last.lower():
            return team
    for team in [team_a, team_b]:
        if team.lower() in text.lower():
            return team
    return 'UNCLEAR'


def independent_picks(raw, a, b, mn):
    """마커 없는 응답용 보조 판정 2개: 언급 횟수가 많은 팀 / 처음 언급된 팀 (동률이면 None)"""
    na, nb = _aliases(a, b, 'A', mn), _aliases(b, a, 'B', mn)
    pa, pb = _find_all(raw, na), _find_all(raw, nb)
    cnt = ('A' if len(pa) > len(pb) else 'B') if len(pa) != len(pb) else None
    first = ('A' if pa[0] < pb[0] else 'B') if (pa and pb) else ('A' if pa else ('B' if pb else None))
    return cnt, first


def run(df, label, has_v3):
    out = df[['Scenario ID', 'Team A', 'Team B', 'Winner']].copy()
    review = []
    for t in TECH:
        orig, fixed, rule = [], [], []
        for _, r in df.iterrows():
            raw = str(r[f'Raw_{t}'])
            a, b = r['Team A'], r['Team B']
            orig.append(extract_winner_orig(raw, a, b))
            res, rl = parse_winner_ex(raw, a, b, r['Match Name'])
            fixed.append({'A': a, 'B': b, 'UNCLEAR': 'UNCLEAR'}[res])
            rule.append(rl)
        out[f'Orig_{t}'] = orig
        out[f'Fixed_{t}'] = fixed
        out[f'Rule_{t}'] = rule
        out[f'OrigOK_{t}'] = (out[f'Orig_{t}'] == out['Winner']).astype(int)
        out[f'FixedOK_{t}'] = (out[f'Fixed_{t}'] == out['Winner']).astype(int)
        for i, r in df.iterrows():
            need, why = False, ''
            if rule[i] != 'marker':                          # 명시 마커가 없는 응답만 점검
                cnt, first = independent_picks(str(r[f'Raw_{t}']), r['Team A'], r['Team B'], r['Match Name'])
                if label == '익명':
                    mine = {'Team A': 'A', 'Team B': 'B'}.get(fixed[i])
                else:
                    mine = 'A' if fixed[i] == r['Team A'] else ('B' if fixed[i] == r['Team B'] else None)
                if mine is None or cnt != mine or first != mine:   # 세 방법이 모두 같을 때만 통과
                    need, why = True, '마커 없음, 판정 방법 간 불일치'
            if has_v3 and fixed[i] != r[V3[t]]:               # 저장된 v3와 판정이 다르면 확인
                need, why = True, (why + ' + ' if why else '') + 'v3와 불일치'
            if need:
                review.append({'조건': label, 'Scenario ID': r['Scenario ID'], '기법': t,
                               'Team A': r['Team A'], 'Team B': r['Team B'],
                               '파서판정': {'A': 'Team A', 'B': 'Team B'}.get(fixed[i], fixed[i]) if label == '익명' else fixed[i],
                               '사유': why, '응답끝부분': str(r[f'Raw_{t}'])[-400:].replace('\n', ' ⏎ '),
                               'Human_pick': ''})
    return out, review


named = pd.read_csv('experiment_results_440_reparsed.csv', dtype=str).fillna('')
anon = pd.read_csv('experiment_results_text_anon.csv', dtype=str, encoding='utf-8-sig').fillna('')

o_named, r_named = run(named, '실명', True)
o_anon, r_anon = run(anon, '익명', False)
o_named.to_csv('parsed_text_named.csv', index=False, encoding='utf-8-sig')
o_anon.to_csv('parsed_text_anon.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(r_named + r_anon).to_csv('review_text.csv', index=False, encoding='utf-8-sig')

print('== 파서 적용 결과 (사람 판정 전, 임시 값) ==')
for label, o in [('실명', o_named), ('익명', o_anon)]:
    for t in TECH:
        print(f"{label} {t:<8} ①원래 파서 {o[f'OrigOK_{t}'].sum()}/440 | ②교정 파서 {o[f'FixedOK_{t}'].sum()}/440"
              f" | UNCLEAR {(o[f'Fixed_{t}'] == 'UNCLEAR').sum()}")
print(f"\n사람이 읽을 행: 실명 {len(r_named)}행 + 익명 {len(r_anon)}행 -> review_text.csv")