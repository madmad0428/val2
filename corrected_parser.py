"""교정 파서 (재작성본). 응답 원문에서 승자를 뽑는다.

규칙 (우선순위 순)
  1. 명시적 마커("최종 승자", "최종 답변", "승리팀:" 등) 뒤에 나오는 팀.
     마커가 여러 번이면 마지막 마커부터 본다. 마커 뒤 '같은 줄 + 다음 줄'에 한 팀만 나올 때만 채택
     (표 머리처럼 두 팀이 같이 나오면 건너뜀. 같은 줄에 두 팀이면 먼저 나온 팀)
  2. 응답 첫 150자에 한 팀만 나오면 그 팀
  3. 마지막 300자에서 더 뒤에 나온 팀
  4. 모두 실패하면 UNCLEAR -> 사람이 원문을 읽고 판정

팀 인식: 대소문자 무시, 단어 경계 확인 ("Team A"가 "Team Alpha"에 걸리지 않게),
        한글 "팀A/팀B", 경기 줄([경기] X vs Y)의 별칭(NRG 등), 약칭(Team 제거, Esports 제거,
        상대 팀에 없는 4자 이상 단어: Heretics 등)도 같은 팀으로 본다.

사용: from corrected_parser import parse_winner
      parse_winner(raw_text, team_a, team_b, match_name=None) -> 'A' | 'B' | 'UNCLEAR'
"""
import re

MARKERS = ['최종 승자', '최종승자', '승리팀', '최종 답변', '최종답변', '최종 답', '최종 결론', '최종결론',
           '최종 예측', '최종예측', '예측 승자', '승자', '우승팀', '우승', '예측', '결론', 'winner', 'Winner']
STOP_WORDS = {'team', 'esports', 'gaming', 'club', 'e-sports'}


def _aliases(team, other, side, match_name):
    names = {team}
    if match_name and ' vs ' in match_name:
        left, right = [x.strip() for x in match_name.split(' vs ', 1)]
        names.add(left if side == 'A' else right)
    if team in ('Team A', 'Team B'):
        names.add('팀' + team[-1])
        return {n for n in names if n}
    base = re.sub(r'^Team\s+', '', team, flags=re.I)
    base = re.sub(r'\s+(Esports|Gaming|Club)$', '', base, flags=re.I)
    names.add(base)
    other_words = {w.lower() for w in re.findall(r'[A-Za-z0-9ÜÖÄü]+', other)}
    for w in re.findall(r'[A-Za-z0-9ÜÖÄü]+', team):
        if len(w) >= 4 and w.lower() not in STOP_WORDS and w.lower() not in other_words:
            names.add(w)
    return {n for n in names if len(n) >= 2}


def _find_all(text, names):
    pos = []
    for n in names:
        pat = r'(?<![A-Za-z0-9가-힣])' + re.escape(n) + r'(?![A-Za-z0-9])'
        for m in re.finditer(pat, text, flags=re.I):
            pos.append(m.start())
    return sorted(pos)


def parse_winner(text, team_a, team_b, match_name=None):
    return parse_winner_ex(text, team_a, team_b, match_name)[0]


def parse_winner_ex(text, team_a, team_b, match_name=None):
    """(결과, 규칙) 반환. 규칙: 'marker'(명시 마커) / 'head'(첫150자) / 'tail'(끝300자) / 'none'"""
    text = str(text).replace('\r\n', '\n')   # 윈도우 git checkout(CRLF)에서도 같은 결과가 나오게 통일
    na = _aliases(team_a, team_b, 'A', match_name)
    nb = _aliases(team_b, team_a, 'B', match_name)

    def which(seg, allow_both):
        pa, pb = _find_all(seg, na), _find_all(seg, nb)
        if pa and pb:
            return ('A' if pa[0] < pb[0] else 'B') if allow_both else None
        if pa:
            return 'A'
        if pb:
            return 'B'
        return None

    # 1) 명시적 마커: 마지막 마커부터
    hits = set()
    for mk in MARKERS:
        for m in re.finditer(re.escape(mk) + r'\s*[:：]?', text):
            hits.add(m.end())
    for end in sorted(hits, reverse=True):
        rest = text[end:end + 200]
        lines = rest.split('\n')
        same_line = lines[0]
        r = which(same_line, allow_both=True)          # 같은 줄: 먼저 나온 팀
        if r:
            return r, 'marker'
        nxt = next((l for l in lines[1:] if l.strip()), '')
        r = which(nxt, allow_both=False)               # 다음 줄: 한 팀만 나올 때
        if r:
            return r, 'marker'

    # 2) 첫 150자에 한 팀만
    head = text[:150]
    ha, hb = _find_all(head, na), _find_all(head, nb)
    if bool(ha) != bool(hb):
        return ('A' if ha else 'B'), 'head'

    # 3) 마지막 300자에서 더 뒤에 나온 팀
    tail = text[-300:]
    ta, tb = _find_all(tail, na), _find_all(tail, nb)
    if ta and tb:
        return ('A' if ta[-1] > tb[-1] else 'B'), 'tail'
    if ta:
        return 'A', 'tail'
    if tb:
        return 'B', 'tail'

    return 'UNCLEAR', 'none'