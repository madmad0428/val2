import anthropic
client = anthropic.Anthropic(api_key=" ")

fake_prompt = """이제 아래 경기를 예측해.

[대회] VCT 2026: Americas Stage 1
[경기] Sentinels vs KRU Esports
[맵] Lotus

팀A: Sentinels
- 평균 ACS: 214.4 | 평균 KD: 4.0 | 평균 ADR: 137.2
- 선제킬 획득: 12회 | 선제킬 허용: 9회
- 풀바이 승률 80% | 멀티킬 35회

팀B: KRU Esports
- 평균 ACS: 170.4 | 평균 KD: -4.0 | 평균 ADR: 112.4
- 선제킬 획득: 9회 | 선제킬 허용: 12회
- 풀바이 승률 20% | 멀티킬 10회

반드시 마지막 줄에 아래 형식으로만 출력해.
{"winner": "Sentinels" 또는 "KRU Esports"}"""

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
    messages=[{"role": "user", "content": fake_prompt}]
)
print("실제 승자: KRU Esports")
print("변조 스탯: Sentinels가 압도적 우세")
print()
print(response.content[0].text)

