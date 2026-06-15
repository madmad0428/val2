import os
import time
import pandas as pd
import anthropic
from datetime import datetime

# ── 설정 ────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # 환경변수에서 읽기
MODEL = "claude-haiku-4-5-20251001"            # 저렴한 Haiku 사용
MAX_TOKENS = 512
TEMPERATURE = 0                                # 재현성을 위해 0 고정
DELAY = 1.0                                    # API 호출 간격 (초)
DATA_PATH = "dataset_final.csv"               # 데이터셋 경로
SAVE_PATH = "experiment_results.csv"          # 결과 저장 경로
# ────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=API_KEY)

def call_claude(prompt: str) -> str:
    """Claude API 호출 후 응답 텍스트 반환"""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  ⚠️  API 오류: {e}")
        return "ERROR"

def extract_winner(response: str, team_a: str, team_b: str) -> str:
    """응답에서 팀A 또는 팀B 추출"""
    resp = response.strip()
    if team_a.lower() in resp.lower():
        return team_a
    elif team_b.lower() in resp.lower():
        return team_b
    else:
        # 팀명이 없으면 응답 앞 50자 그대로 반환 (수동 확인용)
        return f"UNCLEAR: {resp[:50]}"

def is_correct(predicted: str, winner: str) -> int:
    """정답 여부 반환 (1=정답, 0=오답, -1=불명확)"""
    if predicted.startswith("UNCLEAR") or predicted == "ERROR":
        return -1
    return 1 if predicted == winner else 0

def run_experiment():
    df = pd.read_csv(DATA_PATH)
    total = len(df)

    # 이미 결과가 있는 행은 건너뛰기 (중단 후 재시작 지원)
    start_idx = 0
    if os.path.exists(SAVE_PATH):
        done = pd.read_csv(SAVE_PATH)
        start_idx = len(done)
        print(f"이전 진행 감지: {start_idx}개 완료, {total - start_idx}개 남음\n")
        df = df.iloc[start_idx:].reset_index(drop=True)

    print(f"{'='*55}")
    print(f"  VCT 프롬프트 실험 시작")
    print(f"  모델: {MODEL} | Temperature: {TEMPERATURE}")
    print(f"  총 시나리오: {total}개 | 시작: {start_idx+1}번째부터")
    print(f"  예상 API 호출 수: {len(df) * 3}회")
    print(f"{'='*55}\n")

    results = []

    for i, row in df.iterrows():
        sid = row['Scenario ID']
        team_a = row['Team A']
        team_b = row['Team B']
        winner = row['Winner']
        actual_num = start_idx + i + 1

        print(f"[{actual_num:03d}/{total}] {sid} | {row['Match Name']} | {row['Map']}")

        # ── Few-shot ──
        resp_fs = call_claude(row['Prompt_FewShot'])
        pred_fs = extract_winner(resp_fs, team_a, team_b)
        corr_fs = is_correct(pred_fs, winner)
        time.sleep(DELAY)

        # ── CoT ──
        resp_cot = call_claude(row['Prompt_CoT'])
        pred_cot = extract_winner(resp_cot, team_a, team_b)
        corr_cot = is_correct(pred_cot, winner)
        time.sleep(DELAY)

        # ── Role ──
        resp_role = call_claude(row['Prompt_Role'])
        pred_role = extract_winner(resp_role, team_a, team_b)
        corr_role = is_correct(pred_role, winner)
        time.sleep(DELAY)

        # 결과 출력
        fs_mark  = "✅" if corr_fs==1  else ("❓" if corr_fs==-1  else "❌")
        cot_mark = "✅" if corr_cot==1 else ("❓" if corr_cot==-1 else "❌")
        rol_mark = "✅" if corr_role==1 else ("❓" if corr_role==-1 else "❌")
        print(f"       실제 승자: {winner}")
        print(f"       Few-shot {fs_mark} {pred_fs}")
        print(f"       CoT      {cot_mark} {pred_cot}")
        print(f"       Role     {rol_mark} {pred_role}\n")

        results.append({
            **row.to_dict(),
            'Result_FewShot': pred_fs,
            'Result_CoT': pred_cot,
            'Result_Role': pred_role,
            'Raw_FewShot': resp_fs,
            'Raw_CoT': resp_cot,
            'Raw_Role': resp_role,
            'Correct_FewShot': corr_fs,
            'Correct_CoT': corr_cot,
            'Correct_Role': corr_role,
        })

        # 10개마다 중간 저장 (중단 대비)
        if (i + 1) % 10 == 0:
            save_results(results, start_idx, SAVE_PATH)
            print(f"  💾 중간 저장 완료 ({actual_num}/{total})\n")

    # 최종 저장
    save_results(results, start_idx, SAVE_PATH)
    print_summary(SAVE_PATH)

def save_results(results: list, start_idx: int, path: str):
    """기존 결과에 이어서 저장"""
    new_df = pd.DataFrame(results)
    if start_idx > 0 and os.path.exists(path):
        old_df = pd.read_csv(path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined.to_csv(path, index=False)
    else:
        new_df.to_csv(path, index=False)

def print_summary(path: str):
    """최종 정확도 요약 출력"""
    df = pd.read_csv(path)
    total = len(df)

    print(f"\n{'='*55}")
    print(f"  실험 완료 — 최종 결과 요약")
    print(f"{'='*55}")

    for method, col in [("Few-shot", "Correct_FewShot"),
                         ("CoT",      "Correct_CoT"),
                         ("Role",     "Correct_Role")]:
        valid = df[df[col] != -1]
        correct = (valid[col] == 1).sum()
        unclear = (df[col] == -1).sum()
        acc = correct / len(valid) * 100 if len(valid) > 0 else 0
        print(f"  {method:<12} {correct}/{len(valid)} 정답 → {acc:.1f}%  (불명확 {unclear}개)")

    print(f"\n  결과 파일: {path}")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    if not API_KEY:
        print("❌ ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   터미널에서 아래 명령어 실행 후 다시 시도하세요:")
        print("   Windows: set ANTHROPIC_API_KEY=sk-ant-...")
        print("   Mac/Linux: export ANTHROPIC_API_KEY=sk-ant-...")
    else:
        run_experiment()
