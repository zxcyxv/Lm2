# Lm2

`Lm2`는 기존 `Lm` 저장소에서 **MSE+CE 결합 이후의 latent-orbit
계보만** 분리한 재현용 저장소다. 과거의 독립 단발 실험, checkpoint와
전체 연구 wiki는 포함하지 않는다.

## 저장소 구조 (2026-08-13 재정리)

코드가 root에 평평하게 수백 개 쌓여 우선순위를 알기 어려웠던 문제를
정리하기 위해 다음과 같이 재배치했다.

```text
src/rotlm/            현재 계보가 실제로 쓰는 core 아키텍처 + 평가 코드
tests/                위 core 코드를 커버하는 테스트만 보존
experiments/scripts/  byte256 계보의 실행 진입점 (train_/eval_/analyze_/
                       audit_/benchmark_/replay_)
experiments/records/  각 실험의 manifest/메트릭 (git 추적)
docs/                 계보·설계 문서 (낡은 "current architecture" 스냅샷
                       문서는 trash/docs로 옮기고 제외). 2026-08-07
                       이론 문서들이 근거로 삼는 원문 대화 로그
                       (제미나이대화내용.txt 등)도 여기 함께 둔다 —
                       개인 메모가 아니라 인용 대상이다.
outputs/               체크포인트/생성물. 2026-08-01 이전 checkpoint는
                       삭제했다 (Git에는 애초에 포함되지 않음)
trash/                 낮은 차원(width 896) 시절 계보 — query_k1, ha_skew,
                       k1/k3/k4 trajectory, wikitext2 실험과 그 시절
                       전용 모델/테스트 코드. 재현 가치는 있지만 지금
                       계보가 참조하지 않는다.
미확인/personal_notes/  코덱스 세션 백업/복원 노트와 세션 tarball.
                       코드도 인용 대상도 아니다. 판단 보류.
```

`experiments/scripts/`의 각 파일은 대부분 바로 이전 스크립트를
`import ... as base`로 재사용하고 몇 개 상수만 override하는 방식으로
계보를 이어간다. 그래서 이 디렉터리는 서로의 형제 import가 깨지지
않도록 **평평하게** 유지한다 (타입별 하위 폴더로 더 쪼개지 않음).

## 현재 계보와 미해결 이슈

### 관측된 증상: 13M 대비 121M

2026-08-01 byte-level(`byte256`) 학습으로 전환하며 width 896 → 1344
(13M)로 올렸다. 이 스케일에서는 아래 지표가 안정적이었다.
2026-08-06 width 1344 → 4352(121M)로 올린 뒤 아래 증상이 나타났다.
manifest 기준으로 recurrence, loss, seed, split, label 수,
head/key/value 차원, encoder 깊이는 고정하고 **width만** 바꾼
비교다.

두 스케일의 실측 기록 (`experiments/records/.../metrics.tsv`):

| step | 13M H1 NLL | 121M H1 NLL | 13M block(H16) NLL | 121M block(H16) NLL |
|---:|---:|---:|---:|---:|
| 1000 | 1.712 | 2.092 | 3.406 | 5.873 |
| 3357 | 1.498 | 1.741 | 3.605 | 7.756 |
| 6714 | 1.448 | 1.579 | 3.617 | **8.571** |

기록된 증상은 두 가지다.

**(1) LR 2e-4 시도에서 train CE와 block NLL이 동시에 상승.** 121M 첫
시도는 warmup이 peak LR에 도달하는 step 500에서 train CE `3.084 →
3.162`, block NLL `4.160 → 5.722`, raw gradient norm `12.29 → 14.15`를
기록했다. 13M 부모는 같은 구간에서 단조 하강했다. 이 시도는 사용자
지시로 중단해 `EXP-20260806-.../lr2e-4_attempt/`에 보존했고 재개하지
않았다.

**(2) LR을 낮춘 뒤에도 H5--H16 monitor만 상승.** 5e-5 / 최종 1e-4
재시도에서는 학습 대상인 H1--H4 CE는 계속 개선되는 반면, loss
gradient가 닿지 않는 H5--H16 monitor(block NLL)는 위 표처럼 계속
상승한다. 13M은 같은 지표가 3.4~3.98 범위에 머문다.

단, 121M manifest는 H5--H16의 지위를 이렇게 못박아 두었다:
"H5--H16 are explicitly out-of-objective transfer diagnostics. Their
degradation alone must not stop or alter the H4 training run."
(실제로 1e-4 시도가 H5--H16 악화를 근거로 잘못 중단됐다가 step-1,000
체크포인트에서 그대로 재개된 이력이 있다.) 즉 (2)는 등록된 실패
기준이 아니라 관측된 monitor 거동이다.

### 이후 측정한 것 (원인 판정 아님)

2026-08-07 memory-norm audit
(`EXP-20260807-byte256-unitary-feedback-121m-step13428-memory-norm-audit`)
이 121M step-13,428 체크포인트를 H16까지 굴려 복소 메모리 `S`의
Frobenius norm을 측정했다. 측정값만 옮기면:

- H1 평균 `21,027.31` → H16 평균 `186,328.77` (`8.861x`)
- H16에서 `‖S‖/sqrt(Σ_r‖W_r‖²)` 평균 `3.376977`, 범위
  `[2.806558, 3.546877]`
- H2--H16 구간에서 cross term이 음수였던 trajectory-step `0.00%`,
  총 메모리 에너지가 감소한 trajectory-step `0.00%`

해당 audit의 `interpretation.md`는 스스로 범위를 이렇게 한정한다:
"This is a forward-state audit. It does not infer causality for loss or
gradient behavior." 즉 위 수치는 **forward 상태에 대한 관측이며, 위
증상의 원인으로 확정된 것이 아니다.**

별도의 심볼릭 분석이
[docs/recurrent_gradient_path_analysis.md](docs/recurrent_gradient_path_analysis.md)에
있다. 이것도 코드 수준의 미분 경로 분석이지 실행 증거가 아니다.

2026-08-07에 실행된 후속 스크립트들(`experiments/scripts/`):
detached-input raw-read postnorm, step-100 gradient localization,
central VJP 분석, full-BPTT joint postnorm 비교.
**원인은 아직 확정되지 않았고 문제도 해결되지 않았다.**

가장 최근 실행 진입점은

```text
experiments/scripts/train_byte256_unitary_full_bptt_raw_read_joint_postnorm_h4_stride4_h16_monitor_10epoch_13m.py
```

계보 전체 서사는 [docs/lineage.md](docs/lineage.md)에 있으나,
`docs/lineage.md`는 2026-08-03 시점(H16 boundary recurrence / fused
scan)까지만 갱신돼 있다. **width-4352 스케일업과 그 이후 붕괴/감사
구간은 lineage.md에 없으므로** `experiments/records/`의 개별
manifest(특히 위에 인용한 두 EXP)와 `experiments/scripts/`의
2026-08-06~07 파일을 직접 참고해야 한다.

## 설치와 검증

```bash
python -m pip install -e '.[test,data]'
pytest
```

## 데이터

현재 계보는 byte-level 데이터(`data/wikitext103_bytes/`, 설명은
[data/wikitext103_bytes/README.md](data/wikitext103_bytes/README.md))를
쓴다. 준비 스크립트는

```bash
python experiments/scripts/prepare_wikitext103_bytes.py
```

옛 BPE 토큰화 wikitext103(`data/wikitext103/`)는 width-896 시절
계보가 쓰던 데이터로, 해당 준비 스크립트는 `trash/`에 있다.

## 실행 예시

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python experiments/scripts/train_byte256_unitary_full_bptt_raw_read_joint_postnorm_h4_stride4_h16_monitor_10epoch_13m.py
```

각 스크립트는 실행 시 cwd가 저장소 root라고 가정한다
(`experiments/records/...`, `outputs/experiments/...` 같은 상대
경로를 그대로 사용하기 때문). checkpoint와 생성물은 `outputs/`에
생성되며 Git에는 포함되지 않는다.
