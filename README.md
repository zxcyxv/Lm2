# Lm2

`Lm2`는 기존 `Lm` 저장소에서 **MSE+CE 결합 이후의 latent-orbit
계보만** 분리한 재현용 저장소다. 과거의 독립 단발 실험, checkpoint,
토큰화된 데이터와 전체 연구 wiki는 포함하지 않는다.

## 현재 모델

현재 실험은 한 실제 prefix를 한 번만 인코딩하고 다음 계산을 공유한다.

- clean `K h_A ... K^4 h_A`
- learned sigma tape
- inverse decoder의 prefix FFN, attention, QKV/KV

그 뒤 독립 Gaussian noise로 세 개의 4-token trajectory만 분기한다.
각 trajectory의 점수는 정답 4토큰 CE 합의 음수이며,
`tau=0.05` detached responsibility로 CE를 가중한다. 현재 run에는
hidden-state MSE가 없다.

```text
h_A ── shared K orbit/sigma ──┬─ noisy trajectory 1 ─┐
                              ├─ noisy trajectory 2 ─┼─ detached CE selection
                              └─ noisy trajectory 3 ─┘
```

자세한 계보는 [docs/lineage.md](docs/lineage.md), 현재 사전등록은
[shared tau-0.05 manifest](experiments/records/EXP-20260724-k4-ha-skew-stride1-batch64-three-trajectory-detached-ce-tau005-shared-13m/manifest.md)에 있다.

## 설치와 검증

```bash
python -m pip install -e '.[test,data]'
pytest
```

WikiText-103 토큰화:

```bash
python prepare_wikitext103_bpe.py
```

현재 1000-step 실험:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_k4_ha_skew_three_trajectory_detached_ce_tau005_shared_13m.py \
  --steps 1000 --batch 64 --microbatch 16 --schedule-steps 6000
```

데이터는 `data/wikitext103/`, checkpoint와 생성물은 `outputs/`에
생성되며 Git에는 포함되지 않는다.
