# 2026-08-15 — 표준 계열과 어긋나 있던 세 지점

이 문서는 하루치 실험 중 **표준 선형어텐션 / 표준 트랜스포머와 이 구현이
어긋나 있던 부분**만 기록한다. 그날 시도한 나머지 변형(읽기정규화 제거,
메모리 감쇠, S post-normalization, postnorm 재귀, RoPE hidden phase 초기화,
인코더 fan-in 초기화, 8칸 요인설계)은 곁가지였고, 아래 세 번째 항목이
밝혀진 시점에서 **그 측정값들은 전부 다시 재야 한다**. 이유는 마지막 절에
적는다.

측정은 전부 byte-level `data/wikitext103_bytes`, context 256,
anchor stride 16, horizon 16, 균등 CE, seed 1337, effective batch 64,
`clip_norm=1.0`, AdamW(0.9, 0.95), warmup 100, cosine schedule 6000이다.
별도 표기가 없으면 step 1000 시점 값이다.

---

## 1. MVA (multi-value attention) 레이아웃

**표준**: Mamba 계열은 B/C(= key/query)를 전 head가 공유하고 value만 head별로
둔다. 이 저장소는 Q/K를 head마다 따로 투영하고 있었다.

**변경**: `shared_kv_heads=True`. 레이어당 Q/K 투영 비용이
`width x heads x key_dim` → `width x key_dim`으로 줄어든다.

**측정** (121M, 병목 있는 구성, seed/데이터순서/스케줄 동일):

| 구성 | blk | h1 | h16 | gnorm | phase가 차지한 g² |
|---|---:|---:|---:|---:|---:|
| per-head Q/K | 3.1590 | 2.5076 | 3.1970 | 10.05 | — |
| MVA (공유 Q/K) | 3.1728 | 2.5250 | 3.2071 | 12.55 | 82.1% |

손실은 사실상 동일하고(+0.014) 파라미터만 121,336,064 → 119,386,368으로
줄었다. **레이아웃 변경 자체는 손실 중립이며, 이후 모든 실험은 MVA로 진행했다.**

주의: 이 시점 기록된 `phase가 차지한 g²`(82.1%)는 3번 항목의 결함 아래에서
측정된 값이다.

---

## 2. read bottleneck 제거

**표준**: multi-head attention은 `n_heads x head_dim = d_model`로 묶여 있어
어텐션 출력이 잠재 전체 폭을 채운다.

**어긋나 있던 것**: 이 구현의 `read_width = 2 x heads x value_dim`은
`width`와 무관하게 **496 고정**이었다. width를 올릴수록 중앙 recurrence가
잠재의 좁은 부분공간에만 쓰게 된다.

| width | read_width | 비율 |
|---:|---:|---:|
| 1344 (13M) | 496 | 37% |
| 4352 (121M) | 496 | **11%** |

즉 width 스케일업이 병목을 3.4배 조이고 있었다.

**변경**: `value_dim = width / (2 x heads)`로 두어 `read_width == width`를 강제.

**측정**:

| 구성 | width | blk | h1 | h16 | gnorm |
|---|---:|---:|---:|---:|---:|
| 121M 병목 O (per-head) | 4352 | 3.1590 | 2.5076 | 3.1970 | 10.05 |
| 121M 병목 X (MVA) | 3872 | 3.1608 | 2.4691 | 3.1852 | 10.62 |
| 13M 병목 X, heads 8 | 1264 | 3.0527 | 2.0273 | 3.1428 | 0.99 |
| 13M 병목 X, heads 16 | 1248 | 3.0688 | 2.1156 | 3.1391 | 0.71 |
| 13M 병목 X, heads 32 | 1216 | 3.0637 | 2.0721 | 3.1454 | 1.17 |

**병목 제거만으로는 손실이 움직이지 않았다** (121M 3.1608 vs 3.1590).
13M에서 head 수를 4~32로 스윕해도 3.053~3.069로 무차별했다.

파라미터 수를 맞춘 상태에서 head/value 배분은 이 규모에서 판별력이 없다.
다만 병목 제거는 표준 정합성 자체로 유지할 가치가 있고, 이후 모든 실험이
`read_width == width`로 진행됐다. 13M 병목 제거 구성은 6000스텝 완주 시
blk **2.9190**을 기록했다.

---

## 3. lm_head — 최종 정규화가 없었다

**이 문서에서 유일하게 손실을 크게 움직인 항목이며, 나머지 측정을 무효화한다.**

**표준**: 모든 트랜스포머는 unembedding 앞에 최종 정규화(`ln_f`)를 둔다.
GPT-2, LLaMA 예외 없음. weight tying 여부와는 무관한 별개의 구성요소다.

**어긋나 있던 것**: `K1DecoderAblationLM.token_logits`의
`head_mode="simplex-raw-tied"`가

```python
logits = 16.0 * hidden @ E.T
```

로 **정규화되지 않은 hidden에 고정 배율 16을 곱한다.** 같은 함수의 다른 두
모드는 정규화가 있다 — `simplex-tied`는 `F.normalize`(코사인, ±16 유계),
fallback은 `head_norm`. 쓰이던 모드만 유계화가 없었다.

`hidden`은 가역 인코더를 역으로 통과시킨 결과이고, 디코더가 받는 tape는
인코더가 실제로 만든 적 없는 **예측 상태**라 항상 off-manifold다. 그 증폭된
크기가 정규화 없이 곧장 softmax로 들어간다.

**초기화 시점 측정** (`ln(256) = 5.5452`가 무작위 예측의 CE):

| width | head | logit RMS | logit max | 초기 CE | 초기 gnorm |
|---:|---|---:|---:|---:|---:|
| 1264 | `simplex-raw-tied` | 15.71 | 60.05 | **41.83** | 128.7 |
| 3872 | `simplex-raw-tied` | 15.98 | 69.87 | **44.57** | **675** |
| 1264 | 코사인 | 0.43 | 1.65 | 5.59 | — |
| 1264 | `head_norm @ E.T` | 0.95 | 3.68 | 5.88 | — |
| 3872 | 코사인 | 0.19 | 0.73 | 5.56 | — |
| 3872 | `head_norm @ E.T` | 0.72 | 2.82 | 5.79 | — |

**아무것도 모르는 초기 모델이 무작위보다 8배 나쁜 점수에서 출발하고 있었다.**

**변경**: `train_postnorm_recurrence.py --head standard` — 학습 gain을 가진
RMSNorm 뒤에 tied unembedding, 고정 배율 없음 (`TiedLMHead`).

**측정**:

| 구성 | 초기 blk | 초기 gnorm | step 100 (warmup peak) | blk@1000 | h1 | gnorm |
|---|---:|---:|---|---:|---:|---:|
| 121M 기존 head | 44.86 | 675 | **12.9~16.3 (발산)** | — | — | — |
| 121M 표준 head, lr 1e-4 | 5.82 | 12.2 | 3.1413 (하강) | **3.0066** | **1.7917** | 0.28 |
| 13M 표준 head, lr 3e-4 | 5.85 | 19.1 | 3.1721 (하강) | **3.0213** | 1.9075 | 0.34 |

세 가지가 동시에 해결됐다.

1. **warmup이 peak LR에 닿는 지점의 붕괴가 사라졌다.** 그 전까지 121M 런은
   예외 없이 step 100에서 무너졌다 (blk 12.9~16.3, gnorm 300~380). 이 저장소
   `README.md`가 기록한 최초 붕괴(2e-4 시도, step 500에서 CE와 block NLL 동반
   상승)와 같은 서명이다.
2. **gnorm이 clip 아래로 내려왔다.** 0.28~0.34. 그 전까지 121M은 전 구간
   7~20에서 내려오지 않아 매 스텝 clip에 걸려 있었다.
3. **width 스케일링이 처음으로 성립했다.**

| | 13M | 121M | |
|---|---:|---:|---|
| 기존 계보 (step 1000) | 3.0537 | 3.1590 | **역전** |
| 표준 head | 3.0213 | **3.0066** | 정상 |

121M이 **LR 3배 불리한 조건**(1e-4 vs 3e-4)에서 이겼다. 양쪽 모두 val NLL
반등 0회.

### 순전파 단계별 width 스케일 계측

문제 위치를 특정한 근거. 초기화 시점, width 1264 vs 3872 배율:

| 단계 | 배율 | |
|---|---:|---|
| 인코더 출력 / root z0 | **3.41** | `shift_lm.RevBlock`이 fan_in과 무관하게 `std=0.02` 고정 |
| QKV prenorm 이후 | 1.00 | |
| q / k / v | ~1.00 | |
| 메모리 S | 1.04 | |
| read | 1.00 | |
| innovation δ | **1.00** | |
| 잠재 z | 1.01 | |
| postnorm | 1.00 | |
| 역변환 디코더 출력 | **1.31** | |
| logit max | 60.05 → 69.87 | |

**prenorm 이후 디코더 입력까지 전 구간이 width 중립이다.** 그날 파고든
읽기정규화·감쇠·prenorm 위치·θ 초기화·병목·MVA는 전부 이 배율 1.00 구간을
건드리고 있었다. 스케일링 열화는 그 바깥에 있었다.

인코더의 `std=0.02` 고정도 별도 결함이다. `1/sqrt(fan_in)`으로 바꾸면 위
3.41배가 1.02배로 평평해진다. 다만 그것만으로 121M 학습이 좋아지지는 않았고
(blk 4.4949, gnorm 271), 손실을 실제로 고친 것은 head 쪽이었다.

---

## 4. 121M에 적절한 토큰 규모

현재 데이터셋과 실행 조건:

```
train split          =  55,000,000 bytes  (0.055B)
1000 step x 64 x 256 =  16,384,000 bytes  = 0.30 epoch
```

**위 실험들은 1 epoch도 돌지 않았다.** train split의 30%만 봤다.

Chinchilla 기준(대략 20 tokens/param):

| | 최적 토큰 | 현재 런(0.016B) 대비 | 10 epoch(0.55B) 대비 |
|---|---:|---:|---:|
| 13M | ~2.6억 | **1/16** | 2.1배 초과 |
| 121M | ~24억 | **1/150** | **1/4.4** |

- **10 epoch = 0.55B로는 121M에 부족하다.** 24억에 닿으려면 약 **44 epoch**,
  현재 batch 기준 **146,000 step**이 필요하다.
- 13M은 10 epoch면 최적점을 두 배 넘긴다.
- 다만 55M 바이트를 44회 반복하는 것은 24억 토큰과 다르다. 같은 데이터를
  반복하면 새 정보가 들어오지 않아 파라미터를 못 쓴다. **데이터셋 크기 자체가
  121M의 상한을 정한다.**

따라서 121M의 용량을 제대로 보려면 **스텝을 늘리는 것보다 데이터를 늘리는
것이 먼저다.**

### 평가 지표에 대한 주의

`blk`은 16개 horizon의 단순 평균인데, 뒤쪽 horizon은 바닥이 높아 모델 크기로
내려가지 않는다. 이 세션 전 구성에서 `h16`은 3.13~3.17에 머물렀다.

| | 13M | 121M | 차이 |
|---|---:|---:|---:|
| h1 | 1.9075 | 1.7917 | **−0.1158** |
| h16 | 3.1415 | 3.1297 | −0.0118 |
| blk | 3.0213 | 3.0066 | −0.0147 |

**h16이 평균을 지배해 blk 차이가 h1 차이의 1/8로 눌린다.** 스케일 효과를 보려면
horizon별로 보아야 한다.

---

## 5. 이 문서 밖의 측정에 대한 경고

3번이 밝혀지기 전에 낸 **손실 기반 순위는 전부 무효로 봐야 한다.** 초기 CE
41.8~44.6, 초기 gnorm 128~675 상태에서 최적화가 출발했고, 순위가 "그 병리를
누가 잘 흡수하느냐"로 결정됐을 가능성이 있다.

다시 재야 하는 것:

- 8칸 요인설계(읽기정규화 x QKV prenorm x root normalization)와 "한 모서리만
  유독 좋다"는 결론
- "h1이 순위를 지배하고 h16은 전 칸 동일"이라는 관측
- RoPE hidden-phase 초기화가 구조에 따라 부호가 뒤집힌다는 관측
- 메모리 감쇠 / S post-normalization / read-output normalization 평가
- step 1000 노이즈 바닥 추정치(±0.005)

이미 뒤집힌 사례: 8칸 1등이던 frobenius 읽기정규화 구성 3.0546 대비,
읽기정규화 **없는** 구성 + 표준 head가 **3.0213**. "읽기정규화가 필수처럼
보인다"던 그림이 head를 고치자 반대로 간다.

head와 무관하게 살아남는 것 (구조·수학 검증):

- 새 트레이너가 predecessor를 bit-exact 재현 (read-state tape `0.000e+00`)
- 병렬 prefix scan = 순차 재귀 (`0.00e+00`), 정의식 이중합 대조 (`8.9e-16`)
- 메모리 감쇠 연산자의 우선순위·크기 검증
- 위 4절의 순전파 단계별 width 스케일 계측

## 재현

```bash
# 13M
python train_postnorm_recurrence.py --steps 1000 --schedule-steps 6000 \
  --microbatch 16 --qkv-prenorm --no-root-norm \
  --memory-decay --read-normalization output --head standard \
  --width 1264 --peak-lr 3e-4

# 121M
python train_postnorm_recurrence.py --steps 1000 --schedule-steps 6000 \
  --microbatch 16 --qkv-prenorm --no-root-norm \
  --memory-decay --read-normalization output --head standard \
  --width 3872 --peak-lr 1e-4

# 기존 구조 bit-exact 재현
python train_postnorm_recurrence.py --read-normalization frobenius \
  --qkv-prenorm --no-root-norm
```

기록: `outputs/head_13m/`, `outputs/head_121m/`, `outputs/head_121m_lr3e4/`
