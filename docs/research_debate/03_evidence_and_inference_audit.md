# 실험 증거와 추론 경로 감사

## 1. 문서의 범위

이 문서는 설계 논쟁에 사용된 현재 증거를 한곳에 모으되, 수치와
해석을 같은 것으로 취급하지 않는다.

- 수치는 기존 experiment record의 TSV를 그대로 우선한다.
- 이 문서는 수치를 복제한 해석용 색인이다.
- test split은 사용하지 않았다.
- checkpoint 하나, seed 하나, 짧은 1000-update 학습에서 얻은
  관측을 보편 명제로 확대하지 않는다.
- 원 manifest와 후속 code-path audit가 충돌하는 경우 둘 다 보존하고
  후속 범위 정정을 표시한다.

주요 원자료는 다음과 같다.

- [initial-condition manifest](../../experiments/records/EXP-20260725-k4-ha-skew-stride1-batch64-micro16-initial-condition-three-trajectory-prior-tau005-h1-online-mse-13m/manifest.md)
- [training metrics](../../experiments/records/EXP-20260725-k4-ha-skew-stride1-batch64-micro16-initial-condition-three-trajectory-prior-tau005-h1-online-mse-13m/metrics.tsv)
- [block-4 generation manifest](../../experiments/records/EXP-20260725-k4-ha-skew-stride1-batch64-micro16-initial-condition-three-trajectory-prior-tau005-h1-online-mse-13m/step1000_block4_generation/manifest.md)
- [block-4 generation metrics](../../experiments/records/EXP-20260725-k4-ha-skew-stride1-batch64-micro16-initial-condition-three-trajectory-prior-tau005-h1-online-mse-13m/step1000_block4_generation/metrics.tsv)
- [matched GPT-2 analysis](../../experiments/records/EXP-20260725-k4-ha-skew-stride1-batch64-micro16-initial-condition-three-trajectory-prior-tau005-h1-online-mse-13m/step1000_gpt2_ar_sentence_comparison/analysis.md)

## 2. 현재 step-1000 실험이 실제로 학습한 것

실험은 다음 initial-condition trajectory를 사용했다.

\[
\eta_i=R(\operatorname{sg}(h_A),z_i),
\qquad
u_{i,1}=Kh_A+\eta_i,
\qquad
u_{i,j}=K^{j-1}u_{i,1}.
\]

prefix prior \(\pi_i\)와 네 token path cost

\[
C_i=\sum_{j=1}^{4}\operatorname{CE}_{i,j}
\]

에서 detached posterior responsibility를 만들었다.

\[
q_i=\operatorname{sg}
\operatorname{softmax}(\log\pi_i-C_i/0.05).
\]

전체 loss는 다음 세 항이다.

- posterior-weighted four-token trajectory CE
- clean \(Kh_A\to h_B\) attached online relative MSE
- \(q\)와 prefix prior 사이 KL

### 구현 사실: 없는 loss

- clean `K hA` logits에 대한 별도 AR CE
- branch state의 h1--h4 latent MSE
- horizon별 새 innovation
- canonicalizer \(P\)
- simplex mapping
- dynamic or context-generated \(K_A\)

따라서 이 checkpoint를 “표준 AR MLE 경로를 block으로 근사하도록
직접 학습한 모델”이라고 부르면 안 된다. 정확한 표현은
“four-token branch CE와 clean h1 latent regression을 함께 학습한
initial-condition mixture”다.

## 3. step-1000 optimization evidence

### 요약

| 지표 | step 1000 |
|---|---:|
| validation marginal NLL | 5.638737 |
| validation posterior NLL | 5.625551 |
| validation oracle NLL | 5.625516 |
| validation prior-selected NLL | 6.847853 |
| prior winner accuracy | 0.428070 |
| responsibility max | 0.997774 |
| effective trajectories | 1.006825 |
| mean residual sigma | 0.087781 |
| operator orthogonality error | \(1.39\times10^{-5}\) |
| branch-distance preservation error | \(2.24\times10^{-6}\) |

posterior가 거의 한 branch에 집중했고, prior winner accuracy는 random
chance `1/3`보다 높았다. branch distance와 operator orthogonality도
사전등록 tolerance 안에 있었다.

### 중간 checkpoint 흐름

| step | validation marginal NLL | prior winner accuracy | clean h1 relative MSE |
|---:|---:|---:|---:|
| 100 | 6.834341 | 0.349915 | 0.016925 |
| 250 | 6.305680 | 0.414856 | 0.016869 |
| 500 | 5.949849 | 0.421448 | 0.013870 |
| 750 | 5.773024 | 0.422821 | 0.013740 |
| 1000 | 5.638737 | 0.428070 | 0.013768 |

대화 중 step 250의 `prior_win≈0.415`를 보고 조기 결론보다 학습을
계속 관찰하기로 한 판단은 이후 기록과 일치했다. prior agreement와
marginal NLL은 1000까지 개선됐고 clean h1 state alignment도
유지됐다. 다만 이 optimization 지표 개선이 free-generation
coherence를 보장하지는 않았다.

### horizon별 token accuracy

| selector | h1 | h2 | h3 | h4 |
|---|---:|---:|---:|---:|
| posterior | 0.243349 | 0.159780 | 0.114023 | 0.095669 |
| oracle per-token | 0.243347 | 0.159698 | 0.114075 | 0.095673 |
| prior | 0.183807 | 0.089569 | 0.054718 | 0.057922 |

낮은 posterior temperature로 responsibility가 거의 한 branch에
집중해 posterior와 oracle 집계가 매우 가까웠다. prior-selected
accuracy는 horizon이 멀어질수록 크게 낮아졌다.

### latent relative MSE

| state | h1 | h2 | h3 | h4 |
|---|---:|---:|---:|---:|
| branch trajectory | 0.021528 | 0.031805 | 0.042578 | 0.052504 |
| clean \(K^j h_A\) | 0.013768 | 0.024135 | 0.034933 | 0.044886 |

orthogonal \(K\)가 spectral explosion을 막았지만 error는 horizon과 함께
증가했다. branch states는 clean states보다 gold encoder state에서
더 멀었다. 그러나 branch CE는 더 좋은 token mode를 찾을 수 있으므로
MSE만으로 branch usefulness를 판단할 수 없다.

## 4. 기존 control과의 비교

| experiment | best step | validation marginal NLL |
|---|---:|---:|
| independent-noise CE-only | 1000 | 5.787555 |
| independent-noise + clean h1 MSE | 1000 | 5.884858 |
| learned initial condition + prior + clean h1 MSE | 1000 | 5.638737 |

현재 initial-condition run은 두 기록 control보다 낮은 marginal NLL을
보였다.

### 이 비교가 지지하는 것

- 현재 목적함수와 parameterization의 결합은 기록된 control보다
  step-1000 marginal NLL이 좋았다.
- learned initial residual이 0으로 collapse하지 않았다.
- prefix prior가 chance 이상의 winner agreement를 학습했다.
- branch geometry가 \(K\) 아래 안정적으로 유지됐다.

### 이 비교가 분리하지 못하는 것

initial-condition run은 control과 다음이 동시에 다르다.

- independent Gaussian 대신 learned branch code
- horizon마다 noise를 넣는 대신 initial residual 하나
- prefix prior와 prior KL 추가
- physical microbatch 16 대 32
- stochastic validation tape 제거

따라서 수치만으로 “후속 innovation이 없어야 한다”거나 “initial
condition이 유일한 원인”이라고 결론낼 수 없다. 특히 과거
compounding-noise control은 independent innovation과 다른 MSE
구성을 함께 사용했으므로 **correlated fixed-\(z\) tape를 검증한 적이
없다.**

## 5. matched generation protocol

64개의 고정 WikiText-103 validation prompt에 대해:

- prompt 64 tokens
- continuation 64 tokens
- greedy decoding
- 동일 tokenizer와 시작점
- latent model step 1000
- 비교용 standard GPT-2 step 1000

를 사용했다.

latent 정책은 다음 네 가지였다.

1. `clean_ar`
2. `prior_ar`
3. `clean_block4`
4. `prior_block4`

matched GPT-2는 13,033,680 parameters, latent model은 13,087,332
parameters였다. train/validation/tokenizer checksum은 두 저장소에서
일치했다.

그러나 동일 parameter count와 update count는 동일 supervision 또는
동일 compute를 뜻하지 않는다. GPT-2는 각 context position의 dense
next-token CE를 받고, latent model은 overlapping four-horizon branch
CE와 state/prior loss를 받았다.

## 6. 생성 결과

| policy | distinct-2 | immediate repeat | period-2 | period-4 | exact block repeat | collapsed | longest run |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 AR | 0.2768 | 0.0930 | 0.1923 | 0.2557 | 0.1167 | 0.2969 | 3.70 |
| clean AR | 0.1791 | 0.5801 | 0.5602 | 0.6435 | 0.5333 | 0.7031 | 33.92 |
| prior AR | 0.2634 | 0.3867 | 0.3727 | 0.3852 | 0.3333 | 0.4688 | 22.16 |
| clean block-4 | 0.2773 | 0.5511 | 0.4461 | 0.6617 | 0.4583 | 0.5781 | 22.16 |
| prior block-4 | 0.3380 | 0.2299 | 0.0640 | 0.4951 | 0.2687 | 0.1719 | 4.19 |

`collapsed`는 identical-token run 8 이상 또는 distinct-1 0.1 미만인
sample이다.

### GPT-2의 failure pattern

matched GPT-2도 1000 update에서 좋은 장문을 만들지 못했다.

- 초반에는 국소적으로 그럴듯한 clause를 만들기도 했다.
- 이후 WikiText heading, 고빈도 phrase, corpus-format template를
  반복했다.
- 동일 token fixed point보다 phrase/template attraction이
  두드러졌다.

따라서 “13M/1000-step에서 문장 품질이 낮다”는 사실 자체는 parallel
architecture의 반증이 아니다.

### clean latent orbit의 failure pattern

`clean_ar`는 64개 중 45개가 collapse heuristic에 걸렸고, median
longest identical run이 39 tokens였다. 매 token 재인코딩해도 생성
prefix가 좁은 decoder/token basin으로 들어가면 clean \(Kh_A\)가 같은
token attractor를 반복했다.

`clean_block4`는 distinct-2가 GPT-2와 비슷하지만 immediate repeat와
period-4 repeat가 매우 높다. aggregate diversity만으로 sentence
coherence를 판단할 수 없음을 보여 준다.

### initial-condition branch의 failure pattern

`prior_block4`는 동일 token collapse를 크게 줄였다.

- distinct-2: `0.3380`
- collapsed fraction: `0.1719`
- mean longest identical run: `4.19`

하지만 lag signature가 강한 4-position cycle을 보였다.

- lag-1: `0.2299`
- lag-2: `0.0640`
- lag-4: `0.4951`
- aligned exact block repeat: `0.2687`

block 내부 immediate repeat는 `0.2757`, 재인코딩 boundary repeat는
`0.0833`이었다. 네 token boundary가 언어적으로 의미가 없는데도
출력 통계에 강하게 드러났다.

현재 증거가 지지하는 가장 좁은 해석은 다음이다.

> initial residual은 horizon state를 기하학적으로 분리하여 동일
> token fixed point를 줄였지만, 그 분리가 semantic sequence보다
> 4-slot classification shortcut으로 사용됐다.

## 7. “AR보다 block-4가 왜 좋아 보이는가”에 대한 code-path 감사

### clean pair

`clean_ar`:

```text
encode actual prefix
take K h
decode one token
append and re-encode
```

`clean_block4`:

```text
encode actual prefix
take K h, K² h, K³ h, K⁴ h
decode four tokens
append and re-encode
```

두 정책은 clean orbit에 대해서는 주로 re-grounding interval이
다르다. 하지만 clean path는 h1 latent MSE만 직접 받고 clean logits의
별도 AR CE를 받지 않았다. 따라서 `clean_ar`는 표준 AR MLE policy가
아니다.

### prior pair

`prior_ar`:

```text
encode actual prefix
recompute prior
select branch
recompute eta
decode one token
append and repeat
```

`prior_block4`:

```text
encode actual prefix
compute prior once
select one branch
propagate the same initial condition for four horizons
decode four tokens
append and repeat
```

따라서 `prior_ar`는 매 token rebranching하고, `prior_block4`는 네
token 동안 trajectory를 유지한다.

### 보존해야 할 기록 충돌

block-4 generation manifest는 matched pair가 “re-grounding interval만
다른” 정책이라고 사전등록했다. 후속 code-path audit에서 prior
pair에는 branch selection과 residual reinjection frequency 차이도
있음이 확인됐다.

원 manifest는 당시 프로토콜의 기록이므로 수정하거나 삭제하지 않는다.
대신 결론의 범위를 다음처럼 정정한다.

> 기록된 `prior_ar` 대 `prior_block4` 결과는 AR re-anchoring의 순수
> 효과가 아니라 token-local rebranching과 block-level commitment의
> 효과까지 합친 비교다.

이 차이는 `prior_block4`가 `prior_ar`보다 나은 다양성 지표를 보인
것을 “open-loop가 AR보다 우월하다”는 증거로 사용할 수 없게 한다.

## 8. 속도 관측의 범위

64-token generation wall time은 다음이었다.

| policy | wall seconds | encoder calls/sample |
|---|---:|---:|
| clean AR | 2.684 | 64 |
| clean block-4 | 0.569 | 16 |
| prior AR | 2.512 | 64 |
| prior block-4 | 0.640 | 16 |

block-4는 이 작은 평가에서 약 4배의 encoder-call 감소와 비슷한
wall-time 개선을 보였다. 이는 원래 계산 이점이 실제 구현에서도
존재함을 보여 준다.

그러나 prior pair는 위에서 설명한 대로 동일 stochastic dynamics가
아니다. 속도 수치는 block commitment 정책의 end-to-end cost이며,
순수하게 re-anchoring만 제거한 matched dynamics cost로 해석하면
안 된다.

## 9. 증거 상태표

| 주장 | 상태 | 근거와 한계 |
|---|---|---|
| orthogonal \(K\)가 branch distance를 보존했다 | 관측 | step-1000 error \(2.24\times10^{-6}\) |
| initial residual이 완전히 collapse하지 않았다 | 관측 | sigma `0.0878`, trajectory diversity nonzero |
| prior가 chance 이상의 winner relation을 학습했다 | 관측 | `0.4281 > 1/3`; calibration 성공과는 다름 |
| initial-condition run이 기록 control보다 NLL이 낮다 | 관측 | 여러 변경이 함께 있어 causal attribution 불가 |
| branch persistence가 동일 token collapse를 줄였다 | 관측 | generation pattern상 지지; loss/architecture 분리 전 |
| branch가 coherent semantics에 commit했다 | 반증 범위 | step-1000 greedy block-4에서는 지지되지 않음 |
| residual 회전이 4-phase cycle의 유일한 원인이다 | 미검증 가설 | clean block-4도 강한 period-4를 보임 |
| 후속 innovation이 반드시 필요하다 | 미검증 | independent noise는 실패했지만 correlated tape 미실험 |
| dynamic \(K_A\)가 필요하다 | 지지 없음 | 현재 핵심 가설은 global \(K\) closure |
| simplex 또는 \(P\)가 필요하다 | 지지 없음 | 해당 결핍을 분리한 evidence 없음 |
| block-4가 matched AR dynamics보다 품질이 좋다 | 판정 불가 | prior pair dynamics 불일치, clean AR 직접 CE 부재 |

## 10. 다음 실험이 반드시 고쳐야 할 comparator

차기 architecture의 AR/block 비교는 같은 trajectory plan과 같은
one-step operator를 사용해야 한다.

```text
same initial plan m0
same transition at every token
same decoder

AR:    emit -> semantic state만 actual prefix로 re-anchor
block: emit 없이 같은 transition을 open-loop로 계속 적용
```

두 경로의 유일한 차이가 semantic re-anchoring 여부가 되면, 그때의
logit divergence와 token agreement가 순수한 rollout error다.

함께 필요한 objective audit는 다음과 같다.

- clean one-step logits에 직접 CE가 있는가
- branch path와 AR comparator가 같은 one-step transition을 쓰는가
- AR에서 trajectory code를 매 step 다시 선택하지 않는가
- block horizon 위치 정보가 shortcut으로만 사용되지 않는가
- marginal NLL, sample quality, prior calibration을 구분하는가

## 11. 현재 증거로 가능한 결론

1. 1000-update absolute prose quality만으로 architecture를 판정할 수
   없다. matched GPT-2도 심하게 undertrained됐다.
2. 그렇더라도 latent model에는 GPT-2와 다른 architecture-specific
   failure가 있다.
3. clean orbit은 강한 token fixed point를 보였다.
4. initial-condition branch는 fixed point를 약화했지만 4-phase
   limit cycle을 만들었다.
5. geometric branch persistence와 all-horizon token correctness만으로
   semantic trajectory commitment가 보장되지 않는다.
6. 현재 AR/block 결과는 공정한 동일-dynamics re-anchoring ablation이
   아니므로 새로운 comparator가 필요하다.
