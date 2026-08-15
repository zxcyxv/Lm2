# Latent-orbit 아키텍처 논쟁 기록

## 문서의 성격

이 디렉터리는 2026-07-25까지 이어진 설계 대화를 단순한 최종 결론이
아니라 **논쟁의 계보**로 보존한다. 어떤 제안이 왜 등장했고, 어떤
반론으로 범위가 줄었으며, 실제 실험이 어디까지 지지했는지를 분리해
기록한다.

이 문서는 대화의 축어록이 아니다. 수식과 용어를 현재 저장소의
구현에 맞춰 정규화한 연구 노트다. 구현 사실과 수치는 experiment
record를 우선하며, 이 문서군은 그 사실을 둘러싼 해석과 설계 논리를
담는다.

상태 표시는 다음 뜻으로 사용한다.

- **정립**: 현재 아키텍처의 계약 또는 직접적인 수학적 귀결
- **구현 사실**: 현재 저장소 코드와 manifest로 확인된 내용
- **관측**: 특정 checkpoint와 평가 프로토콜에서 얻은 증거
- **가설**: 관측을 설명하지만 아직 분리 실험으로 검증되지 않은 해석
- **보류**: 논리적으로 가능하지만 현재 최소 구조에는 채택하지 않은 안
- **기각**: 현재 목표의 필요조건이라는 주장이 철회된 안
- **제안**: 아직 구현하거나 실험하지 않은 다음 설계

## 가장 짧은 결론

이 연구의 핵심은 문맥마다 새 연산자를 생성하는 것이 아니다. causal
encoder가 token shift를 하나의 전역 시불변 latent action으로
linearize하여

\[
K E(x_{\le t})\approx E(x_{\le t+1})
\]

가 실제 encoder state 전체에서 성립하게 만드는 것이다. 그러면 같은
\(K\)의 거듭제곱으로 여러 미래 state를 만들고 병렬 decode할 수 있다.

exact inverse decoder는 틀린 latent를 의미론적으로 고쳐 주는 별도
추론기가 아니다. 따라서 encoder와 \(K\)의 local transition error가
작아야 한다. 다만 성공을 위해 latent가 정답 state와 완전히 같을
필요는 없고, 올바른 token decision region 안에 있으면 충분하다.

현재 남은 가장 큰 문제는 단순한 spectral error explosion보다
**multimodal trajectory commitment**다. 하나의 initial residual만
\(K\)로 회전시키는 현재 방식은 branch를 기하학적으로 분리했지만,
step-1000 생성에서 의미론적 trajectory 대신 4-position phase code를
학습한 정황이 관측됐다.

현재의 종합 제안은 \(K\)를 동적으로 바꾸는 것이 아니라 state를
확장하는 것이다.

```text
decoder-visible semantic state c
remaining trajectory plan      m

c_next = K c + B m
m_next = S m
```

이는 전체 state에 대해서는 하나의 전역 시불변 연산자다. plan은
초기에 한 번 선택되고, semantic state와 분리된 채 고정 dynamics로
전개된다. 이 제안은 아직 실험 evidence가 아니다.

## 문서 지도

1. [핵심 계약과 기각된 우회로](01_core_contract.md)
   - approximate conjugacy
   - exact inverse decoder에 대한 초기 오해
   - one-step residual과 finite-horizon error
   - canonicalizer \(P\), simplex/ALR, BYOL·DINO 비유의 범위

2. [Trajectory commitment 논쟁](02_trajectory_commitment.md)
   - MSE의 mode averaging과 noise의 의도
   - independent innovation, initial condition, correlated tape
   - \(h_A\) detach와 prior 입력
   - per-step rebranching과 fixed global code의 구분
   - 4-phase shortcut 가설과 그 반론

3. [실험 증거와 추론 경로 감사](03_evidence_and_inference_audit.md)
   - current step-1000 training metrics
   - matched GPT-2 및 AR/block-4 생성 패턴
   - `prior_ar`와 `prior_block4`가 실제로 동일 동역학 비교가 아니었던
     이유
   - 증거가 지지하는 범위와 지지하지 않는 범위

4. [확장된 시불변 연산자 제안](04_augmented_time_invariant_operator.md)
   - semantic state와 remaining plan의 분리
   - 하나의 initial seed와 고정 shift로 per-step innovation을
     결정론적으로 펼치는 방법
   - 원래 병렬화 철학을 보존하는 공정한 AR comparator
   - 최소 ablation과 반증 기준

5. [CE-only AR의 병렬 고정점 복원](05_parallel_ar_fixed_point_solver.md)
   - 표준 teacher forcing checkpoint의 병렬 Jacobi 추론
   - greedy AR exact 복원의 귀납적 보장
   - \(n\)-token block이 \(n-1\) iteration을 요구한 이유
   - exact 복원 가능성과 실제 병렬 가속의 구분

## 논쟁의 시간적 계보

### 1. 출발점: \(D(h_B)=B\)와 \(K^j\) 병렬 생성

최초의 핵심 직관은 다음이었다.

1. encoder가 prefix \(A\)를 \(h_A\)로 보낸다.
2. \(K h_A\approx h_B\)를 학습한다.
3. decoder가 encoder의 exact inverse라면 \(h_B\)는 다시 \(B\)로
   읽힌다.
4. 같은 shift law가 성립하면 \(K^2h_A\approx h_C\),
   \(K^3h_A\approx h_D\)가 된다.
5. 이 state들을 한 번에 decode하면 AR re-encoding 없이 여러 token을
   병렬 생성할 수 있다.

대화에서 한때 이를 \(K\)의 “가환성”이라고 불렀지만, 더 정확한
표현은 token shift와 latent action 사이의 **approximate conjugacy**
또는 intertwining이다.

\[
K E\approx E S.
\]

\(K\)가 자기 자신과 가환한다는 사실은 자명하며, 병렬화의 핵심은
같은 \(K\)가 모든 실제 encoder state에서 동일한 shift를 표현한다는
점이다.

### 2. decoder가 semantic corrector라는 오해의 철회

초기에는 \(K\)가 대략적인 생성 신호만 만들고, inverse decoder가
\(K h_A\)를 \(h_B\)처럼 해석하도록 학습할 수 있다고 기대했다.
논쟁을 통해 다음처럼 정정됐다.

- exact inverse decoder는 독립적인 보정기가 아니다.
- \(K h_A\)가 크게 틀렸을 때 이를 canonical \(h_B\)로 복원하는
  many-to-one mechanism은 구조에 없다.
- 핵심 계산 책임은 encoder geometry와 \(K\)의 transition에 있다.
- 그러나 LM head의 token decision region 자체는 many-to-one이므로
  exact latent equality가 논리적 필요조건은 아니다.

### 3. “\(K h_A\)가 \(h_B\)와 가깝다”만으로 충분한가

한 시점에는 \(K h_A\approx h_B\)여도 \(K^2h_A\)가 여전히 \(h_B\)
근처라면 token 반복이 생길 수 있다는 문제가 제기됐다. block-2
실험의 2-token 반복도 이 관점에서 해석됐다.

후속 논의에서는 그 반복이 첫 horizon에만 loss를 주던 실험의 특수한
결과였고, 1000-step 미학습도 혼입돼 있으므로 일반적 구조 결함으로
확정할 수 없다고 범위를 줄였다.

정리된 수학적 결론은 다음이다.

- 한 anchor에서 \(K h_A\approx h_B\) 하나만 맞는다고
  \(K^2h_A\approx h_C\)가 자동으로 따라오지는 않는다.
- 그러나 전역 \(K\)에 대해 모든 stride-1 state에서
  \(K h_t\approx h_{t+1}\)를 학습하면 local closure 전체가
  감독된다.
- orthogonal \(K\)는 local residual을 spectral하게 증폭하지 않는다.

### 4. 기존 checkpoint 측정의 정보 가치 논쟁

필요한 mechanism이 구조적으로 없다고 이미 증명됐다면 기존
checkpoint에서 basin 크기만 재는 것은 다음 구조를 결정할 정보가
아니라는 비판이 나왔다. 이는 “진단을 계속할 것인가, 필요조건을
만족하는 구조를 합성할 것인가”라는 연구 단계의 구분을 만들었다.

다만 이후 step-1000 generation audit는 구조적 결핍을 단순 재확인한
것이 아니라 fixed-point collapse와 4-phase cycle을 분리했다.
따라서 현재의 더 정확한 원칙은 다음과 같다.

> 수학적으로 결정된 결핍을 checkpoint로 다시 증명하지 않는다.
> 하지만 여러 가능한 실패 mechanism을 구분할 수 있다면 기존
> checkpoint audit도 새로운 정보를 줄 수 있다.

### 5. canonicalizer \(P\) 제안과 철회

decoder 앞에 non-invertible projection \(P\)를 두어 off-manifold
proposal을 canonical state로 끌어오는 안이 제안됐다. identity,
idempotence, residual locality, contraction 같은 계약도 함께
논의됐다.

사용자는 목표가 무오차 AR 등가가 아니라 128-token 정도의 유용한
finite-horizon 병렬 생성이라는 점, token마다 추가 correction을
계산하면 이점이 줄어든다는 점을 지적했다. 이후 \(P\)는 최소
필요조건에서 제외됐다.

현재 지위는 다음과 같다.

- \(P\)의 idempotence는 token 생성 목표에서 도출되지 않는다.
- 강한 \(P\)는 두 번째 LM이 되어 \(K\)를 우회할 수 있다.
- raw latent error가 충분히 작은데 decoder sensitivity만 실패한다는
  증거가 있을 때 readout ablation으로 검토할 수 있다.

### 6. simplex/ALR 제안과 범위 축소

latent를 확률 simplex로 전단사 매핑하고 stochastic transition을
적용하면 rollout이 공간 밖으로 발산하지 않는다는 안이 제안됐다.
ALR inverse의 경계 발산을 최종 token 결정의 sharpening으로 해석하는
반론, stationary collapse는 문맥별 \(K_A\)라면 적용되지 않는다는
반론도 나왔다.

후속 검토에서 다음이 분리됐다.

- ALR은 정보 손실 없는 재좌표화가 될 수 있다.
- 그러나 simplex 내부라는 사실이 valid encoder manifold를
  보장하지 않는다.
- ALR inverse 출력은 최종 vocabulary logit이 아니라 nonlinear
  inverse decoder의 입력이므로 경계 conditioning을 단순한 token
  sharpening으로 동일시할 수 없다.
- 현재 구현은 문맥별 \(K_A\)가 아니라 전역 orthogonal \(K\)다.
- orthogonal \(K\)가 이미 spectral explosion을 없애므로 simplex는
  현재 문제의 필수 해법이 아니다.

### 7. BYOL·DINO collapse와의 유사성 논쟁

simplex semantic code, centering, sharpening, rank-1 transition collapse가
BYOL/DINO의 representation collapse 문제와 닮았다는 의견이 나왔다.
비유는 일부 유용하지만 동일 문제는 아니다.

- future online encoder state는 semantic target을 제공한다.
- centering과 sharpening은 branch usage와 representation collapse를
  완화할 수 있다.
- 그러나 \(K p_A=p_B\)만 만족하는 rank-1 operator가
  \(K^2p_A=p_C\)로 전진하지 않는 문제는 horizon closure의 문제다.
- collapse 방지만으로 correct temporal transport가 생기지는 않는다.

### 8. 목표 수준의 재확정

목표는 무한 horizon에서 AR과 오차 없이 동치인 모델이 아니다.

- AR은 매 token 실제 출력을 다시 encode하여 error를 제거한다.
- block generation은 open-loop error를 감수하는 대신 병렬성을 얻는다.
- 이 tradeoff 자체를 없애기보다 local error를 충분히 낮춰 약
  128-token까지 decision을 보존하는 것이 목표다.

이 재확정에 따라 \(P\), simplex, iterative correction은 최소
아키텍처에서 빠졌다.

### 9. noise를 trajectory commitment로 해석

\(K h_A\to h_B\) 회귀는 multimodal future의 평균에 가까워질 수 있다.
따라서 주변의 여러 proposal을 trajectory 전체 CE로 경쟁시키면
mode-seeking이 가능하다는 발상이 도입됐다.

독립 horizon noise는 각 step에서 semantic branch가 다시 바뀔 수
있었다. 이에 따라 현재 실험은 context-conditioned initial residual을
한 번만 넣고 이후 같은 \(K\)로 운반하는 방식으로 바뀌었다.

### 10. initial-condition 실험의 성공과 실패

step-1000 결과는 다음 두 사실을 동시에 보였다.

- branch distance가 보존되고 prior winner accuracy가 chance를 넘었으며,
  validation marginal NLL도 사전등록 control보다 좋아졌다.
- free generation에서는 동일 token fixed point가 줄었지만 강한
  4-position cycle이 나타났다.

따라서 “geometric branch commitment”는 어느 정도 성립했지만
“semantic trajectory commitment”는 입증되지 않았다.

### 11. AR이 block-4보다 나빠 보인 이유에 대한 추론 감사

현재 `prior_ar`는 매 token 실제 prefix를 다시 encode한 뒤 prior와
initial residual을 새로 계산한다. `prior_block4`는 하나를 고른 뒤
4 horizon 동안 유지한다.

따라서 둘은 단순히 re-anchoring 간격만 다른 것이 아니다.
`prior_ar`는 token-local rebranching이고 `prior_block4`는 block-level
commitment다. 또한 clean center는 h1 MSE를 받지만 직접적인 clean AR
CE 경로는 없다. 이 때문에 당시 생성 결과로 “open-loop block이
re-anchored AR보다 본질적으로 우월하다”고 결론내릴 수 없다.

### 12. 후속 innovation 논쟁

사용자는

\[
u_{i,2}=K u_{i,1}+\eta_k(Ku_{i,1})
\]

처럼 두 번째 step에서도 새 correction이 필요하지 않느냐고 물었다.
이로부터 두 모델이 분리됐다.

- initial-condition model: 하나의 branch가 전체 미래를 선택
- per-step transition model: 각 step에 새로운 innovation이 존재

매 step 새로운 \(k\)를 고르면 branch tree와 순차 의존이 생긴다.
반면 하나의 global code \(z_i\)에서 상관된 innovation tape 전체를
미리 만들면 branch identity와 병렬 계산을 보존할 수 있다.

### 13. causal state와 plan의 긴장

외부 분석은 canonical \(h_B=E(A,B)\)가 보지 않은 \(C,D,E\)의
실현값을 담을 수 없으므로, initial residual 하나가 \(h_B\)와
canonical하게 정렬되면서 미래 plan까지 품으라는 요구는 모순이라고
비판했다.

이 비판은 중요한 tension을 포착했지만, initial randomness 전체를
처음의 \(z\)에 미리 샘플할 수 있으므로 time-invariant initial-value
model 자체가 논리적으로 불가능한 것은 아니다. 또한 token 성공에는
canonical equality보다 decision-region membership만 필요하다.

남은 실질적 문제는 하나의 회전 residual이 current semantics와
remaining plan을 동시에 표현하게 만든 현재 parameterization이다.

### 14. 현재 종합: 연산자를 바꾸지 말고 state를 완전하게 만든다

최종적으로 다음이 정리됐다.

- 전역 시불변 \(K\)는 충분할 수 있다.
- 단, one-step law는 한 anchor가 아니라 encoder state 전체에서
  성립해야 한다.
- multimodal sample을 선택한 뒤의 완전한 state는 canonical semantic
  state 하나가 아니라 semantic state와 remaining plan의 쌍이다.
- 이 둘을 확장된 하나의 전역 시불변 operator로 전개하면 initial
  commitment와 stepwise innovation을 동시에 표현할 수 있다.

이 제안의 자세한 계약과 반증 기준은
[04_augmented_time_invariant_operator.md](04_augmented_time_invariant_operator.md)에
기록한다.

### 15. CE-only AR를 병렬 고정점으로 복원

후속 실험은 branch, closure, MSE를 모두 제거하고 표준 one-step CE로
학습한 exact-inverse K=1 모델을 사용했다. 추론에서
\(K^1h_A,\ldots,K^nh_A\)를 병렬 초기값으로 decode한 뒤, 후보 미래
전체를 simultaneous causal argmax로 반복 갱신했다.

동일 모델의 greedy AR 출력은 block 4, 8, 16, 32에서 각각
3, 7, 15, 31회에 정확히 복원됐다. 이는 우연한 수치 패턴이 아니다.
첫 token이 구조적으로 AR과 같고 causal Jacobi iteration이 확정
prefix를 한 번에 적어도 한 token씩 늘리므로 \(n-1\) 수렴이
귀납적으로 보장된다.

따라서 표준 AR를 병렬 fixed-point 문제로 재표현하고 정확히 푸는
것은 실제로 가능하다. 다만 현재 solver는 causal frontier를 한
iteration당 한 칸만 전진시켜 계산 가속은 얻지 못했다. 자세한
구분은
[05_parallel_ar_fixed_point_solver.md](05_parallel_ar_fixed_point_solver.md)에
기록한다.

## 현재 정립된 것

- 현재 구현의 \(K\)는 모든 문맥이 공유하는 전역 orthogonal
  operator다.
- 같은 \(K\)의 power는 autonomous dynamics이며, 동적으로 생성되는
  operator와 다르다.
- global stride-1 one-step closure와 orthogonality는 finite-horizon
  transport의 핵심 계약이다.
- exact inverse decoder는 독립 semantic corrector가 아니다.
- \(P\), idempotence, simplex, dynamic \(K_A\)는 현재 최소
  필요조건이 아니다.
- 현재 step-1000 initial-condition 모델의 geometric branch
  persistence는 semantic coherence를 보장하지 않았다.
- 기록된 `prior_ar`와 `prior_block4`는 같은 stochastic dynamics의
  순수 re-anchoring ablation이 아니다.
- 표준 CE AR의 greedy trajectory는 병렬 causal fixed-point
  iteration으로 exact 복원 가능하다.
- 현재 Jacobi solver의 \(n-1\) exact 수렴은 첫 token 동일성과 causal
  triangular dependency의 귀납적 결과이며, 병렬 가속의 증거는
  아니다.

## 아직 열린 것

- 128-token까지 필요한 one-step residual과 decoder margin의 실제
  크기
- phase cycle의 원인이 initial residual의 제한된 orbit인지,
  objective의 horizon shortcut인지, 두 요인의 상호작용인지
- fixed global plan code가 arbitrary tape 없이 충분한 semantic
  innovation sequence를 표현할 수 있는지
- prefix prior를 argmax selector로 써야 하는지, trajectory
  distribution으로 sample해야 하는지
- augmented semantic/plan state가 \(K\)의 책임을 보존하면서 현재
  initial-condition 모델을 실제로 능가하는지
- causal frontier를 한 iteration에 여러 token 전진시키는 spectral,
  multigrid 또는 parallel-prefix update가 가능한지

## 원자료

- [아키텍처 철학](../architecture_philosophy.md)
- [실험 계보](../lineage.md)
- [현재 initial-condition experiment manifest](../../experiments/records/EXP-20260725-k4-ha-skew-stride1-batch64-micro16-initial-condition-three-trajectory-prior-tau005-h1-online-mse-13m/manifest.md)
- [step-1000 GPT-2 비교 해석](../../experiments/records/EXP-20260725-k4-ha-skew-stride1-batch64-micro16-initial-condition-three-trajectory-prior-tau005-h1-online-mse-13m/step1000_gpt2_ar_sentence_comparison/analysis.md)
