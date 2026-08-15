# 핵심 계약과 기각된 우회로

## 1. 문제를 가장 정확하게 쓰기

token sequence의 causal prefix state를

\[
h_t=E(x_{\le t})
\]

라고 둔다. \(E\)는 reversible causal encoder이고, decoder는 같은
파라미터로 정의되는 exact inverse다.

\[
D=E^{-1}.
\]

token 하나를 뒤로 미는 이산 shift를 \(S\)라 하면 이 모델이 학습하려는
관계는

\[
K E(x_{\le t})\approx E(Sx_{\le t})
\]

이다. 즉

\[
K E\approx E S.
\]

이것이 성립하면

\[
K^j E\approx E S^j
\]

이므로, 한 번의 prefix encoding 뒤에 \(K^1h_t,\ldots,K^Hh_t\)를
만들어 미래 token들을 한꺼번에 decode할 수 있다.

### “가환성”이라는 표현의 정규화

대화에서는 이 성질을 \(K\)의 가환성이라고 부르기도 했다. 하지만
\(K^aK^b=K^{a+b}\)는 한 행렬의 power에 대해 자명하다. 실제로 필요한
명제는 다음이다.

- token shift \(S\)와 latent shift \(K\)가 encoder를 사이에 두고
  approximate하게 intertwine한다.
- 같은 \(K\)가 모든 실제 encoder state에서 동일한 one-token
  transition을 표현한다.

따라서 이 문서에서는 이를 **global approximate conjugacy** 또는
**global one-step closure**라고 부른다.

## 2. 시불변 \(K\)로 충분하다는 명제의 정확한 범위

### 정립: 한 점의 근사만으로는 충분하지 않다

\[
Kh_A\approx h_B
\]

한 식만으로

\[
K^2h_A\approx h_C
\]

가 논리적으로 자동 도출되지는 않는다. local residual을

\[
r_t=Kh_t-h_{t+1}
\]

라고 하면

\[
K^2h_A-h_C
=K(Kh_A-h_B)+(Kh_B-h_C)
=Kr_A+r_B.
\]

첫 항뿐 아니라 \(h_B\)에서의 local residual도 작아야 한다.

### 정립: 전역 one-step law라면 power로 충분하다

같은 \(K\)에 대해 모든 실제 state의 \(r_t\)가 작다면

\[
e_j=K^jh_t-h_{t+j}
\]

는

\[
e_j=\sum_{q=0}^{j-1}K^{j-1-q}r_{t+q}
\]

가 된다. 현재처럼 모든 위치를 stride-1 anchor로 사용하면
\(r_t,r_{t+1},\ldots\)가 각각 다른 anchor의 h1 loss로 감독된다.

따라서 문맥별 \(K_t\)를 생성하지 않아도 된다. 오히려 전역 \(K\)는
encoder가 언어 shift를 공통 latent action으로 linearize한다는 강한
가설이며, 이 가설이 병렬화의 원천이다.

### 정립: orthogonality가 주는 것과 주지 않는 것

현재 구현은

\[
K=\exp(A-A^\top)
\]

이므로 \(K\)가 orthogonal이다.

\[
\|K^qv\|_2=\|v\|_2.
\]

따라서

\[
\|e_j\|_2
\le \sum_{q=0}^{j-1}\|r_{t+q}\|_2.
\]

orthogonality는 다음을 준다.

- spectral explosion 방지
- branch residual의 norm과 pairwise distance 보존
- finite-horizon error의 최악 선형 bound

그러나 다음을 자동으로 주지는 않는다.

- local residual의 상쇄
- 올바른 token decision
- multimodal future의 한 mode 선택
- 128-token까지 충분한 error budget

즉 `orthogonal`은 `정확한 transition`과 동의어가 아니다.

## 3. exact inverse decoder에 대한 초기 오해

### 초기 가설

초기 직관은 \(K\)가 정확한 \(h_B\)를 만들 필요 없이 대략적인
“생성 신호”만 주면 decoder가 이를 자연스럽게 \(h_B\)처럼 취급할 수
있다는 것이었다.

이를 강하게 쓰면 다음과 같은 기대다.

> \(K h_A\)가 \(h_B\)와 상당히 달라도 decoder가 Lipschitz
> continuity나 학습된 invariance를 통해 \(B\)로 교정한다.

### 정정

exact inverse decoder는 independent semantic corrector가 아니다.
encoder와 decoder는 같은 reversible mapping의 양방향이며,
decoder 앞에는 여러 off-manifold state를 하나의 canonical state로
합치는 별도 연산이 없다.

따라서 latent transport의 책임은 주로 다음에 있다.

- encoder가 shift에 적합한 geometry를 형성한다.
- \(K\)가 그 geometry에서 local transition을 정확히 학습한다.
- LM head까지의 decision margin이 남은 residual을 허용한다.

### 그러나 exact equality도 필요조건은 아니다

전체 readout을

\[
F(h)=\operatorname{LMHead}(D(h))
\]

라고 하면 argmax token에 대해 latent space는 이미 decision region들로
분할된다.

\[
\mathcal R_y=\{h:\arg\max F(h)=y\}.
\]

성공의 최소 조건은

\[
K^jh_t\in\mathcal R_{x_{t+j}}
\]

이다. 즉

\[
K^jh_t=h_{t+j}
\]

는 충분조건이지만 논리적 필요조건은 아니다.

이 구분이 중요한 이유는 두 극단을 모두 피하기 위해서다.

- decoder가 큰 latent error를 알아서 고친다는 기대는 근거가 없다.
- 반대로 epsilon 수준의 exact state equality가 없으면 token 생성이
  불가능하다는 주장도 과하다.

현재 전략은 decision region을 직접 설계하기보다 latent error를 매우
작게 만들어 안정적으로 region 안에 들어가는 것이다.

## 4. 모델의 실제 목표: 무한 AR 동치가 아닌 finite horizon

이 모델은 AR error correction을 공짜로 제거하려는 모델이 아니다.

AR 생성은 매 token마다

```text
emit token -> append to prefix -> encode again
```

을 수행하여 state를 실제 생성 prefix에 다시 고정한다. block
generation은 이 feedback을 생략하기 때문에 local residual이
누적되는 것이 원리적인 tradeoff다.

현재 목표는 다음과 같다.

- error accumulation 자체를 0으로 만들지 않는다.
- encoder와 \(K\)의 local residual을 최대한 작게 만든다.
- orthogonality로 residual의 spectral amplification을 막는다.
- decoder margin 안에서 약 128-token까지 useful token decision을
  유지한다.

따라서 token마다 반복 보정하는 구조는 정확도를 높일 수 있어도
원래 계산 이점과 직접 충돌한다.

## 5. block 반복에 대한 논쟁

### 제기된 문제

\(Kh_A\approx h_B\)여도 \(K^2h_A\)가 \(h_C\)가 아니라 여전히
\(h_B\)와 비슷하면 두 번째 token도 \(B\)가 된다. 과거 block-2의
2-token 반복이 이런 horizon non-separation 때문일 수 있다는
가설이 나왔다.

### 범위 축소

그 block-2 실험은 첫 token에만 loss를 주던 조건이었다. 현재 실험은
모든 네 horizon에 token CE를 준다. 또한 1000 update라는 짧은
최적화 구간이 겹쳐 있다.

따라서 과거 반복은 다음을 시사하지만 일반 명제를 확정하지 않는다.

- h1-only loss는 이후 horizon의 phase와 semantics를 보장하지 않는다.
- horizon state들이 token decision 관점에서 분리돼야 한다.
- 반복이 생겼다는 사실만으로 새로운 \(P\)나 simplex가
  필요하다고 결론내릴 수 없다.

후속 step-1000 generation에서는 동일 token 반복뿐 아니라 4-position
cycle이 분리 관측됐다. 이는 단순 horizon collapse보다 더 복잡한
shortcut이 가능함을 보여 준다.

## 6. 필요조건과 최적화 장치의 구분

### 현재 핵심 계약

1. **Orbit-ready encoder**

   실제 causal state들을 하나의 전역 shift action으로 linearize해야
   한다.

2. **Global local closure**

   모든 stride-1 state에서 \(Kh_t\approx h_{t+1}\)여야 한다.

3. **Non-explosive propagation**

   현재는 orthogonal \(K\)가 이 역할을 맡는다.

4. **Token decision supervision**

   학습할 horizon의 state가 실제 token loss를 받아야 한다.

5. **Sufficient decoder margin**

   finite-horizon residual이 올바른 token region을 벗어나지 않아야
   한다.

6. **Multimodal commitment**

   하나의 생성 trajectory 안에서 branch identity가 의미론적으로
   일관돼야 한다.

### 유용할 수 있지만 아직 필요조건이 아닌 것

- all-horizon latent MSE
- residual-direction tube loss
- decoder-aware consistency
- readout canonicalizer
- simplex coordinate
- 문맥별 또는 state-dependent \(K\)

all-horizon latent loss는 128-step credit path를 짧게 하고 correlated
residual을 직접 줄일 수 있다. 그러나 전역 \(K\)+stride-1 h1
regression에서 local closure 자체가 빠진 것은 아니다.

## 7. 기존 checkpoint를 더 측정할 가치에 대한 원칙

논쟁 중 한때 “필요 mechanism이 구조적으로 없다는 결론이 이미
났다면 checkpoint 측정은 결핍의 크기만 재므로 정보이론적 이점이
없다”는 비판이 나왔다.

이 비판은 다음 상황에서 맞다.

- 실험 결과와 무관하게 수학적으로 불가능한 계약을 반복 측정할 때
- 동일 failure를 새 지표 이름으로 다시 확인할 때
- 다음 설계 선택을 구분하지 못하는 diagnostic을 추가할 때

하지만 다음 상황에서는 기존 checkpoint도 가치가 있다.

- fixed point와 limit cycle처럼 서로 다른 failure class를 구분할 때
- decoder sensitivity와 latent transport error를 분해할 때
- 추론 코드가 학습 동역학과 일치하는지 감사할 때
- 다음 두 아키텍처가 서로 다른 예측을 내놓는 지표를 측정할 때

실제로 generation audit는 clean orbit의 token fixed point와
initial-condition branch의 4-phase cycle을 구분했다. 이는 단순한
“성능이 나쁘다”보다 차기 설계를 선택하는 데 더 많은 정보를 줬다.

## 8. canonicalizer \(P\) 논쟁

### 제안

off-manifold proposal을 decoder 전에 canonical future state로
수렴시키는 \(P\)가 제안됐다.

\[
K^jh_A\to P(K^jh_A)\to D\to\text{LM head}.
\]

제안된 계약에는 다음이 포함됐다.

- \(P(h_t)=h_t\)
- \(P(P(z))=P(z)\)
- residual locality
- target state 주변 contraction
- horizon-shared parameters

### 이 제안이 포착한 진짜 문제

“decoder가 \(K h_A\)를 \(h_B\)처럼 알아서 취급한다”는 기대에는
many-to-one correction mechanism이 없었다. \(P\)는 그 빠진 역할을
명시적인 모듈로 만든다는 점에서 논리적으로 일관됐다.

### 최소 구조에서 제외된 이유

1. token head가 이미 many-to-one decision region을 제공한다.
2. \(P\)를 readout 직전에만 두면 raw orbit error는 줄지 않는다.
3. transition 사이에 두면 nonlinear recurrent correction이 된다.
4. 강한 \(P\)는 미래를 직접 계산하여 \(K\)를 우회할 수 있다.
5. idempotence는 projection이라는 이름에서 나온 조건이지 token
   correctness의 귀결이 아니다.
6. 목표가 exact infinite AR equivalence가 아니라 finite-horizon
   approximation이므로 계산비 증가를 정당화할 증거가 없다.

### 현재 지위

**기각된 것은 \(P\)의 가능성이 아니라 필요조건이라는 주장이다.**

다음 evidence가 나올 경우에만 readout ablation으로 되살릴 수 있다.

- latent rollout state가 \(h_{t+j}\)에 충분히 가깝다.
- 그런데 작은 latent residual 방향에 decoder logits가 과도하게
  민감하다.
- 가벼운 one-shot readout adapter가 token margin을 크게 개선한다.

## 9. simplex와 ALR 논쟁

### 원래 제안

ALR 같은 전단사 coordinate map \(f\)로

\[
h\leftrightarrow p\in\Delta
\]

를 만들고 simplex 안에서 stochastic transition을 학습한다.

\[
h_A\xrightarrow f p_A\xrightarrow K \hat p_B
\xrightarrow{f^{-1}}\hat h_B\xrightarrow D B.
\]

이 제안은 다음 이점을 주장했다.

- \(f^{-1}(f(h_B))=h_B\)로 exact inverse 이점 보존
- stochastic \(K\)의 simplex closure
- \(K^j p_A\)가 simplex 밖으로 발산하지 않음
- KL 기반 transition loss

### 반론 1: simplex closure는 semantic closure가 아니다

simplex의 임의 점이 valid encoder state를 뜻하지 않는다. ALR은
\(\mathbb R^{d-1}\)를 simplex interior와 재좌표화할 뿐, encoder
manifold나 token semantics를 자동으로 만든다.

따라서

> probability처럼 보이는 좌표

와

> transition에 유용한 semantic probability

는 다르다.

### 반론 2: ALR boundary는 최종 logit이 아니다

ALR inverse

\[
h_i=\log\frac{p_i}{p_{\mathrm{ref}}}
\]

는 \(p_i\to0\)에서 ill-conditioned하다. 이를 “불필요 token을
\(-\infty\) logit으로 보내는 정상적인 sharpening”이라고 해석하는
반론이 있었다.

하지만 여기서 \(h\)는 vocabulary logit이 아니라 nonlinear inverse
decoder의 입력이다. 큰 log-ratio가 decoder를 지난 뒤 같은 token
confidence로 안정적으로 squash된다는 보장은 없다. 따라서 minimum
probability floor, clipping, temperature 또는 decoder-aware
conditioning이 필요할 수 있다.

### 반론 3: contraction은 semantics도 지울 수 있다

stochastic transition의 contraction은 noise를 줄일 수 있지만
horizon identity와 context separation도 줄일 수 있다. stationary
collapse 주장은 문맥별 \(K_A\)에는 그대로 적용되지 않지만, 현재
구현은 애초에 문맥별 \(K_A\)가 아니다.

### 현재 지위

현재 orthogonal \(K\)가 이미 Euclidean spectral explosion을
제거한다. 따라서 simplex는

- error contraction
- semantic/phase separation
- ALR conditioning

사이의 다른 tradeoff를 갖는 비교군일 수는 있지만, 현재 문제의
필수 해법은 아니다.

## 10. BYOL·DINO 비유의 유효 범위

논쟁에서는 다음 유사성이 제기됐다.

- future encoder가 target representation을 제공한다.
- centering은 한 code로의 collapse를 막는다.
- sharpening은 mode assignment를 강화한다.
- teacher/student 또는 online/target 정합이 semantic code를 만든다.

이 비유는 representation collapse를 생각하는 데 유용하다. 하지만
temporal operator에는 추가 계약이 있다.

예를 들어 모든 column이 같은 \(p_B\)인 rank-1 matrix는

\[
Kp_A=p_B
\]

를 완벽히 만족해도

\[
K^2p_A=p_B
\]

에 머문다. centering과 sharpening이 branch usage를 개선할 수는
있지만 \(p_B\to p_C\)라는 temporal closure를 자동으로 만들지는
않는다.

현재 전역 \(K\)에서는 stride-1 local closure가 이 역할을 한다.
문맥별 \(K_A\)를 anchor에서 한 번 생성해 power를 쓰는 설계라면
별도의 same-operator all-horizon supervision이 필요하다.

## 11. 이 문서에서 유지하는 최종 판단

- 시불변 전역 \(K\)는 표현력 부족의 동의어가 아니다.
- global one-step conjugacy가 성립하면 power rollout이 핵심 이점을
  제공한다.
- exact inverse decoder는 semantic repair module이 아니다.
- local residual을 거의 0으로 만드는 encoder/\(K\) 학습이 우선이다.
- token decision region 때문에 exact state equality는 필요조건이
  아니다.
- \(P\)와 simplex는 증거 없이 최소 구조에 넣지 않는다.
- 현재 가장 큰 미해결 항목은 trajectory의 stochastic plan을
  decoder-visible semantic state와 어떻게 충돌 없이 운반하느냐이다.

