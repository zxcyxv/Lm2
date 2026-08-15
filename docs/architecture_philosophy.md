# Latent-orbit 아키텍처 철학과 필요조건

## 문서 상태

- 이 문서는 실험 결과가 아니라 현재까지의 아키텍처 논쟁을 정리한
  설계 메모다.
- 구현 사실은 현재 저장소를 기준으로 하며, 실험 계보와 수치는
  [lineage.md](lineage.md)와 각 experiment record를 우선한다.
- 아래에서 `필요조건`과 `유용한 최적화 장치`를 구분한다. 후자를
  실험 없이 필요조건으로 승격하지 않는다.

## 1. 핵심 명제

causal prefix의 latent state를

\[
h_t=E(x_{\le t})
\]

라고 하고 decoder가 encoder의 exact inverse

\[
D=E^{-1}
\]

라고 하자. 이 아키텍처가 학습하려는 것은 token shift \(S\)와 latent
operator \(K\) 사이의 approximate conjugacy다.

\[
K E(x_{\le t})\approx E(Sx_{\le t})=h_{t+1}
\]

이 관계가 실제 state 전체에서 성립하면 같은 \(K\)의 거듭제곱으로

\[
K^j h_t\approx h_{t+j}
\]

를 얻을 수 있다. 따라서 한 prefix를 한 번 encode한 뒤
\(K h_t,\ldots,K^Hh_t\)를 만들고 한꺼번에 decode할 수 있다. 생성된
token을 매번 다시 prefix에 삽입하고 encode하지 않는 것이 이
아키텍처의 핵심 계산 이점이다.

목표는 AR과 오차 없이 동치인 무한 rollout이 아니다. 허용되는
AR--parallel tradeoff를 받아들이되, 학습된 local transition error를
충분히 낮춰 예컨대 128-token horizon까지 token decision을 보존하는
것이다.

## 2. Exact inverse decoder의 역할에 대한 정정

초기의 잘못된 기대는 \(K h_t\)가 \(h_{t+1}\)에서 상당히 벗어나도
decoder가 이를 의미론적으로 교정할 것이라는 생각이었다. exact
inverse decoder는 독립적인 semantic corrector가 아니다. 그 동작은
encoder와 같은 파라미터로 결정되며, latent transition의 핵심 계산은
encoder와 \(K\)가 담당해야 한다.

그렇다고 서로 다른 latent point를 반드시 동일한 canonical state로
합쳐야 하는 것은 아니다. 전체 token readout

\[
F(h)=\operatorname{LMHead}(D(h))
\]

은 argmax 또는 sampling까지 포함하면 이미 many-to-one이다. 서로
다른 latent point가 같은 token decision region에 들어갈 수 있다.
따라서 성공 조건은 반드시

\[
K^j h_t=h_{t+j}
\]

일 필요가 없고,

\[
K^j h_t\in\mathcal R_{x_{t+j}}
\]

이면 된다. 다만 이 decision-region 조건을 안정적으로 만족시키기
위해 latent error 자체를 거의 0에 가깝게 학습하는 것이 현재의
직접적인 전략이다.

## 3. `K`의 거듭제곱도 엄연한 dynamics다

용어를 구분해야 한다.

- \(h_{j+1}=Kh_j\)는 state가 시간에 따라 변하는 autonomous linear
  dynamical system이다.
- 이때 \(K\) 자체는 모든 시간과 문맥에서 동일한 time-invariant
  operator다.
- \(K_t\) 또는 \(K_A=g(h_A)\)는 state/time/context-dependent
  operator다.

따라서 \(K\)를 연쇄적으로 곱하는 것을 `dynamic evolution`이라고
부르는 것은 정확하다. 다만 이는 `dynamically generated operator`와
다른 뜻이다.

현재 철학에서 하나의 전역 \(K\)는 결핍이 아니라 강한 핵심 가설일
수 있다. encoder가 언어의 shift를 하나의 공통 latent action으로
linearize할 수 있다면, 동일한 \(K\)의 power라는 간결함이 병렬화의
원천이 된다. 문맥별 \(K_A\)를 도입하면 표현력은 늘지만 이 전역
conjugacy 가설을 약화시키고 별도의 long-rollout 감독을 요구한다.

두 설계를 섞어서 평가하면 안 된다.

## 4. One-step error와 누적 error

전역 \(K\)에서 local residual을

\[
r_t=Kh_t-h_{t+1}
\]

라고 하자. \(j\)-step rollout error는

\[
e_j=K^jh_t-h_{t+j}
\]

이고 다음 recurrence를 만족한다.

\[
e_{j+1}=Ke_j+r_{t+j}
\]

따라서

\[
e_j=
\sum_{q=0}^{j-1}K^{j-1-q}r_{t+q}.
\]

중요한 점은 \(K^2h_t-h_{t+2}\)가 최초의
\(Kh_t-h_{t+1}\) 하나로만 결정되지는 않는다는 것이다.

\[
K^2h_t-h_{t+2}
=K(Kh_t-h_{t+1})+(Kh_{t+1}-h_{t+2}).
\]

그러나 현재처럼 같은 전역 \(K\)를 사용하고 모든 위치를 stride-1
anchor로 삼아 h1 regression을 주면, 두 번째 항 역시 다음 anchor의
h1 loss로 직접 감독된다. 즉

\[
\sum_t d(Kh_t,h_{t+1})
\]

는 전역 \(K\)에 대해서 이미 teacher-forced local closure 전체다.
별도로 \(K h_{t+1}\to h_{t+2}\) loss를 중복해서 만들 필요가 없다.

반대로 문맥별 \(K_A\)를 anchor에서 한 번 생성하고 \(K_A^j\)를
사용한다면, 다음 anchor의 h1 loss는 \(K_{A+1}h_{A+1}\)을 감독할
뿐 \(K_Ah_{A+1}\)을 감독하지 않는다. 그 설계에서는 same-operator
closure 또는 all-horizon rollout loss가 별도로 필요하다.

## 5. Orthogonal `K`가 바꾸는 필요조건

현재 구현은

\[
K=\exp(A-A^\top)
\]

이므로 \(K\)가 orthogonal이다.

\[
\|K^qv\|_2=\|v\|_2.
\]

따라서 local residual은 증폭되지 않으며

\[
\|e_j\|_2
\le
\sum_{q=0}^{j-1}\|r_{t+q}\|_2
\]

라는 finite-horizon bound를 얻는다. 최악에는 선형 누적이지만
spectral explosion은 없다.

이 조건 아래에서 별도의 `residual-aware stabilizer`는 구조적
필요조건이 아니다. 우선 해야 할 일은 모든 위치의 \(r_t\)를 충분히
작게 만드는 것이다. 실제 residual 방향의 tube loss나 all-horizon
latent loss는 다음 역할을 하는 선택적 최적화 장치다.

- residual의 상관 때문에 특정 horizon에서 오차가 같은 방향으로
  합쳐지는 현상을 직접 줄인다.
- 128-step 최종 오차에 더 짧은 credit path를 제공한다.
- decoder decision margin을 실제 rollout error 방향으로 넓힌다.

즉 h1 stride-1 regression과 orthogonality가 이론적 계약이고,
all-horizon loss는 최적화 및 finite-horizon 인증을 강화하는
수단이다. 후자를 전자와 같은 수준의 필요조건으로 부르지 않는다.

## 6. 왜 latent canonicalizer `P`는 최소 구조가 아닌가

한때 다음과 같은 readout canonicalizer가 제안됐다.

\[
K^jh_t\longrightarrow P(K^jh_t)\longrightarrow D\longrightarrow
\operatorname{LMHead}.
\]

이는 현재 목표에 비해 과한 가정이다.

1. token head가 이미 many-to-one decision region을 제공하므로 여러
   latent point를 하나의 hidden state로 합칠 필요가 없다.
2. \(P\)를 \(K^jh_t\) 뒤에서 한 번만 사용하면 raw \(K\)-orbit의
   누적 오차는 줄이지 못하고 readout boundary만 바꾼다.
3. \(P\)를 매 transition 사이에 넣으면 nonlinear recurrent
   correction이 되어 단순한 \(K^j\) 병렬화의 이점을 약화시킨다.
4. 강한 \(P\)는 두 번째 미래 예측기가 되어 \(K\)의 transport
   책임을 대신할 수 있다.
5. \(P(P(z))=P(z)\)라는 idempotence는 projection이라는 명칭에서
   나오는 조건일 뿐 token 생성 목표에서 도출되는 필요조건이 아니다.
   상수 함수도 idempotent이므로 올바른 미래 보존도 보장하지 않는다.

따라서 \(P\)는 latent error가 이미 충분히 작지만 decoder sensitivity
때문에 token decision만 실패한다는 evidence가 나온 뒤 검토할
readout ablation이다.

## 7. Simplex 제안의 정확한 지위

ALR 같은 전단사 \(f\)를 사용해

\[
h_t\xrightarrow{f}p_t\in\Delta,\qquad
p_t\xrightarrow{K}Kp_t,\qquad
Kp_t\xrightarrow{f^{-1}}\hat h_{t+1}
\]

로 바꾸는 것은 정보 손실 없는 재좌표화가 될 수 있다. stochastic
\(K\)는 simplex를 보존하고 total variation이나 KL에서 오차를
비팽창적으로 만들 수 있다.

그러나 다음은 자동으로 따라오지 않는다.

- simplex 내부의 모든 점이 valid encoder state인 것은 아니다.
- simplex coordinate가 transition에 유용한 semantic code가 되는
  것은 joint orbit loss가 학습해야 한다.
- ALR inverse는 경계에서 ill-conditioned이며, 그 출력은 최종
  vocabulary logit이 아니라 nonlinear inverse decoder의 입력이다.
- Markov contraction은 오류뿐 아니라 horizon identity와 semantic
  separation도 줄인다.

현재 orthogonal \(K\)는 이미 Euclidean error explosion을 제거한다.
따라서 simplex는 근본적 해결책이나 필수 구성요소가 아니라,
finite-horizon contraction과 phase 보존 사이의 다른 inductive
bias를 시험하는 비교군이다.

## 8. Noise trajectory의 의도와 성립 범위

MSE로

\[
Kh_t\to h_{t+1}
\]

를 회귀하면 multimodal conditional future의 평균 쪽으로 갈 수 있다.
그 주변에 여러 noise sample을 만들고 trajectory 전체 CE의 soft-min
또는 best-of-\(N\)으로 학습하는 것은 정당한 mode-seeking 발상이다.

현재 목적함수는 각 trajectory의 horizon별 CE를 더해

\[
C_i=\sum_{j=1}^{H}\operatorname{CE}_{i,j}
\]

를 만든 뒤 하나의 responsibility를 trajectory 전체에 적용한다.
따라서 horizon마다 다른 branch를 고르는 것은 아니며,
sequence-level commitment의 요소가 이미 있다. 앞선 noise도 \(K\)를
통해 다음 horizon으로 전파된다.

그럼에도 noise/branch 방식만으로 충분하지 않은 이유는 다음과 같다.

### 8.1 이전 Gaussian 방식은 mode의 위치를 학습하지 않는다

zero-mean diagonal Gaussian은 평균 주변의 scale만 학습한다. 실제
mode가 놓인 방향, mode별 질량, 비대칭 구조를 명시적으로 표현하지
않는다. encoder와 decoder가 우연히 Gaussian 영역을 적절한 token
basin들로 조직해야 한다.

### 8.2 Best-of-\(N\)은 좋은 prior mass를 보장하지 않는다

gold future CE로 가장 좋은 sample을 선택하면 `적어도 하나의 좋은
sample이 존재함`은 학습할 수 있다. 하지만 임의로 뽑은 한 sample이
좋을 확률이나 정답 없이 좋은 sample을 고를 수 있는가는 보장하지
않는다. 최신 initial-condition 실험은 prefix-only prior를 추가했지만,
그 prior가 plausible future distribution을 잘 나타내는지는
winner agreement 하나로 결정되지 않는다.

### 8.3 Horizon마다 새 noise를 넣으면 재분기가 가능하다

이전 compounding-noise 방식은 개념적으로

\[
\hat h_{j+1}=K\hat h_j+\epsilon_{j+1}
\]

인 forced stochastic dynamics에 가깝다. 하나의 noise tape 전체를
미리 뽑았다는 의미에서는 trajectory sample이지만, 매 horizon의
독립 innovation이 semantic mode를 다시 바꿀 수 있다.

autonomous dynamics에서 가장 강한 trajectory commitment는 초기
조건에 branch identity를 넣고 이후에는 같은 \(K\)로 전개하는
것이다.

\[
u_i=Kh_t+R_\phi(h_t,z_i),
\qquad
\hat h_{i,j}=K^{j-1}u_i
\]

또는

\[
h_t^{(i)}=h_t+R_\phi(h_t,z_i),
\qquad
\hat h_{i,j}=K^jh_t^{(i)}.
\]

이 구조에서 \(z_i\)는 전체 orbit의 branch를 고르고 \(K\)는 phase를
진행시킨다. 모든 power는 여전히 병렬 계산할 수 있다. 장기 미래에
필요한 추가 randomness가 있다면 horizon별 독립 noise보다 하나의
global trajectory code에서 \(R_j(z_i)\)를 결정적으로 생성하는 편이
commitment를 더 명확하게 보존한다.

최신 initial-condition 실험은 첫 번째 식을 구현했다. 이 구조는
branch distance를 보존하고 동일 token fixed point를 줄였지만,
step-1000 free generation에서는 4-position phase cycle을 보였다.
따라서 기하학적 branch persistence만으로 semantic trajectory
commitment가 보장되지는 않는다.

## 9. 현재 구현에서 이미 있는 것과 없는 것

현재 저장소의 최신 initial-condition 실험에는 다음이 이미 있다.

- reversible causal encoder와 exact inverse decoder
- 모든 문맥이 공유하는 전역 orthogonal \(K\)
- stride-1 anchor와 같은 online encoder pass의 미래 gold states
- \(K^1,\ldots,K^4\)에 대한 token CE
- 네 token CE 합으로 선택하는 세 개의 learned initial-condition
  trajectory
- context-conditioned branch code와 prefix-only prior
- initial residual의 \(K\)-propagation과 trajectory별 causal decoder
  tape
- clean h1 attached online relative MSE

반면 현재 실험에는 다음이 아직 없다.

- 128-token horizon까지 충분히 작은 one-step residual
- 128-step finite-horizon rollout에 대한 직접적인 학습 또는 인증
- clean one-step logits에 대한 직접적인 AR CE
- branch semantic state의 h1--h4 canonical alignment
- 같은 trajectory plan을 유지하는 공정한 AR/block comparator
- prefix prior의 distributional calibration 또는 sampling evidence
- 실제 \(K\)-residual이 decoder margin 안에 있는지 측정하는 지표

중요한 정정은 다음과 같다.

- 전역 \(K\)+stride-1 h1 regression에서는 local closure가 이미
  암묵적으로 들어 있으므로 별도 필요조건이 아니다.
- orthogonal \(K\)에서는 residual amplification stabilizer도 별도
  필요조건이 아니다.
- 문맥별 \(K_A\)는 표현력을 위한 대안이지 현재 철학의 누락된
  필수 구성요소가 아니다.

## 10. 현재의 최소 계약

현시점의 가장 작은 일관된 아키텍처 계약은 다음과 같다.

1. encoder는 실제 causal state 전체를 하나의 전역 shift action으로
   linearize한다.
2. 같은 orthogonal \(K\)가 모든 state에 작용한다.
3. stride-1 h1 regression으로 모든 local residual
   \(Kh_t-h_{t+1}\)을 작게 만든다.
4. 모든 학습 horizon의 token CE로 decoder decision을 직접
   감독한다.
5. multimodality는 \(K\)를 바꾸기보다 branch-specific initial
   condition 또는 shared trajectory code로 표현한다.
6. all-horizon latent loss와 residual-aligned tube loss는 128-step
   최적화가 h1 loss만으로 부족할 때 추가하는 수단이다.

이 계약에서는 `K의 작은 local error`, `orthogonal propagation`,
`decoder margin`, `trajectory initial condition`이 각자 다른 책임을
가진다. `P`, idempotence, simplex, 문맥별 \(K_A\)를 근거 없이 동시에
추가하지 않는다.

## 관련 논쟁 기록

이 문서는 현재 최소 계약을 요약한다. 해당 계약에 도달한 반박 과정,
step-1000 evidence의 범위, initial condition과 per-step innovation의
논쟁, semantic state와 remaining plan을 분리하는 아직 미구현 상태의
확장 시불변 연산자 제안은
[research_debate/README.md](research_debate/README.md)에서 이어진다.

## 11. 표준 AR의 병렬 고정점 표현

후속 CE-only 실험은 모델을 변경하지 않고도 greedy AR generation을
다음 병렬 fixed-point equation으로 쓸 수 있음을 확인했다.

\[
y_t=\arg\max p_\theta(\cdot\mid A,y_{<t}).
\]

후보 block 전체의 우변을 동시에 평가하는 Jacobi update는 동일
checkpoint의 AR 궤적에 정확히 수렴했다. \(K\)-orbit 초기화에서는
첫 token이 이미 AR과 동일하므로 길이 \(n\) block 전체가 늦어도
\(n-1\) iteration에 확정된다.

이는 “AR은 병렬 방식으로 정확히 표현할 수 없다”는 강한 불가능
주장을 기각한다. 정확한 표현과 계산상 가속은 다르다. 현재 update는
causal frontier를 iteration마다 한 token만 확정하므로 필요한
iteration이 \(O(n)\)이다.

따라서 이후 solver의 목표는 단순 exact 수렴이 아니라, 앞 prefix가
아직 모두 확정되지 않은 상태에서도 먼 위치의 결정을 안정화하여
한 iteration에 여러 causal dependency를 해결하는 것이다. 상세
실험과 증명은
[CE-only AR 병렬 고정점 기록](research_debate/05_parallel_ar_fixed_point_solver.md)에
둔다.
