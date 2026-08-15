# Trajectory commitment 논쟁

## 1. 왜 deterministic \(K\)만으로 multimodality가 문제가 되는가

causal prefix \(x_{\le A}\)가 주어졌을 때 다음 token과 이후 sequence는
대개 하나로 결정되지 않는다. 표준 MLE에서 모델은 조건부 분포를
학습하고, AR inference는 매 step 그 분포에서 새로운 결정을 내린다.

latent state regression을

\[
Kh_A\to h_B
\]

로만 보면 여러 가능한 \(h_B\)의 평균이나 중심에 가까워질 수 있다.
그 중심이 어떤 실제 token mode의 decision region에도 충분히 깊게
들어가지 않으면 deterministic rollout이 무너질 수 있다.

noise trajectory의 원래 의도는 이 평균 주변에 여러 proposal을 만들고
그중 gold future와 맞는 proposal에 gradient를 집중하여 mode-seeking을
유도하는 것이었다.

## 2. “noise”라는 이름 아래 섞여 있던 세 의미

논쟁에서 \(\eta\)는 세 가지 다른 수학적 역할로 사용됐다.

### 2.1 decoder robustness noise

```text
state + small corruption -> same token
```

decoder decision margin을 넓히기 위한 noise다. trajectory mode를
표현할 필요는 없다.

### 2.2 initial-condition code

```text
choose eta once
u1 = K hA + eta
uj = K^(j-1) u1
```

\(\eta\)가 전체 미래 orbit을 고른다. 이후에는 autonomous dynamics만
남는다.

### 2.3 per-step innovation

```text
u(j+1) = K uj + eta(j+1)
```

표준 stochastic transition처럼 각 step에 새로운 조건부 선택이
들어간다.

이 셋을 구분하지 않으면 “noise를 더 넣어야 한다”와 “한 번 고른
trajectory를 흔들면 안 된다”가 모순처럼 보인다. 실제로는 서로 다른
모델 계약이다.

## 3. 초기 compounding-noise 설계

과거 구현은 clean powers를 만들고 horizon마다 독립 Gaussian
innovation을 넣었다.

\[
\begin{aligned}
s_1 &= Kh_A,\\
s_2 &= K(s_1+\epsilon_1),\\
s_3 &= K(s_2+\epsilon_2),
\end{aligned}
\]

decoder는 각각 \(s_j+\epsilon_j\)를 받았다.

이 설계의 장점은 앞선 noise가 다음 horizon으로 전파된다는 점이다.
그러나 독립 \(\epsilon_j\)는 각 horizon에서 mode를 다시 바꿀 수 있다.
하나의 sampled sequence가 아니라 token-local branch들의 조합으로
최적화될 수 있다.

초기 100-step 비교에서는 learned sigma가 약 `0.05`에서 `0.010227`로
줄었고 h2/h3 성능 개선도 관측되지 않았다. 이 결과가 반증한 것은
“독립 Gaussian compounding noise + 당시 MSE 목적”이다. 상관된
trajectory tape나 initial-condition model 전체를 반증하지는 않는다.

## 4. trajectory-level CE가 추가한 것

세 trajectory 각각에 대해 네 horizon CE를 더한다.

\[
C_i=\sum_{j=1}^{4}\operatorname{CE}_{i,j}.
\]

그리고 하나의 detached responsibility를 trajectory 전체에 적용한다.

\[
q_i=\operatorname{sg}
\operatorname{softmax}_i(-C_i/\tau).
\]

이 변경은 중요한 의미를 가진다.

- horizon 1은 branch 1, horizon 2는 branch 2처럼 gold selector가
  직접 갈아타지 않는다.
- branch \(i\)는 원칙상 \(B,C,D,E\) 전체의 후보가 된다.
- 낮은 \(\tau\)는 best-of-\(N\)에 가까운 mode-seeking을 만든다.

하지만 이것만으로 joint semantics가 생기는 것은 아니다. 한 branch의
네 state가 각각 올바른 token class를 맞히기만 하면 CE 합은 낮아진다.
네 state가 하나의 일관된 proposition을 나타내는지는 별도 문제다.

## 5. 현재 initial-condition architecture

현재 step-1000 실험은 세 fixed branch code \(z_i\)와
context-conditioned initializer를 사용한다.

\[
\eta_i=R(\operatorname{sg}(h_A),z_i).
\]

prefix-only prior는

\[
\pi_i=P(\operatorname{sg}(h_A))
\]

를 만든다. residual은 prior-weighted centroid가 0이 되도록 중심화된다.

\[
u_{i,1}=Kh_A+\eta_i,
\qquad
u_{i,j}=K^{j-1}u_{i,1}.
\]

즉 branch를 처음에 한 번만 선택하고 후속 noise는 없다.

### 이 설계가 해결한 것

- branch identity가 horizon마다 새로 샘플되지 않는다.
- orthogonal \(K\)가 branch pairwise distance를 보존한다.
- 모든 horizon state를 한꺼번에 만들 수 있다.
- branch index 하나가 네 token CE 전체로 평가된다.
- prefix prior가 gold 없이 branch를 선택할 inference path를 제공한다.

### 이 설계가 해결하지 않은 것

- 하나의 residual orbit이 semantic future를 표현한다는 보장
- branch state가 canonical encoder state에 가까워야 한다는 계약
- clean AR path에 대한 직접 CE
- prefix prior가 arbitrary branch label이 아니라 유용한 mode mass를
  학습한다는 보장
- 4-token phase shortcut 방지
- 128-token plan capacity

## 6. \(h_A\) detach 논쟁

현재 initializer와 prior 입력은 `stopgrad(hA)`다.

### 도입 의도

- clean encoder/\(K\) geometry는 h1 state loss가 담당한다.
- branch initializer는 이미 만들어진 causal feature에서 residual과
  prior만 학습한다.
- branch loss가 encoder를 변형해 clean \(K\) 계약을 우회하는 것을
  막는다.
- 어떤 모듈이 개선을 만들었는지 attribution을 단순화한다.

### 비용

- encoder는 trajectory mode를 prior가 읽기 쉬운 feature로 스스로
  조직하는 gradient를 받지 않는다.
- initializer가 현재 representation에 적응해야 하며 공동 표현학습이
  제한된다.

따라서 detach는 정보 차단이 아니라 gradient routing 선택이다.
forward pass에서는 \(h_A\)의 모든 값이 그대로 들어간다.

### \(h_A\) 대신 \(Kh_A\)를 넣는 경우

현재 \(K\)는 orthogonal이므로 invertible하다. 따라서 정보량만 보면
\(h_A\)와 \(Kh_A\)는 동등하다.

\[
h_A=K^\top(Kh_A).
\]

입력을 \(Kh_A\)로 바꾸는 것은 새로운 미래 정보를 주는 것이 아니다.
대신 initializer가 residual을 놓아야 하는 **proposal geometry**와
더 직접 맞닿게 하고, detach를 제거하면 branch objective가 \(K\)와
encoder로 흐르는 경로를 바꾼다.

따라서 `KhA input`과 `detach 제거`는 별개의 ablation이어야 한다.

## 7. 사용자의 per-step correction 질문

제기된 식은 다음이었다.

\[
u_{i,1}=Kh_A+\eta_i(h_A)
\]

뒤에

\[
u_{i,2}=Ku_{i,1}+\eta_k(Ku_{i,1})
\]

를 적용해야 하지 않느냐는 것이었다.

직관은 명확하다.

- \(u_{i,1}\)이 \(h_B\) 역할을 한다면,
- AR은 \(B\) 뒤에서 다시 다음 mode를 선택한다.
- 그러므로 두 번째 transition에도 새로운 innovation이 있어야
  \(C\) mode를 고를 수 있지 않느냐는 질문이다.

이 질문은 버그 수정이 아니라 stochastic process의 정의를 바꾼다.

### fresh \(k\)를 매 step 선택하는 경우

- branch 수가 개념적으로 \(M^H\)가 된다.
- 어떤 \(k\)를 고를지 순차적으로 결정해야 한다.
- trajectory identity가 token-local branch identity로 바뀐다.
- state-dependent MLP가 실제 LM transition을 담당하고 \(K\)가
  장식이 될 수 있다.
- 닫힌 \(K^j\) rollout의 계산 이점이 약해진다.

이 설계는 AR과 더 비슷하지만 현재 아키텍처의 핵심 주장과는 다른
모델이다.

## 8. initial condition과 per-step innovation을 화해시키는 tape

한 번 고른 global code \(z_i\)에서 innovation sequence 전체를 미리
결정할 수 있다.

\[
(\eta_{i,1},\ldots,\eta_{i,H})=R(h_A,z_i).
\]

그 뒤

\[
u_{i,j}=Ku_{i,j-1}+\eta_{i,j}
\]

를 적용한다. 전개하면

\[
u_{i,j}
=K^jh_A+
\sum_{r=1}^{j}K^{j-r}\eta_{i,r}.
\]

모든 \(\eta_{i,r}\)가 처음부터 존재하므로 horizon들을 parallel scan
또는 batched power로 계산할 수 있다. branch index를 다시 고를
필요도 없다.

### arbitrary tape의 위험

큰 MLP가 \(H\)개의 full-width correction을 직접 출력하면

- 각 horizon의 정답 state를 MLP가 직접 생성할 수 있다.
- \(K\)는 identity나 phase counter로 퇴화할 수 있다.
- 128-token 출력 head가 사실상 작은 parallel LM이 된다.

따라서 tape 아이디어를 쓰더라도

- global code 하나
- 낮은 plan dimension
- horizon-shared dynamics
- low-rank injection
- correction energy budget

같은 구조적 제한이 필요하다.

## 9. 외부 분석의 initial-condition 비판

외부 분석은 다음 논증을 제시했다.

1. canonical \(h_B=E(A,B)\)는 \(C,D,E\)를 보지 못했다.
2. initial-condition branch \(u_{i,1}\)은 선택된 \(C,D,E\) plan을
   품어야 한다.
3. 그러므로 \(u_{i,1}\)은 canonical \(h_B\)와 같을 수 없다.
4. 따라서 re-anchoring consistency와 trajectory commitment는
   initial-condition-only state 안에서 충돌한다.
5. plan을 별도 tape나 side channel로 빼야 한다.

이 분석이 포착한 핵심은 강하다.

- decoder-visible current semantics
- 아직 소비되지 않은 future plan

을 한 residual에 동시에 담도록 요구하는 현재 parameterization에는
긴장이 있다.

### 과장된 부분 1: initial randomness model은 불가능하지 않다

AR의 모든 미래 randomness를 처음에 하나의 충분히 큰 seed로 미리
샘플할 수 있다. 이후 deterministic shift가 seed의 다음 부분을
순서대로 노출하면 stochastic process를 deterministic initial-value
system으로 표현할 수 있다.

따라서 causal \(h_A\)가 미래를 보지 못한다는 사실은
initial-condition model 자체를 반증하지 않는다. 미래 실현값은
\(h_A\)가 아니라 별도의 \(z_i\)가 담으면 된다.

### 과장된 부분 2: \(K\)가 conditional mean일 필요는 없다

MSE가 평균 해를 유도할 수 있다는 사실과 학습된 \(K\)가 문자 그대로
모든 state에서 conditional mean operator라는 주장은 다르다.
CE, encoder geometry, best-of-\(N\) branch가 함께 학습되므로
\(K\)의 실제 역할은 경험적으로 확인해야 한다.

### 과장된 부분 3: canonical equality는 필요조건이 아니다

\(u_{i,1}\neq h_B\)여도 같은 token decision region 안에 있고 이후
trajectory가 맞을 수 있다. canonical alignment는 강력한 안정화
전략이지만 token correctness의 논리적 필요조건은 아니다.

### 남는 유효 결론

initial-condition stochasticity는 가능하지만, **plan을
decoder-visible semantic residual 하나에 숨기는 현재 표현이 가장
좋은 parameterization이라는 보장은 없다.**

## 10. 4-phase code 가설

현재 initial residual은

\[
\eta_i,K\eta_i,K^2\eta_i,K^3\eta_i
\]

로 전개된다. orthogonal \(K\) 때문에 norm은 같고 방향만 변한다.
이 신호는 decoder가 horizon phase를 읽는 clock으로 쓰기 쉽다는
가설이 제기됐다.

### 이 가설을 지지하는 관측

- `prior_block4`의 lag-2 repeat는 `0.0640`으로 낮지만 lag-4 repeat는
  `0.4951`로 높다.
- block 내부 immediate repeat는 `0.2757`, re-encoding boundary는
  `0.0833`이다.
- 생성문에 slot-like four-position template가 반복된다.

### 단독 원인으로 확정할 수 없는 이유

- clean block-4는 residual이 없는데도 period-4 repeat가 `0.6617`로
  더 높다.
- decoder는 future position/RoPE를 통해 horizon 정보를 이미 받는다.
- 896차원 orthogonal \(K\)는 하나의 단순 회전 clock보다 훨씬 많은
  rotation plane을 가질 수 있다.
- all-horizon CE 자체가 phase-specific classification shortcut을
  허용한다.

따라서 가장 안전한 현재 해석은 다음이다.

> initial residual의 제한된 orbit과 horizon별 token CE가 함께
> semantic trajectory보다 phase-coded output을 쉽게 만들었을
> 가능성이 있다.

이는 **가설**이며 loss-only, residual-only 분리 ablation이 필요하다.

## 11. prefix prior의 올바른 해석

현재 prior winner accuracy는 gold future CE로 고른 posterior winner와
prefix-only prior argmax의 일치율이다.

이 지표는 chance `1/3`보다 높으면 branch label과 prefix 사이에
학습 가능한 관계가 생겼음을 보여 준다. 그러나 prefix에서 실제
held-out future를 결정론적으로 알아맞히는 것이 생성 모델의 목적은
아니다.

같은 prefix에서 여러 future가 타당할 수 있으므로 prior는 본질적으로

- gold branch classifier

가 아니라

- plausible trajectory modes 위의 probability mass

를 나타내야 한다.

따라서 이후 평가에서는 다음을 분리해야 한다.

- argmax branch의 sentence quality
- prior sampling의 quality/diversity
- mixture marginal NLL
- oracle posterior와 prior의 calibration

`prior_win` 하나를 trajectory model의 성공 여부로 해석하면 안 된다.

## 12. 현재 종합

현재까지의 논쟁은 다음 네 선택지를 구분했다.

| 설계 | branch 선택 | 후속 변화 | 병렬성 | 주된 위험 |
|---|---|---|---|---|
| clean \(K^j\) | 없음 | 같은 \(K\) | 최대 | mode averaging/fixed point |
| initial residual | 처음 한 번 | \(K^j\eta_0\) | 최대 | phase code, plan/semantic 혼합 |
| fresh innovation | 매 step | state-dependent \(\eta\) | 낮음 | rebranching, \(K\) 우회 |
| coherent tape | 처음 한 번 | fixed code가 \(\eta_j\) 생성 | 유지 가능 | tape generator가 LM을 대체 |

가장 유망한 방향은 마지막 두 안의 단순 절충이 아니다. semantic
state와 remaining plan을 분리하고, 둘을 하나의 확장된 시불변
operator로 전개하는 것이다. 그러면

- branch는 처음에 한 번만 선택하고,
- horizon마다 다른 effective innovation이 생기며,
- decoder는 canonical semantic 부분만 보고,
- 모든 horizon은 여전히 같은 operator의 power로 계산된다.

이 구조는 [확장된 시불변 연산자 제안](04_augmented_time_invariant_operator.md)에
정리한다.

