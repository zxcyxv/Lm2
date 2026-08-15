# 확장된 시불변 연산자 제안

## 1. 상태

- **분류:** 제안
- **구현:** 없음
- **학습 evidence:** 없음
- **목적:** 현재 initial-condition model과 coherent innovation tape
  논쟁을 원래의 global-\(K\) 철학 안에서 종합

이 문서는 experiment manifest가 아니다. 실제 구현과 실행 전에는
별도 manifest에 질문, matched control, seed, split, parameter budget,
성공 기준을 사전등록해야 한다.

## 2. 출발 명제

현재 아키텍처의 가장 큰 이점은 문맥이나 horizon마다 다른 transition
network를 실행하는 데 있지 않다. encoder가 token shift를 하나의
전역 시불변 latent action으로 linearize하면

\[
Kh_t\approx h_{t+1}
\]

라는 local law 하나로

\[
K^jh_t\approx h_{t+j}
\]

를 얻을 수 있다는 데 있다.

따라서 다음 설계가 피해야 할 것은 명확하다.

- state-dependent \(K(h)\)
- horizon마다 별도 transition MLP
- token마다 새 branch search
- decoder 앞의 반복 canonicalization
- future token 수만큼 순차 nonlinear computation

동시에 현재 initial residual의 문제도 해결해야 한다.

- current token semantics와 remaining future plan이 한 residual에
  섞인다.
- 이후 가능한 branch signal은
  \(\eta,K\eta,K^2\eta,\ldots\)로 제한된다.
- branch identity는 보존돼도 semantic innovation sequence가
  충분하지 않을 수 있다.
- decoder가 residual orbit을 horizon phase code로 사용할 수 있다.

## 3. 핵심 전환: 연산자가 아니라 state를 확장한다

sampled trajectory가 시작된 뒤의 완전한 state를 두 부분으로 나눈다.

\[
s_j=(c_j,m_j).
\]

- \(c_j\): 현재 token prefix를 나타내는 decoder-visible semantic state
- \(m_j\): 아직 소비되지 않은 trajectory plan

초기에는

\[
c_0=h_A,
\qquad
m_{i,0}=G(h_A,z_i)
\]

로 둔다. \(z_i\)는 trajectory code이며 처음에 한 번만 선택한다.

그 뒤에는 모든 문맥과 horizon에서 같은 transition을 적용한다.

\[
\begin{aligned}
c_{j+1}&=Kc_j+Bm_j,\\
m_{j+1}&=Sm_j.
\end{aligned}
\]

이를 하나의 block operator로 쓰면

\[
\begin{bmatrix}
c_{j+1}\\
m_{j+1}
\end{bmatrix}
=
\underbrace{
\begin{bmatrix}
K&B\\
0&S
\end{bmatrix}
}_{\bar K}
\begin{bmatrix}
c_j\\
m_j
\end{bmatrix}.
\]

\(\bar K\)는 하나의 전역 시불변 operator다.

\[
s_j=\bar K^j s_0.
\]

decoder는 \(c_j\)만 받는다.

\[
c_j\xrightarrow{D=E^{-1}}\text{LM head}\to x_{A+j}.
\]

plan memory \(m_j\)는 decoder에 들어가지 않는다.

## 4. 이 구조가 논쟁을 화해시키는 방식

### 4.1 시불변 \(K\)의 power라는 핵심 유지

후속 correction이 있더라도 외부에서 horizon별 MLP를 호출하는 것이
아니다. \(B\)와 \(S\)는 \(\bar K\)의 고정 sub-block이다. 모든 future
state는 여전히 같은 operator의 power다.

### 4.2 branch는 처음에 한 번만 선택

\(m_{i,0}\)가 trajectory 전체를 고른다. horizon 2에서 새 \(k\)를
선택하지 않으므로 \(M^H\) branch tree가 없다.

### 4.3 horizon마다 다른 effective innovation

전개하면

\[
c_j
=K^jh_A+
\sum_{r=0}^{j-1}K^{j-1-r}BS^rm_{i,0}.
\]

따라서 step \(r\)의 effective innovation은

\[
\eta_{i,r}=BS^rm_{i,0}
\]

이다. 모두 하나의 initial plan에서 나오지만 매 horizon 같은
residual의 단순 회전일 필요는 없다.

### 4.4 canonical semantics와 remaining plan의 분리

첫 token에서

\[
c_1\approx h_B
\]

로 정렬할 수 있다. \(C,D,E\)에 대한 commitment는 \(c_1\)에 숨길
필요 없이 \(m_1\)에 남는다.

따라서 전체 sampled state는

\[
(h_B\text{에 가까운 }c_1,\;m_1)
\]

이고, 같은 \(\bar K\)를 적용하면 다음 state로 전진한다.

이것이 다음 두 요구를 동시에 만족한다.

- exact inverse decoder는 canonical semantic 좌표를 받는다.
- 미래 trajectory identity는 token decoding 후에도 사라지지 않는다.

## 5. stochastic AR을 initial-value system으로 보는 관점

finite-horizon AR process의 innovation들을

\[
\epsilon_1,\ldots,\epsilon_H
\]

라고 하자. 이 randomness 전체를 시점 0에 하나의 tape로 미리
샘플하고, 고정 shift operator가 매 step 다음 \(\epsilon_j\)를
노출하게 만들 수 있다.

즉 “매 step stochastic”과 “처음만 random, 이후 deterministic”은
state를 충분히 확장하면 서로 배타적이지 않다.

```text
sample all future randomness once
             |
             v
[semantic state, unread random tape]
             |
      same shift every step
```

현재 제안의 \(m_0,S,B\)는 이 deterministic dilation을 압축해
학습하는 형태다.

- 충분히 큰 explicit tape는 임의 finite-horizon innovation을 표현할
  수 있다.
- 작은 \(m\)과 선형 \(S\)는 그 tape에 low-dimensional structure가
  있다고 가정한다.
- encoder와 nonlinear exact inverse decoder는 언어 dynamics가 이
  lifted linear system에서 표현되도록 좌표를 학습한다.

따라서 핵심 연구 질문은 “\(K\)가 동적이어야 하는가”가 아니라 다음이
된다.

> 128-token trajectory의 stochastic commitment를 얼마나 작은 plan
> state와 얼마나 단순한 시불변 dynamics로 압축할 수 있는가?

## 6. 현재 initial-condition model과의 관계

현재 방식은

\[
u_{i,j}=K^jh_A+K^{j-1}\eta_i
\]

다.

확장 구조는

\[
c_{i,j}
=K^jh_A+
\sum_{r=0}^{j-1}K^{j-1-r}BS^rm_{i,0}
\]

다.

현재 방식에는 별도의 plan memory가 없고, 한 residual이 이후 모든
horizon signal을 담당한다. 확장 구조에서는 plan이 \(S\) 아래
진행되며 필요한 부분이 매 step \(B\)를 통해 semantic state로
주입된다.

따라서 이 제안은 arbitrary per-horizon MLP보다 현재 구조에 더
가깝고, current initial residual보다 표현력이 높다.

## 7. 구조적 제한

확장 state가 새로운 LM으로 \(K\)를 우회하지 않게 하려면 다음 제한이
필요하다.

### 7.1 작은 plan dimension

첫 실험은 `32` 또는 `64` dimensions 정도로 제한한다. full-width
future state를 horizon별로 직접 저장하지 못하게 한다.

### 7.2 stable \(S\)

\[
\|S\|_2\le1
\]

로 제한한다. 간단한 첫 선택은 fixed random orthogonal \(S\)다.
학습된 \(S\)를 쓰더라도 orthogonal 또는 spectral constraint가
필요하다.

### 7.3 low-rank \(B\)

\[
B=UV^\top
\]

형태로 rank `8`--`16` 정도를 사용한다. plan이 semantic width 전체를
자유롭게 덮어쓰지 못하게 한다.

### 7.4 correction energy budget

\[
\rho_j=
\frac{\|Bm_j\|_{\mathrm{RMS}}}
{\|Kc_j\|_{\mathrm{RMS}}}
\]

를 기록하고 작은 범위로 제한한다. current residual RMS ratio `0.05`를
matched initialization으로 사용할 수 있다.

### 7.5 horizon-specific parameters 금지

\(G\), \(B\), \(S\)에 horizon index별 별도 head를 두지 않는다.
position 정보가 필요한 경우에도 fixed \(S^j\)가 운반하게 한다.

### 7.6 decoder에 plan 직접 입력 금지

decoder가 \(m_j\)를 직접 attention하면 semantic transport를 건너뛸
수 있다. 첫 버전에서는 decoder 입력을 \(c_j\)로 제한한다.

## 8. orthogonality에 대한 선택

\(K\)와 \(S\)를 각각 orthogonal로 두고 \(B\neq0\)로 만들면 위의
upper-triangular \(\bar K\) 전체는 일반적으로 orthogonal이 아니다.
이를 숨기면 안 된다.

첫 실험에서 가장 실용적인 계약은 다음이다.

- semantic error transport \(K\): orthogonal
- plan transport \(S\): orthogonal 또는 non-expansive
- plan-to-semantic injection \(B\): low-rank, small norm
- horizon: finite and preregistered

두 branch plan의 차이에 대해

\[
\|\Delta c_j\|
\le
j\|B\|\|\Delta m_0\|
\]

같은 선형 finite-horizon bound를 얻을 수 있다. exponential
amplification은 피하면서 branch effect가 시간에 따라 semantic
state에 나타나는 것을 허용한다.

전체 \(\bar K\)를 skew generator의 exponential로 만들어 orthogonal
coupling을 강제하는 대안도 가능하지만, 첫 실험에는 계산량과 해석
복잡도가 크다. 이 안은 initial ablation이 성공한 뒤 검토한다.

## 9. 학습 objective

### 9.1 clean local closure

현재 stride-1 h1 relative MSE를 유지한다.

\[
\mathcal L_{\mathrm{clean\ state}}
=d(Kh_t,h_{t+1}).
\]

이는 plan이 없는 center dynamics의 계약이다.

### 9.2 clean one-step AR CE

clean proposal에도 직접 token CE를 준다.

\[
\mathcal L_{\mathrm{clean\ CE}}
=\operatorname{CE}
\bigl(\operatorname{Head}(D(Kh_t)),x_{t+1}\bigr).
\]

현재 checkpoint에서 빠졌던 standard one-step readout anchor다. 이
항이 있어야 내부 AR comparator가 실제 학습 경로가 된다.

### 9.3 trajectory path CE

현재처럼 한 branch의 모든 horizon CE를 합쳐 하나의 responsibility를
사용한다.

\[
C_i=\sum_{j=1}^H\operatorname{CE}_{i,j}.
\]

branch index는 horizon 중간에 바뀌지 않는다.

### 9.4 selected semantic alignment

\[
\mathcal L_{\mathrm{semantic}}
=
\sum_i q_i\sum_{j=1}^H
d(c_{i,j},\operatorname{sg}(h_{t+j})).
\]

이 loss의 목적은 매 step error를 iterative하게 고치는 것이 아니다.
두 state channel의 역할을 식별 가능하게 만드는 것이다.

- 현재 token semantics는 \(c_j\)
- 남은 future commitment는 \(m_j\)

selected branch만 gold path에 정렬하면 다른 branch가 모두 같은
future로 collapse하도록 강제하지 않으면서, phase-only decoder code를
제한할 수 있다.

### 9.5 prior objective

현재 posterior/prior KL을 첫 matched ablation에서는 유지한다. 단,
평가에서는 다음을 분리한다.

- prior argmax
- prior sampling
- marginal mixture likelihood
- winner agreement

held-out gold future는 prefix에서 결정론적으로 알 수 없으므로
winner agreement를 유일한 성공 지표로 사용하지 않는다.

## 10. 공정한 AR/block comparator

이 제안의 중요한 이점은 같은 stochastic plan으로 AR과 block을
비교할 수 있다는 점이다.

### 공통 초기화

\[
c_0=h_A,\qquad m_0=G(h_A,z_i).
\]

같은 \(z_i\), 같은 prior 선택, 같은 \(m_0\)를 두 정책에 사용한다.

### block path

\[
(c_j,m_j)=\bar K^j(c_0,m_0)
\]

를 open-loop로 만들고 \(c_1,\ldots,c_H\)를 병렬 decode한다.

### AR re-anchored path

한 step의 \(\bar K\)를 적용하여 token을 생성한 뒤:

1. semantic part만 실제 생성 prefix의 encoder state로 교체한다.
2. plan part \(m_{j+1}=Sm_j\)는 그대로 유지한다.
3. 같은 \(\bar K\)를 다음 step에 적용한다.

```text
block:
    (c, m) --Kbar--> (c1, m1) --Kbar--> (c2, m2) ...

AR:
    (c, m) --Kbar--> token1
       semantic c1만 E(actual prefix)로 re-anchor
       m1은 그대로 유지
                  --Kbar--> token2 ...
```

두 정책의 유일한 차이는 semantic re-anchoring이다. branch를 다시
고르거나 plan을 재생성하지 않는다.

이때 다음이 원래 연구 질문을 직접 측정한다.

- same-plan AR/block token agreement
- horizon별 logit KL
- \(c_j\)와 re-anchored encoder state의 latent distance
- 128-token까지 decision divergence가 시작되는 horizon

## 11. 첫 ablation 설계

실행 전에 별도 manifest가 필요하지만, 논리적으로 가장 작은 비교는
다음이다.

### 공통 조건

- 동일 WikiText-103 split과 validation starts
- seed 1337
- width, reversible blocks, exact inverse decoder 동일
- global orthogonal semantic \(K\) 동일
- trajectories 3, horizon 4
- effective batch와 update schedule 동일
- 총 parameter budget을 가능한 한 맞춤
- direct clean CE를 control과 proposal 양쪽에 동일하게 추가

### A: current initial residual

\[
u_{i,j}=K^jh_A+K^{j-1}\eta_i.
\]

### B: augmented plan state

\[
c_{j+1}=Kc_j+Bm_j,\qquad m_{j+1}=Sm_j.
\]

### B0: no-plan control

\[
B=0.
\]

semantic \(K\)만으로 얻는 개선과 plan contribution을 분리한다.

### 선택적 factor

semantic all-horizon alignment 유무를 별도 factor로 둔다. architecture
변경과 loss 변경을 한 번에 묶으면 phase 감소의 원인을 알 수 없다.

## 12. 성공 기준의 방향

정확한 threshold는 manifest에서 control 수치를 기준으로 고정해야
한다. 최소한 다음 축은 모두 필요하다.

### Optimization

- clean h1 state MSE 악화 없음
- validation marginal NLL 악화 없음
- nonzero plan usage
- \(K\) gradient와 plan gradient 모두 존재

### Role separation

- selected \(c_j\)의 gold-state distance 감소
- plan을 shuffle하면 coherent path CE가 악화
- \(B=0\)이면 branch benefit이 사라짐
- \(K\)를 identity/freeze한 shortcut이 같은 성능을 내지 못함

### Generation

- same-plan block-4가 same-plan AR의 distinct-2와 token distribution을
  유지
- period-4 excess와 exact block repeat 감소
- same-token collapse가 current initial-condition control보다 늘지
  않음
- within-block와 boundary 통계 차이 감소

### Scaling

- H=4에서 통과한 뒤 H=16, 32, 64, 128 curriculum
- horizon 증가에 따른 logit KL와 state error curve
- encoder calls와 wall time
- plan dimension 대비 성능

## 13. 반증 기준

다음이 관측되면 제안의 핵심 가정이 지지되지 않는다.

1. \(B m_j\)가 \(Kc_j\)보다 커져 사실상 plan LM이 된다.
2. plan dimension을 줄이면 즉시 성능이 무너져 explicit future tape와
   다를 바 없어진다.
3. selected semantic alignment를 줘도 \(c_j\)가 canonical state에서
   멀어진다.
4. same-plan AR도 문법적 coherence를 만들지 못한다.
5. block과 same-plan AR의 격차가 H=4부터 크게 벌어진다.
6. 4-phase excess가 current control보다 줄지 않는다.
7. \(B=0\) control과 차이가 없어 plan state를 사용하지 않는다.
8. dynamic or horizon-specific shortcut 없이는 optimization되지 않는다.

## 14. 이 제안이 추가하지 않는 것

첫 버전에는 다음을 넣지 않는다.

- canonicalizer \(P\)
- idempotence loss
- simplex/ALR mapping
- state-dependent \(K(c)\)
- per-step branch resampling
- horizon-specific decoder
- iterative denoising
- residual-aware correction network

이들을 동시에 추가하면 어떤 mechanism이 필요한지 다시 알 수 없게
된다.

## 15. 최종 설계 철학

이 제안이 바꾸는 것은 “시불변 \(K\)로 충분하다”는 명제가 아니다.
바꾸는 것은 \(K\)가 작용해야 할 완전한 state의 정의다.

canonical encoder state \(h_B\)는 현재 prefix의 semantic state다.
sampled trajectory가 계속 진행되려면 아직 소비되지 않은 randomness도
state에 포함돼야 한다. 이를 semantic coordinate 안에 억지로
숨기지 않고 별도의 plan coordinate로 둔다.

그 결과:

- decoder는 여전히 canonical semantic state를 받는다.
- trajectory randomness는 처음에 한 번만 선택된다.
- 같은 전역 시불변 operator가 모든 horizon을 전개한다.
- 후속 innovation은 새 의사결정이 아니라 initial plan의 결정론적
  unfolding이다.
- 모든 horizon은 병렬 계산 가능하다.

즉 다음 단계의 핵심 문장은 이것이다.

> \(K\)를 동적으로 만들 필요는 없다. 대신 sampled future를
> Markov하게 만드는 데 필요한 state를 빠짐없이 \(K\)의 domain에
> 포함시켜야 한다.

