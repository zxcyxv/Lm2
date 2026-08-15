# 상태의존 latent temporal recurrence: 안정성 중심 재설계 근거

## 문서의 지위

이 문서는 현재 등록된 `unitary-branch-normalized-residual` producer의
구현 설명이 아니라, 그 다음 구조를 선택하기 위한 설계 판단이다. 아직
구현하거나 학습한 결과가 아니며 성능 주장을 하지 않는다.

기존 [gradient-path audit](../../../docs/recurrent_gradient_path_analysis.md)의 현재
그래프 미분, horizon 경로 수, hidden postnorm 반례는 그대로 유효하다.
다만 그 문서의 직접적인 hidden-postnorm 권고보다 한 단계 앞에서 다음을
먼저 결정한다.

> 중앙 재귀가 동일한 문제를 반복해서 푸는 고정점 iteration인지, 아니면
> 다음 미래 token 시점으로 이동하는 latent time evolution인지.

현재 H1--HN objective는 서로 다른 미래 token 위치를 순서대로 예측한다.
따라서 이 문서는 후자를 채택한다. 이 선택에서 후속 구조는 URM block의
복제가 아니라 **고정 context에 조건화된 안정적인 latent temporal
integrator**여야 한다.

## 결론

현재 구조에서 유지할 것은 unitary transport 자체이고, 버려야 할 것은
두 unitary carrier 사이의 blind additive feedback이다.

후속 구조의 원칙은 다음과 같다.

1. encoder가 만든 조건은 변하지 않는 context `c`로 별도 보존한다.
2. `z`와 `S`만 미래 token 시간에 따라 진행하는 dynamic state로 둔다.
3. `R z`와 `U S`는 시간 prediction이므로 유지한다.
4. context는 매 step control law에 제공하되 dynamic carrier에 raw
   addition으로 누적하지 않는다.
5. memory의 `+ k v*` write를 prediction error에 대한 innovation update로
   바꾼다.
6. hidden correction은 독립적인 `O(read)`가 아니라 memory와 같은
   innovation error에서 나온 adjoint/observer gain으로 만들고, tangent
   step과 recurrent-boundary postnorm으로 적용한다.
7. 안정성 판단 단위는 R, U, Q, K, V, O 각각이 아니라 complete joint map
   `(z,S) -> (z_next,S_next)`다.
8. state gradient의 exponential amplification은 제거해야 한다. 반면
   trainable time generator의 `O(N)` parameter sensitivity는 미래 시간 N을
   진행한 결과이므로 별개의, 원칙적으로 타당한 양이다.
9. 이 설계의 기준은 gradient norm을 사후에 줄이는 것이 아니다. memory는
   DeltaNet/normalized-LMS/Kaczmarz update와, complete cell은 unitary
   predictor와 innovation corrector로 이루어진 nonlinear observer와
   구조적으로 같아야 한다. gradient 안정성은 이 동형의 결과다.

## 1. 왜 이 재귀가 존재하는가

### 1.1 같은 답의 반복 개선이 아니다

URM형 recurrence에서는 같은 입력 `x`에 동일 block을 여러 번 적용해 한
문제의 해를 개선한다. iteration index는 계산량이지 데이터의 시간축이
아니다. 그래서 오래된 working state가 contract해도 고정 입력 `x`를
매번 다시 넣으면 조건이 사라지지 않는다.

현재 모델의 horizon index는 다르다.

```text
z_0 -> H1 token state -> H2 token state -> ... -> HN token state
```

각 step은 다른 미래 위치를 나타낸다. 같은 입력에 수렴하는 fixed-point
iteration으로 만들면 긴 rollout이 한 상태로 수렴하거나 주기적으로
고정되어야 하고, 이는 미래 token 시간의 진행이라는 목적과 충돌한다.

따라서 parameter sharing의 정당성도 "같은 solver를 반복한다"가 아니다.
같은 local time-evolution law를 모든 미래 위치에 적용함으로써 훈련
horizon 밖에서도 동일한 법칙을 사용할 수 있게 하는 것이다.

### 1.2 input reinjection의 올바른 의미

초기 encoder state를 `z_0`로 한 번 넣는 것과, 깨끗한 입력 조건을 계속
접근 가능하게 두는 것은 같지 않다. 첫 update 뒤의 `z_1`은 이미 초기
조건과 model-generated correction의 혼합물이다. 이후의 residual path는
깨끗한 입력을 재주입하는 것이 아니라 그 혼합물을 계속 운반한다.

반대로 매 step 다음과 같이 초기 state를 dynamic carrier에 더하는 것도
적절하지 않다.

```text
z_next = z + Block(z + z_0)
```

이 식은 시간 0의 상태와 현재 상태를 계속 누적하며, URM이 outer recurrent
residual을 제거한 이유와 같은 scale accumulation을 만든다.

올바른 분리는 다음과 같다.

```text
c   = normalize(Encoder(prefix, x_t))   # immutable condition
z_r = dynamic hidden at future time r
S_r = dynamic fast memory at future time r
```

각 step의 QKV와 correction network는 `(z_r,c)`를 함께 읽는다. `c`는
재인코딩할 필요 없이 한 번 계산해 cache한다. 이것은 Transformer가 다음
token을 만들 때 고정 prefix에 계속 접근하는 것과 같은 역할이며, dynamic
state에 동일 벡터를 반복 가산하는 것과는 다르다.

### 1.3 unitary가 context를 대체하지는 않는다

Unitary map은 이미 carrier 안에 든 벡터의 norm과 상대 위상을 보존한다.
다음은 보장하지 않는다.

- 초기 조건이 이후 nonlinear read/write 뒤에도 깨끗한 좌표로 남는 것
- decoder가 그 조건을 항상 짧은 경로로 접근할 수 있는 것
- model-generated correction이 초기 조건과 분리되어 있는 것

따라서 unitary transport와 immutable context는 경쟁 관계가 아니다.
전자는 dynamic time state를 운반하고, 후자는 매 시점의 transition을
조건화한다.

## 2. 현재 구조가 불안정한 동역학인 이유

등록된 중앙 step의 핵심은 다음과 같다.

```text
z_bar = R z
u     = RMS(z_bar)
q,k,v = QKV(u)

S_next = U S + k v*
read   = q* S_next / ||S_next||_F
delta  = O(read)
z_next = z_bar + delta
```

`R`과 `U`만 보면 두 transport는 lossless다. 그러나 complete system에는
다음 두 복사 경로가 동시에 있다.

```text
z perturbation은 R을 통해 z에 남으면서 write를 통해 S에도 복사된다.
S perturbation은 U를 통해 S에 남으면서 read를 통해 z에도 복사된다.
```

즉 한 carrier의 에너지를 다른 carrier로 옮기는 구조가 아니라, source에
그대로 보존한 채 destination에 새 항을 더한다. 두 보존계 사이의
state-dependent positive feedback에 damping이나 error subtraction이 없다.

이 때문에 R과 U가 각각 unitary여도 complete Jacobian은 unitary가 아니다.
한 step에서 생긴 expanding direction이 다음 step들과 충분히 정렬되면
gradient는 고정된 `a^N` 또는 변동하는 `a_1 ... a_N` 형태로 증가할 수
있다.

Q, K, V, O 각각이 잘못된 것이 아니다. 문제는 다음 관계가 없다는 데
있다.

- write가 이미 저장된 prediction의 error를 줄이는가
- read correction이 write와 반대 부호의 feedback을 이루는가
- 한 carrier에서 증가한 perturbation만큼 다른 carrier에서 감소하는가
- complete map이 어떤 joint energy에서 non-expansive한가

현재 RMS와 Frobenius normalization은 branch 출력 scale을 제한하지만 이
관계를 만들지는 않는다.

## 3. 현재 구조는 Kalman filter나 optimal control과 동형이 아니다

현재 memory update는 표면적으로 prediction과 correction처럼 보인다.

```text
prediction: U S
correction: + k v*
```

그러나 Kalman/observer correction의 본체는 다음 괄호다.

```text
state_next
  = predicted_state
  + gain * (observation - predicted_observation)
```

현재 `k v*`에는 `observation - predicted_observation`가 없다. 같은 key와
value가 반복되면 오차가 작아지는 것이 아니라 동일 write가 계속 더해진다.
또한 covariance, observation-noise model, Riccati update 또는 그에
상응하는 gain 선택이 없다. `q* S`는 외부 observation도 아니며 모델
자신이 만든 state의 내부 read다.

따라서 현재 모델을 Kalman filter라고 부르면 안정성을 제공하는 바로 그
구조를 누락한 채 predict/correct라는 모양만 비교하는 셈이다.

Optimal control과도 같은 이유로 동형이 아니다. 현재는 learned feedback
controller라고 부를 수는 있지만, 명시적인 cost-to-go, value/costate,
optimality equation, stabilizing feedback gain 또는 Lyapunov certificate가
없다.

학습이 어려운 이유는 Kalman/optimal-control 구조인데 optimizer가 못
찾아서가 아니다. **안정성을 만드는 innovation subtraction과 negative
feedback이 아키텍처에 존재하지 않기 때문**이다.

## 4. memory를 innovation observer로 바꾸는 이유

`*`를 complex conjugate adjoint라고 하자. 후속 memory step은 다음과
같이 구성한다.

```text
S_bar  = U S_r
v_pred = k* S_bar
error  = v - v_pred
S_next = S_bar + beta k error*
```

여기서 `||k|| = 1`, `0 < beta <= 1`로 둔다. `k`와 `v`가 고정된 한
step에서 memory perturbation은 정확히 다음처럼 이동한다.

```text
dS_next = (I - beta k k*) U dS
```

이 식의 의미는 명확하다.

- U는 모든 memory perturbation의 norm을 보존한다.
- current key 방향은 `1-beta`만큼 남는다.
- current key와 직교한 방향은 그대로 보존된다.
- 어느 방향도 memory update 자체 때문에 1보다 크게 증폭되지 않는다.

`beta=1`이면 current key 방향의 old value를 새 target value로 교체한다.
blind additive write처럼 같은 항을 N번 누적하지 않는다.

### 4.1 DeltaNet, normalized LMS, Kaczmarz와의 정확한 동형

다음 prediction-error energy를 두자.

```text
E(S_bar) = 0.5 * ||v - k* S_bar||^2
```

위 memory update는 이 energy의 normalized gradient step과 정확히 같다.
complex conjugation과 행/열 표기 방향만 맞추면 다음 식이다.

```text
negative_gradient_S E = k error*
S_next = S_bar - beta * gradient_S E
```

이것은 fast-weight matrix가 현재 key에서 예측한 value와 실제 write
target의 차이만 수정하는 DeltaNet의 delta rule이다. 따라서 이 부분은
새로운 안정화 휴리스틱이 아니라 실제 recurrent language model에서 쓰는
검증된 memory law와 같은 연산이다.

`||k||=1`, `beta=1`이면 더 강한 등식이 성립한다.

```text
k* S_next = v
```

즉 `S_next`는 `k* S = v`라는 affine constraint 위로 `S_bar`를 정사영한
결과다. 이것이 Kaczmarz projection이다. `0 < beta < 2`이면 current-key
prediction error는 `|1-beta|`배가 되며 반드시 감소한다. 본 후보가
`beta <= 1`을 쓰는 이유는 overshoot까지 제외한 보수적인 구간을 택하기
위해서다.

### 4.2 Kalman/Luenberger observer와의 대응

선형 observer의 본체는 다음 세 줄이다.

```text
x_bar  = A x
error  = observation - C x_bar
x_next = x_bar + gain * error
```

memory subsystem의 대응은 다음과 같다.

```text
A           -> U
C           -> memory observation S -> k* S
observation -> v
gain        -> beta k
```

따라서 `beta k`는 unit key와 isotropic uncertainty를 가정한 normalized
observer/LMS gain이다. covariance를 유지하고
`gain = P C* (C P C* + noise)^-1`로 계산할 때만 exact Kalman/RLS라고
부른다. covariance 없이 모양만 비슷한 현재 additive write를 Kalman이라고
부르지 않는 것과, normalized delta update를 exact Kalman이라고 과장하지
않는 것을 동시에 지킨다.

### 4.3 보존계와 소산계의 operator splitting

Unitary U도 이 구조에서 더 명확한 역할을 갖는다. U는 correction 전에
memory를 다음 시간으로 predict하고, innovation은 그 prediction 중 현재
key로 관측된 오차만 수정한다. lossless prediction과 dissipative correction
역할이 분리된다.

전체적으로 보면 `R/U` step은 norm을 보존하는 conservative/Hamiltonian
prediction이고, innovation step은 명시된 error energy를 감소시키는
dissipative correction이다. 이것은 보존 dynamics와 소산 dynamics를
순서대로 적용하는 predictor-corrector operator splitting이다. 따라서
gradient 안정성을 별도 장치로 발명하는 것이 아니라, 이미 Lyapunov
energy가 있는 passive observer 구조를 채택하고 그 결과로 recurrent
Jacobian의 증폭 경로를 제거한다.

## 5. hidden carrier를 bounded temporal step으로 바꾸는 이유

`R z`는 단순 Transformer skip이 아니다. correction이 0일 때도 미래
latent state를 다음 시간으로 운반하는 homogeneous dynamics다. 이를
삭제하고 `z_next = delta`로 만들면 width-4352 state가 width-496 readout
image로 붕괴한다. 따라서 R transport는 유지해야 한다.

문제는 다음 raw addition이다.

```text
z_next = R z + delta
```

R이 이전 hidden energy를 모두 보존하므로 bounded delta도 horizon에 따라
누적된다. decoder와 QKV가 hidden의 방향만 사용한다면 이 radial energy는
표현 정보가 아니라 암묵적인 step-size history다.

후속 update는 임의의 `Controller`가 아니라 memory와 같은 innovation
energy의 hidden derivative를 사용해야 한다. `error`가 `z_bar`에도
의존하므로 다음 reference form을 둘 수 있다.

```text
z_bar = R z_r

J_z   = derivative(error, z_bar)
delta = -AdjointSolve(J_z, error)
delta_tangent
      = delta
        - z_bar * inner(z_bar, delta) / ||z_bar||^2

z_next = RMS(z_bar + eta * delta_tangent)
```

가장 단순한 `AdjointSolve`는 `J_z* error` 방향의 normalized gradient
step이고, 더 강한 reference는 damped Gauss-Newton/Kalman gain이다.

```text
delta = -J_z* (J_z J_z* + lambda I)^-1 error
```

이렇게 해야 hidden과 memory가 같은 prediction error를 줄이는 하나의
nonlinear observer가 된다. 현재의 독립적인 `O(q* S)`를 그대로 hidden
state에 더하면 DeltaNet/Kalman 동형은 memory에서 끝나고 complete
`(z,S)` cell에는 positive-feedback loop가 다시 남는다. `q* S` read는
decoder observation으로 사용할 수 있지만, recurrent hidden correction을
구동하려면 같은 innovation energy의 gain 안으로 들어와야 한다.

이 구조를 선택하는 이유는 다음과 같다.

- R이 제공하는 full-width temporal carrier를 유지한다.
- correction이 carrier norm을 직접 누적하지 않고 방향만 바꾼다.
- anti-aligned correction이 postnorm 분모를 0에 가깝게 만드는 경로를
  tangent projection이 제거한다.
- 현재 decoder가 읽는 ray와 다음 recurrence가 받는 ray를 일치시킨다.
- `eta`가 innovation correction의 명시적인 시간 step이 되어 raw carrier
  norm이 암묵적인 step-size 역할을 하지 않게 한다.
- tangent projection과 RMS는 sphere 위 observer correction의 retraction이
  된다. 따라서 postnorm은 독립적인 안정화 장치가 아니라 state manifold로
  되돌리는 Riemannian update의 마지막 연산이다.

이 postnorm은 arbitrary activation normalization이 아니다. 현재 graph가
이미 QKV 직전과 decoder 직전에 hidden scale을 버린다는 사실에서 나온다.
scale을 어느 소비자도 정보로 사용하지 않는다면 recurrent carrier에만
남겨 누적시키는 것보다 매 boundary에서 quotient하는 편이 의미와 구현을
일치시킨다.

다만 forward norm 고정만으로 complete gradient 안정성이 증명되지는
않는다. `error`, `J_z`, gain과 memory correction을 모두 포함한 joint map이
다음 절의 조건을 만족해야 한다.

## 6. residual, prenorm, postnorm의 역할 분리

### 6.1 유지하는 residual

- 유한한 encoder/decoder block 내부 residual
- central step 내부의 homogeneous temporal transport `R z`

이 둘은 각각 finite-depth feature processing과 zero-correction time
evolution을 정의한다.

### 6.2 제거하는 residual

다음과 같은 최소 재귀 단위 바깥의 추가 skip은 두지 않는다.

```text
z_next = z + RecurrentBlock(z, context, S)
```

`RecurrentBlock` 안에 이미 `R z` transport가 있으므로 이 식은 old state를
두 번 보존한다. context를 raw addition으로 재주입하면서 이 outer skip까지
두면 URM이 피한 것과 같은 forward norm accumulation을 다시 만든다.

### 6.3 정규화 위치

후속 구조의 normalization 계약은 다음과 같다.

| 대상 | 계약 | 이유 |
|---|---|---|
| immutable context `c` | encoder 뒤 한 번 RMS | 모든 시간에서 같은 조건 scale |
| recurrent hidden `z` | complete step 뒤 post-RMS | 사용되지 않는 radial scale 제거 |
| Q/K/V 공통 입력 | 별도 반복 prenorm 제거 | `z`와 `c`가 이미 boundary-normalized |
| projected Q와 K | unit norm | address magnitude와 similarity gain 분리 |
| projected V | fixed RMS 또는 bounded scale | write target의 무제한 forcing 방지 |
| memory S | global postnorm 없음 | past write의 상대 성분과 carrier 보존 |
| memory read | fixed dimension scale | state-dependent Frobenius divisor 의존 제거 |

현재 QKV prenorm은 raw hidden carrier를 유지하는 동안에는 cubic scale
feedback을 막기 위해 필요했다. recurrent hidden 자체를 postnormalize하면
같은 normalization을 branch 직전에 다시 적용할 구조적 이유가 사라진다.
forward 값이 거의 같더라도 중복 normalization Jacobian은 backward map을
바꾼다.

Q와 K는 address이므로 projection 뒤의 unit normalization이 여전히
필요하다. V는 content지만 현재 설계에서 magnitude를 별도 정보로
등록하지 않았으므로 bounded scale이 맞다.

Innovation memory가 S의 addressed component를 교체하면 current
`/ ||S||_F` read normalization도 필수가 아니다. 이 divisor는 memory가
채워질수록 모든 read를 약하게 만들고 작은 S 근처에서는 큰 derivative를
만든다. q를 unit norm으로 하고 memory update 자체를 안정화한 뒤 fixed
dimension scale을 쓰는 편이 dynamics를 더 직접적으로 해석할 수 있다.

## 7. complete gradient 안정성 기준

joint state를 다음처럼 둔다.

```text
y_r = (z_r, S_r)
y_next = F(y_r; context)
```

요구해야 할 것은 개별 unitary matrix가 아니라 reachable state에서의
incremental stability다.

```text
distance_M(F(y1;c), F(y2;c))
    <= distance_M(y1,y2)
```

미분 가능한 지점에서는 같은 요구를 다음처럼 쓸 수 있다.

```text
J* M J <= M
```

여기서 M은 hidden과 memory의 단위가 다른 것을 반영하는 positive joint
metric이다.

후속 구조는 이 조건을 역할별로 만든다.

- R과 U: neutral, lossless temporal prediction
- memory innovation: 관측된 key 방향에서 contractive correction
- hidden tangent/postnorm step: radial accumulation 제거
- bounded eta와 beta: 한 step correction gain 제한
- immutable context: recurrent Jacobian에는 새 identity path를 추가하지
  않으면서 각 미래 시점의 조건을 제공

마지막으로 `z -> correction -> z/S` coupling은 서로 독립된 positive
shear가 아니라 같은 prediction error의 negative-gradient 또는
Gauss-Newton pair여야 한다. memory의 `beta k error*`와 hidden의
`-AdjointSolve(J_z,error)`가 같은 energy에서 나오면 complete cell은
`unitary predictor + nonlinear observer corrector`가 된다. 임의의 O를
그대로 두고 postnorm만 추가하면 이 동형과 최종 조건은 모두 성립하지
않는다.

여기서 "이미 잘 되는 아키텍처와 동형이면 안정성이 따라온다"는 말의
`동형`은 단순한 블록 모양의 유사성이 아니다. 다음 항까지 같아야 한다.

- 무엇을 prediction state로 보는가
- 어떤 observation residual을 빼는가
- correction gain이 residual의 adjoint/covariance와 어떻게 연결되는가
- prediction 단계의 보존량과 correction 단계의 Lyapunov 감소량이 무엇인가

이 네 항이 맞을 때 gradient 안정성은 설계 목적이 아니라 observer error
dynamics의 결과가 된다.

따라서 향후 Jacobian audit도 `||R||`, `||U||`, QKV scale을 따로 보고
결론내리지 않는다. complete `(z,S)` JVP/VJP와 joint metric의 증분을
검사해야 한다.

## 8. exponential state gradient와 O(N) parameter gradient

### 8.1 제거해야 하는 항

state adjoint는 다음 recurrent product를 지난다.

```text
gradient_at_0
  = J_0* J_1* ... J_(N-1)* gradient_at_N
```

joint map에 지속적인 gain `a > 1`이 있으면 크기는 `a^N`으로 증가한다.
이것은 미래 시간이 길어질수록 가까운 두 latent trajectory가 임의로
분리된다는 뜻이므로 제거 대상이다.

현재 구조는 두 neutral carrier가 perturbation을 서로 복사하기 때문에
이 경로를 허용한다. Innovation subtraction은 memory의 addressed 방향을
non-expansive하게 만들고, tangent/postnorm update는 hidden radial
accumulation을 제거한다. complete coupling까지 joint non-expansive하게
묶으면 exponential factor가 사라진다.

### 8.2 남을 수 있고, 일부는 남아야 하는 항

trainable unitary time generator를 N번 쓰면 phase parameter의 sensitivity는
정확히 time N에 비례할 수 있다.

```text
z_N = R(theta)^N z_0
partial z_N / partial theta = O(N)
```

이것은 activation이나 state gradient explosion이 아니다. token-time
transition의 각도나 속도를 조금 바꾸면 N step 뒤 위치 차이가 N에
비례하는 정상적인 시간 민감도다.

공유 parameter gradient도 각 사용 위치의 기여를 더하므로 neutral
transport에서 최악 `O(N)`이 남는다. H1--HN CE를 평균내면 loss 수에서 온
단순 N배는 제거하지만, 각 loss가 그 이전의 여러 shared uses를 통과하는
triangular path까지 자동으로 상쇄하지는 않는다.

이 O(N)을 어떻게 해석할지는 recurrence 의미에 달려 있다.

- URM처럼 같은 문제를 더 오래 반복했을 뿐이면 arbitrary unroll
  sensitivity이므로 contraction이나 equilibrium formulation으로 없애는
  것이 맞다.
- 현재 모델처럼 N이 실제 미래 token 거리이면 time generator의 O(N)은
  타당한 sensitivity다.

따라서 목표는 모든 gradient를 억지로 `O(1)`로 만드는 것이 아니다.
`a^N` state amplification을 없애고, 시간 생성자의 해석 가능한 `O(N)`만
남기는 것이다.

만약 향후 요구가 "훈련 horizon을 4에서 512로 바꿔도 generator parameter
update까지 완전히 같은 scale"이라면 선택지는 제한된다. neutral transport
generator를 고정하거나, per-use time increment와 optimizer scale을
effective horizon에 맞춰 명시적으로 정규화해야 한다. 이는 stability
수정이 아니라 시간 parameterization 선택이며, learned token-time law의
민감도를 일부 포기하는 tradeoff다.

## 9. 제안하는 complete step

후속 후보를 한곳에 모으면 다음과 같다.

```text
# one-time, immutable condition
c   = RMS(Encoder(prefix, x_t))
z_0 = InitialState(c)
S_0 = 0

for r in 0 ... N-1:
    # temporal prediction
    z_bar = R z_r
    S_bar = U S_r

    # context-conditioned bounded addresses/content
    u = Condition(z_bar, c)
    q = unit(W_Q u)
    k = unit(W_K u)
    v = RMS(W_V u)

    # one shared innovation residual
    v_pred = k* S_bar
    error  = v - v_pred

    # DeltaNet / normalized-LMS / Kaczmarz memory correction
    S_next = S_bar + beta k error*

    # read is an observation/decoder branch, not an arbitrary recurrent skip
    read  = q* S_next / fixed_dimension_scale

    # nonlinear-observer hidden correction from the same error energy
    J_z   = derivative(error, z_bar)
    delta = -AdjointSolve(J_z, error)

    # Riemannian direction-changing hidden time step
    delta_tangent
          = delta
            - z_bar * inner(z_bar, delta) / ||z_bar||^2
    z_next = RMS(z_bar + eta * delta_tangent)
```

이 식은 `Controller`를 임의의 MLP로 남겨두지 않는다. reference semantics는
같은 innovation error의 adjoint 또는 damped Gauss-Newton/Kalman gain으로
확정한다. 아직 남은 구현 질문은 이 reference를 효율적으로 계산하면서
state-dependent 표현력을 보존하는 parameterization이다. 효율화를 위해
독립 MLP gain으로 바꾸는 순간 구조적 동형과 안정성 근거를 다시 증명해야
한다.

> `error`를 줄이는 하나의 negative-feedback energy update로 hidden과
> memory coupling을 묶으면서 현재의 state-dependent 표현력을 얼마나
> 유지할 수 있는가.

## 10. 왜 이 설계가 현재 목적에 더 타당한가

이 설계는 normalization을 많이 넣어서가 아니라 각 연산의 의미가
future-token time dynamics와 맞기 때문에 타당하다.

- unitary R/U는 correction이 없을 때의 time prediction을 담당한다.
- immutable context는 초기 조건을 손실 없이 계속 제공하되 dynamic
  state에 누적되지 않는다.
- normalized Q/K는 주소, bounded V는 memory target이라는 역할을 갖는다.
- innovation update는 DeltaNet/NLMS/Kaczmarz와 동일하게 같은 정보를 더하는
  대신 prediction error를 줄인다.
- hidden correction도 같은 error의 observer gain에서 나오므로 memory와
  hidden이 독립적인 positive feedback을 만들지 않는다.
- hidden postnorm은 실제 소비자가 사용하지 않는 radial scale만 제거한다.
- tangent correction은 carrier를 파괴하지 않고 미래 방향을 바꾼다.
- outer recurrent residual을 추가하지 않아 old state를 이중 보존하지
  않는다.
- joint incremental stability를 목표로 하므로 N=512에서 제거해야 할
  exponential path와 남겨도 되는 time-linear sensitivity를 구분한다.

반대로 recurrence가 미래 token 시간이 아니라 같은 block 해의 반복
개선이라는 실험으로 바뀐다면 이 결론도 바뀐다. 그 경우에는 URM처럼
동일 입력을 최소 재귀 단위에 직접 재주입하고, outer residual을 제거하며,
working state를 contractive하게 만드는 것이 맞다. 두 설계를 섞어서
"unitary이므로 reinjection이 필요 없다" 또는 "URM이 안정적이므로 초기
state를 매 시간 더한다"고 결론내리지 않는다.

## 11. 향후 검증 기준

이 문서는 후속 experiment manifest에서 다음을 사전등록할 근거다.

1. 동일 seed/data/order에서 additive write와 innovation write를 단독 비교
2. hidden raw residual과 tangent postnorm step을 단독 비교
3. context 없음, raw additive context, separate conditioning을 구분
4. H1--H4 CE뿐 아니라 N에 따른 joint-state JVP/VJP gain을 측정
5. state-gradient exponential gain과 shared-parameter O(N) sum을 별도 보고
6. H4로 훈련한 뒤 H16, H64, H512 rollout의 finite state, NLL, generation
   degeneration을 구분
7. transient gnorm peak 하나를 구조적 증거로 사용하지 않고, matched
   trajectory와 complete joint Jacobian evidence를 함께 보존
8. `beta=1`에서 `k* S_next = v` projection identity와 normalized-delta
   reference implementation의 forward/backward equivalence를 단위 테스트
9. hidden correction을 independent O, adjoint gradient, damped Gauss-Newton
   gain으로 나눠 complete joint energy가 실제로 감소하는 범위를 보고

이 검증 전까지 후속 구조는 **이론적으로 동기가 부여된 후보**이며,
경험적으로 우월하다고 기록하지 않는다.
