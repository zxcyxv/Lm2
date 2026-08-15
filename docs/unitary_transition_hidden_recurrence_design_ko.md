# 단일 unitary transition layer의 hidden recurrence 확장 설계

## 문서 범위

이 문서는 [제미나이대화내용.txt](제미나이대화내용.txt)의 단일 레이어를
hidden-state recurrence로 확장할 때 보존할 의미와 Jacobian 안정 조건을
정리한 설계 메모다.

- checkpoint나 기존 실험 수치를 설계 근거로 사용하지 않는다.
- 단일 레이어의 대수와 recurrence의 미분 구조만 사용한다.
- 아래 식은 현재 구현을 기술하는 것이 아니라 권장 설계를 기술한다.

## 1. 원래 단일 레이어의 계약

입력 \(x_t\)가 외부에서 주어지는 단일 레이어는 다음과 같다.

\[
\begin{aligned}
q_t&=q(x_t),\\
k_t&=k(x_t),\\
v_t&=v(x_t),\\
S_{t+1}&=US_t+k_tv_t^\dagger,\\
o_t&=\operatorname{Re}(q_t^\dagger S_{t+1}).
\end{aligned}
\]

기호의 의미는 다음과 같다.

| 기호 | 의미 |
|---|---|
| \(q_t\) | 현재 상태가 memory에서 찾으려는 complex query |
| \(k_t\) | 현재 write의 complex address |
| \(v_t\) | address에 결합되는 value |
| \(S_t\) | 과거 rank-one write가 중첩된 complex key-value operator |
| \(U\) | 과거 write의 시간 위상을 이동하는 unitary operator |
| \(q_t^\dagger S_t\) | query와 memory의 위상 간섭을 이용한 측정 |

\(U^\dagger U=I\)이므로 homogeneous transport는

\[
\lVert US_t\rVert_F=\lVert S_t\rVert_F
\]

를 만족한다. 전개식은

\[
S_t=\sum_{n=1}^{t}U^{t-n}k_nv_n^\dagger
\]

이며, 각 write의 상대 시간 위상과 선형 중첩이 보존된다.

### 1.1 보존된다고 말할 수 없는 것

unitary인 것은 \(US_t\)이지 additive write까지 포함한 전체 update가
아니다.

\[
\lVert US_t+k_tv_t^\dagger\rVert_F
\ne
\lVert S_t\rVert_F
\]

일 수 있다. 예를 들어 \(U=I\)이고 모든 write가 \(W\)로 같으면

\[
S_t=tW
\]

이므로 raw memory norm은 선형으로 증가한다.

또한 일반적인 \(S_t=k_tv_t^\dagger\)의 합은 Hermitian,
positive-semidefinite, trace-one을 보장하지 않는다. 따라서 \(S_t\)는
물리적 density matrix라기보다 complex key-value operator로 해석한다.
위상 이동, 중첩, 보강·상쇄 간섭 해석은 그대로 유지된다.

## 2. Hidden recurrence에서 새로 생기는 문제

원래 \(x_t\)는 외부 입력이므로 write가 이전 memory의 함수가 아니다.

\[
\frac{\partial S_{t+1}}{\partial S_t}=U,
\qquad
\lVert U\rVert_2=1.
\]

hidden을 재귀시키면 다음 폐루프가 생긴다.

\[
\begin{aligned}
S_{t+1}&=US_t+w(h_t),\\
h_{t+1}&=F(h_t,S_{t+1}),\\
w(h_t)&=k(h_t)v(h_t)^\dagger.
\end{aligned}
\]

\(F_h,F_S,w_h\)를 각 함수의 Jacobian이라 하면 결합 state
\((h_t,S_t)\)의 Jacobian은

\[
J_t=
\begin{bmatrix}
F_h+F_Sw_h & F_SU\\
w_h & U
\end{bmatrix}.
\]

여기에는 두 feedback edge가 있다.

\[
h_t\xrightarrow{\,w_h\,}S_{t+1},
\qquad
S_{t+1}\xrightarrow{\,F_S\,}h_{t+1}.
\]

두 edge가 연결되면

\[
S_t\rightarrow h_{t+1}\rightarrow S_{t+1}
\]

이라는 closed loop가 된다. \(F_h\)와 \(U\)가 각각 unitary여도 전체
block Jacobian은 unitary가 아니며 singular value가 \(1\)보다 커질 수
있다.

memory read를 \(\lVert S\rVert_F\)로 나누는 것은 activation의 크기를
제한하지만 Jacobian을 자동으로 제한하지 않는다. 정규화

\[
\widehat S=\frac{S}{\lVert S\rVert_F}
\]

의 미분은 \(1/\lVert S\rVert_F\)에 비례하므로 작은 memory 근처에서는
민감할 수 있다.

## 3. 재귀 확장에서 보존할 의미

재귀 구조는 다음 네 역할을 분리해야 한다.

1. \(U\): 과거 write의 시간 위상을 이동한다.
2. \(kv^\dagger\): 현재 의미를 rank-one operator로 memory에 더한다.
3. \(q^\dagger S\): 중첩된 memory를 위상 간섭으로 측정한다.
4. hidden transition: 측정값이 기존 semantic carrier의 진행을
   조절한다.

측정값이 전체 hidden을 새로 만들어서는 안 된다.

\[
h_{t+1}=O(m_t)
\]

에서는 낮은 차원의 measurement가 전체 hidden을 재구축해야 하므로
기존 carrier가 사라진다.

권장 구조에서는 전체 hidden을 계속 전달하고 measurement는 그 hidden에
적용할 변환만 결정한다.

## 4. 권장 forward recurrence

### 4.1 Unitary hidden transport

\[
\bar h_t=Rh_t,
\qquad
R^\top R=I.
\]

\(R\)은 학습된 pairwise rotation, structured orthogonal matrix 또는
그 곱으로 구성한다.

Q/K/V 입력은 carrier 자체를 변경하지 않고 별도 branch에서
normalization한다.

\[
x_t=N(\bar h_t).
\]

### 4.2 Bounded Q, K, V

각 projection의 방향과 크기를 분리한다.

\[
\begin{aligned}
\widehat q_t&=
\frac{q_{\rm raw}(x_t)}
{\sqrt{\lVert q_{\rm raw}(x_t)\rVert^2+\tau_q^2}},\\
\widehat k_t&=
\frac{k_{\rm raw}(x_t)}
{\sqrt{\lVert k_{\rm raw}(x_t)\rVert^2+\tau_k^2}},\\
\widehat v_t&=
\frac{v_{\rm raw}(x_t)}
{\sqrt{\lVert v_{\rm raw}(x_t)\rVert^2+\tau_v^2}}.
\end{aligned}
\]

\[
q_t=a^q_t\widehat q_t,\qquad
k_t=a^k_t\widehat k_t,\qquad
v_t=a^v_t\widehat v_t.
\]

\(a^q_t,a^k_t,a^v_t\)는 sigmoid 등의 bounded 함수로 생성한다.
그러면

\[
\lVert k_tv_t^\dagger\rVert_F
=\lVert k_t\rVert_2\lVert v_t\rVert_2
\le a^k_{\max}a^v_{\max}.
\]

\(\tau_q,\tau_k,\tau_v>0\)는 작은 projection norm에서 normalization
Jacobian이 발산하지 않도록 하는 smooth floor다.

### 4.3 Unitary memory transport와 superposition

\[
S_{t+1}=US_t+\beta k_tv_t^\dagger,
\qquad
U^\dagger U=I.
\]

\(\beta\)는 bounded write scale이다. 모든 시점에 같은 상수
\(\beta\)를 사용하면

\[
S_t=\beta\sum_{n=1}^{t}U^{t-n}k_nv_n^\dagger
\]

이므로 write 사이의 상대 위상과 상대 가중치는 바뀌지 않는다.

raw memory를 감쇠하지 않으므로 strict superposition 의미는 유지되지만,
무한 horizon에서 raw norm이 bounded하다는 보장은 없다. bounded raw
memory를 무한히 요구한다면 damping, carrier normalization 또는
energy-conserving write로 계약을 변경해야 한다.

### 4.4 Smooth projective measurement

\[
m_t=
\frac{
\operatorname{Re}(q_t^\dagger S_{t+1})
}{
\sqrt{d_k}\,
\sqrt{\lVert S_{t+1}\rVert_F^2+\rho_0^2}
}.
\]

\(\rho_0>0\)는 \(S\approx0\)에서 measurement Jacobian이 발산하지 않도록
한다. 분모는 전체 중첩에 공통인 양의 실수이므로 다음 의미는 유지된다.

- \(q\)와 \(k\) 사이의 복소 위상차
- \(U^{t-n}\)이 부여한 상대 시간 위상
- write 사이의 보강 및 상쇄 간섭
- query와 일치하는 value 방향

제거되는 것은 raw memory의 전체 scale이다.

### 4.5 Measurement-conditioned orthogonal hidden update

measurement에서 bounded angle을 만든다.

\[
a_t=\eta_{\max}\tanh(Gm_t).
\]

이 angle로 orthogonal operator \(Q(a_t)\)를 구성한다.

\[
Q(a_t)^\top Q(a_t)=I.
\]

예를 들면

\[
Q(a)=
M^\top
\operatorname{blockdiag}
\bigl(
R(a_1),\ldots,R(a_{D/2})
\bigr)
M,
\qquad
M^\top M=I.
\]

hidden successor는

\[
\boxed{
h_{t+1}=Q(a_t)\bar h_t
=Q(a_t)Rh_t.
}
\]

따라서 forward hidden norm은

\[
\lVert h_{t+1}\rVert_2=\lVert h_t\rVert_2
\]

로 보존된다. decoder-facing state가 필요하면

\[
p_{t+1}=N(h_{t+1})
\]

을 별도로 사용한다.

measurement 차원이 hidden 폭보다 작아도 measurement는 hidden을
재구성하지 않는다. measurement는 \(Q\)의 angle만 결정하고 전체
\(D\)차원 carrier는 \(Rh_t\)를 통해 그대로 전달된다.

## 5. 248차원과 496차원의 의미

memory가 head \(8\)개, head당 value \(31\)개라면 complex read는

\[
8\times31=248
\]

개의 complex scalar다.

원문의

\[
\operatorname{Re}(q^\dagger S)
\]

를 그대로 사용하면 measurement는 \(248\)차원 실수 벡터다.

실수부와 허수부를 모두 사용하면

\[
[\operatorname{Re}(q^\dagger S),
\operatorname{Im}(q^\dagger S)]
\in\mathbb R^{496}.
\]

이 구조는 두 quadrature를 모두 읽는다. 따라서 다음 중 하나를
명시적으로 선택해야 한다.

| readout | 실수 차원 | 의미 |
|---|---:|---|
| 실수부만 사용 | \(248\) | 원문의 단일 homodyne-style measurement |
| 실수부와 허수부 사용 | \(496\) | 두 quadrature를 모두 보존하는 measurement |

## 6. Gradient 경로 제어

### 6.1 첫 설계: feedback edge 하나만 stop-gradient

forward recurrence의 값은 그대로 유지하면서 Q/K/V 입력에 대해서만
이전 hidden의 backward edge를 끊는다.

\[
(q_t,k_t,v_t)
=f(\operatorname{sg}(x_t);W).
\]

\(\operatorname{sg}\)는 forward에서는 항등 함수다. backward에서는
\(x_t\)로 향하는 gradient만 \(0\)으로 만들며 projection parameter
\(W\)의 gradient는 유지한다.

이때 학습에 사용되는 state Jacobian은

\[
\widetilde J_t=
\begin{bmatrix}
Q_tR & C_t\\
0 & U
\end{bmatrix}.
\]

\(C_t\)는 memory measurement가 hidden rotation을 바꾸는 미분이다.
두 diagonal block은 orthogonal 또는 unitary이고, 아래쪽 feedback
block이 \(0\)이다.

반복 곱에서도 diagonal은 norm \(1\)을 유지한다. off-diagonal은
bounded \(C_t\)의 합으로 남으므로 최악의 누적은 horizon에 대해
선형이며, 두 feedback edge가 반복해서 곱해지는 지수 증폭 경로는
사라진다.

이 방식에서 잘리는 것은 hidden 변화가 이후 Q/K/V 생성에 미치는
장기 credit이다. 다음 gradient는 남는다.

- 각 step의 Q/K/V projection parameter gradient
- memory를 통한 과거 write의 gradient
- measurement controller \(G\)의 gradient
- orthogonal carrier \(Q_tRh_t\)를 통한 hidden gradient

### 6.2 Full BPTT를 사용할 경우

stop-gradient를 제거한 true Jacobian을

\[
J_t=D_t+E_t
\]

로 나눈다. \(D_t\)는 block-unitary transport이고 \(E_t\)는
state-dependent write/read coupling이다.

\[
\lVert D_t\rVert_2=1,\qquad
\lVert E_t\rVert_2\le\kappa_t
\]

이면

\[
\left\lVert
\prod_{t=1}^{H}J_t
\right\rVert_2
\le
\prod_{t=1}^{H}(1+\kappa_t)
\le
\exp\left(\sum_{t=1}^{H}\kappa_t\right).
\]

허용할 최대 Jacobian 증폭을 \(C\)로 정하면 finite horizon \(H\)에서

\[
\sum_{t=1}^{H}\kappa_t\le\log C
\]

를 설계 조건으로 둔다.

\(\kappa_t\)는 다음 값의 상한으로 제어한다.

- angle 범위 \(\eta_{\max}\)
- write scale \(\beta\)
- \(G,W_Q,W_K,W_V\)의 spectral norm
- Q/K/V amplitude 상한
- normalization floor \(\tau_q,\tau_k,\tau_v,\rho_0\)

상수 \(\kappa>0\)인 완전한 feedback을 무한히 반복하면
\(\exp(H\kappa)\) 상한도 계속 커진다. 임의의 horizon에서 true
Jacobian을 균일하게 제한하려면 다음 중 하나가 추가로 필요하다.

- feedback coupling을 시간에 따라 summable하게 감소
- memory 또는 별도 controller에 contraction 도입
- write와 read를 skew-adjoint한 joint unitary coupling으로 변경
- backward feedback edge를 절단

앞의 세 방법은 원래의 감쇠 없는 additive-superposition 계약 일부를
변경한다.

## 7. 권장 최소 계약

첫 구현의 최소 계약은 다음과 같다.

1. \(U\)의 unitary temporal phase transport를 유지한다.
2. \(kv^\dagger\)의 rank-one additive superposition을 유지한다.
3. Q/K/V를 방향과 bounded amplitude로 분해한다.
4. measurement에 smooth projective normalization을 사용한다.
5. measurement가 hidden을 대체하지 않고 orthogonal transform의 angle을
   결정하게 한다.
6. 전체 hidden carrier는 \(Q(a_t)Rh_t\)로 전달한다.
7. 초기 backward에서는 \(h_t\rightarrow Q/K/V\) edge만 끊는다.
8. full BPTT가 필요하면 finite-horizon Jacobian budget을 명시한다.

이 계약에서 memory는 과거 write의 위상 중첩을 담당하고, query는
간섭 측정을 담당하며, measurement는 기존 semantic carrier의 unitary
진행을 조절한다. 낮은 차원의 measurement가 전체 hidden을 재구축하는
책임은 갖지 않는다.
