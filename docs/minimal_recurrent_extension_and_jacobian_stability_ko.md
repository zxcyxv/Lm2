# Unitary Transition SSM의 최소 재귀 확장과 Jacobian 안정성

## 문서 상태와 범위

이 문서는 [원래 구성요소 분석](./original_unitary_transition_ssm_component_analysis_ko.md)을
근거로 hidden-state recurrence를 추가하는 최소 설계를 정리한다.

- 현재 구현을 기술하지 않는다.
- 실험으로 검증된 결론이 아니라 수식에서 도출한 설계 제안이다.
- 원래 memory의 의미를 보존하는 최소 closure와, 더 강한 Jacobian 상한을
  위해 필요한 추가 조건을 구분한다.
- 실수 value와 Re-only read 복원은 의미론적 수정이지만, gradient 안정성
  개입과 분리해 검증할 수 있다.

## 1. 핵심 통찰

원래 구조에서 이미 안정적인 부분은 unitary memory carrier다.

\[
S_t=US_{t-1}+W_t,
\qquad
W_t=k_tv_t^\top,
\qquad
\lVert U\rVert_2=1.
\]

주어진 write tape에 대한 직접 memory Jacobian은 \(U\)다. hidden
recurrence가 새로 추가하는 것은 다음 두 결합이다.

\[
z_t\longrightarrow W_t\longrightarrow S_t,
\]

\[
S_t\longrightarrow m_t\longrightarrow z_{t+1}.
\]

따라서 안정화를 위해 먼저 제어할 대상은 \(S\) carrier가 아니라
`z -> write`와 `read -> next z`의 결합 gain이다.

## 2. 원래 구조에 부족한 closure

committed AR state \(z_t\)에서 원래 연산을 수행하면

\[
q_t=Q(z_t),\qquad
k_t=K(z_t),\qquad
v_t=V(z_t),
\]

\[
S_t=US_{t-1}+k_tv_t^\top,
\]

\[
m_t=\operatorname{Re}(q_t^\dagger S_t)
\]

까지 정의된다. 단일 레이어에 없는 것은 측정 \(m_t\)으로 다음 committed
state를 만드는 규칙뿐이다.

\[
(z_t,m_t)\longrightarrow z_{t+1}.
\]

따라서 최소 확장은 \(U\), additive write, raw \(S\), interference read를
바꾸지 않고 hidden closure만 추가해야 한다.

## 3. 최소 의미 보존형 recurrence

hidden의 norm-preserving transport를

\[
\bar z_t=Rz_t,
\qquad
R^\top R=I
\]

로 둔다. strict-minimum에서는 \(R=I\)로 둘 수 있다.

Q/K/V는 carrier를 변경하지 않는 별도 branch normalization을 사용한다.

\[
u_t=N_b(\bar z_t),
\]

\[
q_t=Q(u_t),\qquad
k_t=K(u_t),\qquad
v_t=V(u_t).
\]

원래 memory update와 read는 그대로 유지한다.

\[
S_t=US_{t-1}+k_tv_t^\top,
\]

\[
m_t=rac{\operatorname{Re}(q_t^\dagger S_t)}{\sqrt{d_k}}.
\]

\(1/\sqrt{d_k}\)는 실현된 state 크기에 의존하지 않는 고정 차원
스케일이다. 원래 식을 문자 그대로 유지하려면 생략할 수 있다.

가장 단순한 hidden closure는

\[
z_{t+1}=N_z(\bar z_t+\alpha Om_t)
\]

이다. 측정은 전체 carrier를 대체하지 않고 다음 committed state에 대한
residual correction으로만 작용한다. full BPTT를 유지하며 carry detach는
사용하지 않는다.

이 식은 최소 의미 보존형 비교군이지만, 그것만으로 전체 Jacobian의 강한
상한이 증명되지는 않는다. \(Om_t\)가 \(-\bar z_t\) 방향으로 접근하면
postnorm 입력의 크기가 작아질 수 있기 때문이다.

## 4. hidden postnorm을 정당화하는 tangent backaction

hidden norm을 의미 없는 gauge 또는 고정된 state radius로 취급한다면,
measurement correction을 \(\bar z_t\)의 접공간에 제한할 수 있다.

먼저 raw measurement correction을

\[
d_t=Om_t
\]

로 두고 radial component를 제거한다.

\[
d_t^\perp
=d_t-ar z_t
\frac{\langle\bar z_t,d_t\rangle}
{\lVert\bar z_t\rVert_2^2}.
\]

그러면

\[
\langle\bar z_t,d_t^\perp\rangle=0.
\]

방향을 보존하는 radial cap을

\[
C_c(x)=\frac{x}{\sqrt{1+\lVert x\rVert_2^2/c^2}}
\]

로 정의하고

\[
\delta_t=C_c(d_t^\perp)
\]

로 둔다. 최종 state는

\[
z_{t+1}
=\lVert\bar z_t\rVert_2
\frac{\bar z_t+\alpha\delta_t}
{\lVert\bar z_t+\alpha\delta_t\rVert_2}
\]

이다.

직교성 때문에

\[
\lVert\bar z_t+\alpha\delta_t\rVert_2^2
=\lVert\bar z_t\rVert_2^2
+\alpha^2\lVert\delta_t\rVert_2^2
\]

이고 따라서

\[
\lVert\bar z_t+\alpha\delta_t\rVert_2
\geq\lVert\bar z_t\rVert_2.
\]

measurement residual이 carrier를 상쇄할 수 없으므로 postnorm 분모에
양의 하한이 생긴다. \(\alpha c/\lVert z\rVert\)는 한 committed AR
step에서 허용되는 최대 방향 변화의 각도 예산으로 해석할 수 있다.

이 정당화는 hidden의 norm이 의미 정보를 담지 않는다는 전제를 필요로
한다. hidden norm 자체가 semantic variable이라면 postnorm과 tangent-only
update를 별도 아키텍처 개입으로 취급해야 한다.

## 5. memory carrier를 정규화하지 않는 이유

원래 memory 전개는

\[
S_t=\sum_{n=1}^{t}U^{t-n}k_nv_n^\top
\]

이다. 매 write 뒤 \(S_t\) 전체를 post-normalize하면 과거 write가 이후
모든 step에서 다시 스케일되므로 이 식이 성립하지 않는다. 과거와 현재
write의 상대 계수도 경계 normalization history에 의존하게 된다.

또한 raw update가 상쇄되어 작아지면 memory postnorm의 미분에는 작은
분모가 나타날 수 있다. 따라서 forward memory 크기를 고정하는 것과
memory Jacobian을 비증폭적으로 만드는 것은 같은 문제가 아니다.

동일한 이유로 realized \(\lVert S_t\rVert_F\)를 별도 read 분모로 사용하는
것도 원래 linear measurement를 0-homogeneous nonlinear measurement로
바꾸며, 작은 memory 근처에 \(1/\lVert S_t\rVert_F\) 민감도를 만든다.

최소 의미 보존형에서는 다음을 사용하지 않는다.

- \(S\) carrier postnorm
- realized \(\lVert S\rVert_F\) read normalization
- carry detach

H4에서 필요하면 state-dependent 분모 대신 write 경계의 고정 예산을
사용한다.

## 6. write와 read 경계의 방향 보존형 bound

각 head의 Q/K/V에 radial cap을 적용한다.

\[
q_t=C_{c_q}(Q(u_t)),
\qquad
k_t=C_{c_k}(K(u_t)),
\qquad
v_t=C_{c_v}(V(u_t)).
\]

이 cap은 complex vector의 모든 성분에 같은 양의 실수 배율을 적용하므로
방향, 채널 간 상대 크기와 위상을 보존한다. 작은 입력에서는 항등에
가깝고 다음 상한을 갖는다.

\[
\lVert q_t\rVert_2\leq c_q,
\qquad
\lVert k_t\rVert_2\leq c_k,
\qquad
\lVert v_t\rVert_2\leq c_v.
\]

그러면

\[
\lVert k_tv_t^\top\rVert_F\leq c_kc_v
\]

이고 unitary transport 아래에서

\[
\lVert S_t\rVert_F\leq t c_kc_v.
\]

고정 horizon \(H=4\)에서는

\[
\lVert S_t\rVert_F\leq4c_kc_v
\]

이며 read도

\[
\lVert m_t\rVert_2
\leq\frac{4c_qc_kc_v}{\sqrt{d_k}}
\]

로 유계다. memory 중첩 계수는 바꾸지 않고 사건 하나가 주입할 수 있는
최대 크기만 제한한다.

출력 activation bound만으로 parameter Jacobian까지 제한되지는 않는다.
강한 상한이 필요하면 \(W_Q,W_K,W_V,W_O\)의 spectral norm도 제한하거나
적어도 직접 측정해야 한다. 특히 cap은 projection 출력이 0에 가까운
구간에서 거의 항등이므로, 그 구간의 미분에는 projection operator norm이
그대로 나타난다.

## 7. 결합 Jacobian

한 step의 write 미분, memory read 미분과 hidden backaction 미분을 각각

\[
B_t=\frac{\partial W_t}{\partial z_t},
\qquad
D_t=\frac{\partial m_t}{\partial S_t},
\qquad
C_t=\frac{\partial z_{t+1}}{\partial m_t}
\]

로 둔다. query가 \(z_t\)에 직접 의존하는 경로를 \(E_t\), measurement를
고정했을 때의 hidden carrier 미분을 \(A_t\)라고 두면

\[
\delta S_t=U\delta S_{t-1}+B_t\delta z_t,
\]

\[
\delta m_t
=D_t\delta S_t+E_t\delta z_t,
\]

\[
\delta z_{t+1}
=A_t\delta z_t+C_t\delta m_t.
\]

결합 state \((z,S)\)의 block Jacobian은

\[
J_t=
\begin{bmatrix}
A_t+C_t(E_t+D_tB_t) & C_tD_tU\\
B_t & U
\end{bmatrix}.
\]

\(A_t\)와 \(U\)가 각각 등척이어도 off-diagonal coupling 때문에 전체
\(J_t\)가 자동으로 unitary가 되지는 않는다. 새 recurrent feedback의
핵심 loop gain은

\[
\lVert C_t\rVert_2
\lVert D_t\rVert_2
\lVert B_t\rVert_2
\]

다. 각 경계의 역할은 다음과 같다.

- bounded \(k,v\): write와 \(B_t\) 규모 제어
- bounded \(q\): memory-read 미분 \(D_t\) 규모 제어
- bounded tangent correction: backaction 미분 \(C_t\) 규모 제어
- unitary \(U,R\): 두 carrier의 직접 경로를 등척으로 유지
- projection spectral bound: activation cap 사이의 국소 Jacobian 제어

bidirectional coupling을 유지하면서 전체 block의 모든 singular value를
정확히 1 이하로 만들려면 augmented-state 전체를 skew-adjoint 또는
orthogonal map으로 다시 설계해야 한다. 이는 additive write라는 원래
구조의 최소 확장이 아니다. 고정 H4의 현실적 안정 조건은 직접 carrier를
등척으로 두고 모든 coupling을 유계로 만들어 geometric feedback
amplification을 제한하는 것이다.

## 8. \(U\)의 입력 의존성

최소 안정형에서는 \(U\)를 global shared unitary로 유지한다. step별

\[
U_t=U(z_t)
\]

를 사용하면 memory 미분에

\[
\frac{\partial U(z_t)}{\partial z_t}S_{t-1}
\]

항이 추가되고

\[
S\longrightarrow z\longrightarrow U(z)\longrightarrow S
\]

라는 별도 feedback loop가 생긴다. 입력 의존 transition을 나중에
검토한다면 immutable root \(z_0\)에서 \(U(z_0)\)를 한 번 생성해 rollout
전체에서 고정하는 방식이 먼저다. 조건부로는 여전히

\[
\frac{\partial S_t}{\partial S_{t-1}}=U(z_0)
\]

가 unitary다.

## 9. detach와 gradient norm의 지위

각 \(z_t\)가 새로운 committed AR 사건이면 미래 CE가 이전 state와
write를 생성한 규칙에 신용을 할당하는 것은 의미상 정상적인 BPTT다.
forward commitment는 stop-gradient를 뜻하지 않는다.

detach는 feedback Jacobian을 제거할 수 있지만 미래 read가 과거 write를
가르치는 경로도 함께 제거한다. 따라서 최소 의미 보존형의 기본값은 full
BPTT다.

또한 shared parameter는 여러 recurrent use와 여러 CE의 gradient를
합산한다. raw global parameter gradient norm이 horizon과 함께 커지는
사실만으로 recurrent Jacobian 폭발을 판정하지 않는다. 안정성 판단은
다음을 분리해야 한다.

- use 사이 state adjoint의 증감 비율
- one-step Jacobian의 JVP/VJP gain 또는 singular-value 추정
- 동일 CE가 이전 horizon으로 전달될 때의 adjoint 변화
- shared parameter에 독립 경로가 합산되는 \(\sqrt H\) 또는 \(H\) 효과

## 10. 구현 순서

원인 분리를 위해 두 단계를 구분한다.

### A. 최소 의미 보존형 비교군

\[
\begin{aligned}
\bar z_t&=Rz_t,\\
S_t&=US_{t-1}+k(z_t)v(z_t)^\top,\\
m_t&=\operatorname{Re}(q(z_t)^\dagger S_t)/\sqrt{d_k},\\
z_{t+1}&=N_z(\bar z_t+\alpha Om_t).
\end{aligned}
\]

- raw \(S\)
- hidden boundary postnorm
- full BPTT
- no realized-state read denominator
- no memory postnorm
- no detach

이 비교군은 원래 구조와 표준 residual closure만으로 H4가 충분히 안정적인지
확인한다. 안정성을 증명하는 구조는 아니므로 recurrent adjoint와 local
Jacobian을 함께 측정해야 한다.

### B. bounded-coupling 후보

A에서 recurrent adjoint amplification이 확인될 때 다음을 각각 분리해
추가한다.

1. Q/K/V의 per-head radial cap
2. measurement correction의 tangent projection
3. bounded angular step
4. 필요한 projection의 spectral bound

여러 개입을 처음부터 동시에 넣으면 어떤 경계가 실제 문제였는지 식별할
수 없다.

## 11. 최종 형태

의미와 안정성 역할을 분리한 후보는 다음과 같다.

\[
\boxed{
\begin{aligned}
S_t&=US_{t-1}+k(z_t)v(z_t)^\top,\\
m_t&=\operatorname{Re}(q(z_t)^\dagger S_t),\\
z_{t+1}&=\text{bounded tangent update of }z_t\text{ by }Om_t.
\end{aligned}}
\]

memory의 unitary transport, additive write, superposition과 interference
measurement는 원형을 유지한다. 안정화는 새 recurrent loop가 생기는 write
입력과 measurement backaction의 크기 및 미분에만 적용한다.
