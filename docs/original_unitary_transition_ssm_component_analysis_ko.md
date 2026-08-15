# 원래 Unitary Transition SSM의 구성요소와 의미

## 문서 범위

이 문서는 [제미나이대화내용.txt](./제미나이대화내용.txt)에 적힌 원래
단일 레이어를 구성요소별로 분해한다. 현재 구현이나 실험 결과를 설명하는
문서가 아니며, 원문의 식과 그 식에서 직접 따라오는 대수적 결과만 다룬다.

원래 식은 다음과 같다.

\[
q_t=(x_tW_Q)e^{i\phi_q},\qquad
k_t=(x_tW_K)e^{i\phi_k},\qquad
v_t=x_tW_V\in\mathbb R^{d_v},
\]

\[
S_t=US_{t-1}+k_tv_t^\dagger,
\qquad
U=e^{-i\theta},
\]

\[
o_t=\operatorname{Re}(q_t^\dagger S_t).
\]

원문에서는 key와 value 차원을 모두 \(d\)로 쓰지만, 역할을 구분하기
위해 여기서는 각각 \(d_k,d_v\)로 표기할 수 있다.

| 객체 | 수학적 형식 | 역할 |
|---|---|---|
| \(U\) | key 공간의 unitary operator | 기존 기억의 시간 위상 이동 |
| \(k_tv_t^\dagger\) | rank-one key-value tensor | 현재 사건의 신규 기록 |
| \(S_t\) | \(\mathcal H_K\otimes\mathcal H_V^*\) | 모든 기록의 거시적 중첩 상태 |
| \(q_t\) | key 공간의 complex vector | 현재 상태가 사용하는 측정 방향 |
| \(q_t^\dagger S_t\) | complex value vector | query에 따른 memory projection |
| \(\operatorname{Re}\) | real quadrature projection | 위상차를 signed 실수 가중치로 관측 |

## 1. \(U\): 내용의 변환이 아니라 시간 위상의 이동

채널별 전이를 사용하면

\[
U=\operatorname{diag}(e^{-i\theta_1},\ldots,e^{-i\theta_{d_k}})
\]

이다. 기존 memory의 한 원소가

\[
S_{a,b}=r_{a,b}e^{i\alpha_{a,b}}
\]

이면 한 번의 전이 뒤에는

\[
(US)_{a,b}=r_{a,b}e^{i(\alpha_{a,b}-\theta_a)}
\]

가 된다. 크기는 유지되고 key 채널의 위상만 이동한다. 따라서 \(U\)는
저장된 payload를 새로 계산하거나 key 채널끼리 섞는 연산이 아니라,
기존 기록의 나이를 채널별 위상으로 누적하는 시계다.

시점 \(n\)에 만들어진 write는 시점 \(t\)에서

\[
U^{t-n}k_nv_n^\dagger
\]

가 된다. 각 write의 key 주소가 나이에 따라 회전하고 value payload는
그대로 남는다.

상대적인 시간차를 위상차로 바꾼다는 점은 RoPE와 유사하다. 차이는
RoPE가 일반적으로 query와 key의 위치별 좌표 변환인 데 비해, 여기서
\(U\)는 recurrent memory를 실제로 운반하는 상태 전이라는 점이다.

\(U^\dagger U=I\)이므로 homogeneous transport는

\[
\lVert US\rVert_F=\lVert S\rVert_F,
\qquad
\left\lVert\frac{\partial(US)}{\partial S}\right\rVert_2=1
\]

을 만족한다. 이 보존은 \(US\)에 관한 것이며, additive write까지 포함한
전체 \(S_t\)의 노름 보존을 뜻하지 않는다.

## 2. write: commit된 사건의 rank-one 연관 기록

현재 입력이 만드는 write를

\[
W_t=k_tv_t^\dagger
\]

라고 두면 각 원소는

\[
(W_t)_{a,b}=k_{t,a}\overline{v_{t,b}}
\]

이다. 원래 식에서는 \(v_t\)가 실수이므로

\[
(W_t)_{a,b}=k_{t,a}v_{t,b}
\]

이다.

- \(k_{t,a}\): 사건이 어떤 key 채널과 위상에 기록되는지 나타내는 주소
- \(v_{t,b}\): 그 주소가 조회될 때 반환할 실수 내용
- \(k_tv_t^\dagger\): 주소와 내용을 결합한 rank-one 관계

query로 write 하나를 읽으면

\[
q^\dagger(k_tv_t^\dagger)
=(q^\dagger k_t)v_t^\dagger
\]

가 된다. \(q^\dagger k_t\)가 검색 계수이고 \(v_t\)가 반환되는
payload다. 따라서 write의 의미는 다음과 같다.

> \(k_t\)라는 조건으로 조회될 때 \(v_t\)를 반환하는 사건 하나를
> memory에 기록한다.

### 2.1 극형식에서 write가 담는 정보

\[
k_{n,a}=|k_{n,a}|e^{i\phi^k_{n,a}}
\]

라고 하면 시점 \(t\)까지 운반된 한 write의 원소는

\[
(W_{n\rightarrow t})_{a,b}
=|k_{n,a}|v_{n,b}
e^{i(\phi^k_{n,a}-(t-n)\theta_a)}
\]

이다.

- \(|k_{n,a}|\): 주소 채널 \(a\)에 대한 기록 강도
- \(\phi^k_{n,a}\): key의 의미적 위상
- \(v_{n,b}\): payload 채널 \(b\)의 내용과 부호
- \((t-n)\theta_a\): 사건의 상대적 나이

즉 write는 committed event를 주소의 세기, 주소의 의미 위상, payload,
시간 위상을 갖는 복소 phasor field로 바꾼다.

단일 write의 크기는

\[
\lVert W_t\rVert_F=\lVert k_t\rVert_2\lVert v_t\rVert_2
\]

이다. 원래 식에는 write gate나 write normalization이 없으므로 한 사건이
주입하는 크기는 \(k_t\)와 \(v_t\)의 크기가 결정한다.

### 2.2 왜 곱셈이 아니라 덧셈인가

\(U\)와 \(W_t\)는 행렬처럼 보이더라도 같은 역할의 객체가 아니다.

- \(U:\mathcal H_K\rightarrow\mathcal H_K\)는 key 공간의 전이 연산자다.
- \(W_t\in\mathcal H_K\otimes\mathcal H_V^*\)는 \(S_t\)와 같은 공간의
  상태 성분이다.

key와 value 차원을 다르게 두어 \(d_k\ne d_v\)로 일반화하면 \(U\)는
정사각 행렬이고 \(W_t\)는 직사각 행렬이어서 이 차이가 shape에도
드러난다.

가법적 갱신은

\[
S_t=US_{t-1}+W_t
\]

이고 \(S_0=0\)이어도 \(S_1=W_1\)로 새 정보가 들어온다. 반대로

\[
S_t=W_tUS_{t-1}
\]

이면 \(S_0=0\)에서 모든 상태가 계속 0이다. 곱셈은 기존 carrier를
변형할 수 있지만 새로운 진폭을 주입하지 못한다.

또한 일반적인 \(W_t=k_tv_t^\dagger\)는 rank가 최대 1이므로 \(d>1\)인
전체 공간의 회전 연산자가 될 수 없다. 수치적으로 곱셈이 가능한 경우에도

\[
W_tUS=k_t(v_t^\dagger US)
\]

가 되어 결과를 \(k_t\) 방향의 rank-one 상태로 투사한다. 이때 \(v_t\)는
반환할 payload가 아니라 기존 상태를 측정하는 bra가 되므로 write의 의미가
사라진다.

사건 의존 회전을 원한다면

\[
R_t=\exp(-iH_t),\qquad H_t=H_t^\dagger
\]

같은 별도 unitary operator가 필요하다. 이 경우 역사는 additive
superposition이 아니라

\[
S_t=R_tUR_{t-1}U\cdots R_1US_0
\]

라는 ordered operator product로 표현된다. 이는 사건들을 각각 기록하는
연상 memory가 아니라 하나의 carrier를 계속 변형하는 다른 동역학이다.

극형식에서 두 연산의 역할은 다음처럼 분리된다.

\[
\text{complex multiplication by }U
\quad\longrightarrow\quad
\text{기존 위상의 시간 이동},
\]

\[
\text{complex addition of }W_t
\quad\longrightarrow\quad
\text{새 사건의 진폭 주입과 간섭}.
\]

극형식 자체가 덧셈을 강제하지는 않는다. 덧셈은 새 사건을 독립된
선형항으로 남기고 read 시점까지 선택을 미루겠다는 superposition 계약에서
나온다.

## 3. \(S_t\): 시간화된 사건들의 거시적 중첩

\(S_0=0\)에서 recurrence를 펼치면

\[
S_t=\sum_{n=1}^{t}U^{t-n}k_nv_n^\dagger
\]

이다. 따라서 \(S_t\)는 하나의 의미 벡터가 아니라, 지금까지 기록된
key-value 관계들이 각자의 나이만큼 회전된 뒤 중첩된 거시 memory다.

한 셀은

\[
S_{t,a,b}
=\sum_{n=1}^{t}
|k_{n,a}|v_{n,b}
e^{i(\phi^k_{n,a}-(t-n)\theta_a)}
\]

이다. 같은 방향의 phasor는 보강되고 반대 방향은 상쇄된다. 새 write
\(s e^{i\beta}\)를 기존 셀 \(r e^{i\alpha}\)에 더하면

\[
\left|r e^{i\alpha}+s e^{i\beta}\right|^2
=r^2+s^2+2rs\cos(\beta-\alpha)
\]

이므로 덧셈 자체가 위상 정렬도에 따른 간섭을 만든다.

이 구조는 update 시점에 어떤 기억을 삭제하거나 선택하지 않는다. 모든
사건을 같은 고정 크기 operator에 중첩하고, 현재 query가 read할 때 선택을
수행한다.

원문의 “모든 과거 정보가 공존한다”는 표현은 각 사건이 전개식의 별도
선형항으로 남는다는 뜻으로 한정해야 한다. 실제 저장값은 항들의 합인
유한 행렬 하나이므로 다음은 보장되지 않는다.

- 사건마다 별도 slot이 존재함
- 서로 다른 모든 입력열을 \(S_t\)에서 역으로 복원할 수 있음
- 중첩된 모든 write가 서로 간섭 없이 개별적으로 조회됨

또한 전체 memory norm은 보존되지 않는다.

\[
\lVert S_t\rVert_F^2
=\lVert S_{t-1}\rVert_F^2
+\lVert W_t\rVert_F^2
+2\operatorname{Re}\langle US_{t-1},W_t\rangle_F
\]

정렬된 write는 최악에 선형 크기 \(\lVert S_t\rVert_F=O(t)\)로 누적될
수 있고, 비상관 write의 전형적 크기는 \(O(\sqrt t)\)일 수 있다. \(U\)가
보존하는 것은 각 기존 기여의 운반 크기이지 additive sum의 총에너지가
아니다.

일반적인 \(k_tv_t^\dagger\)의 합은 Hermitian, positive-semidefinite,
trace-one을 보장하지 않는다. 따라서 원문이 사용한 density matrix 비유는
물리적 동일성이 아니라 complex operator, unitary transport, superposition,
measurement의 대수적 대응으로 제한한다.

## 4. \(q_t\): 현재 사건의 측정 방향

\(q_t\)는 memory에 저장되는 객체가 아니라 현재 상태가 memory에 던지는
일회성 query다.

\[
q_{t,a}=|q_{t,a}|e^{i\phi^q_{t,a}}
\]

- \(|q_{t,a}|\): key 채널 \(a\)를 검사하는 강도
- \(\phi^q_{t,a}\): 어떤 의미·시간 위상의 기억과 보강될지 정하는 기준각

dagger를 사용하므로 read는 절대 위상이 아니라 상대 위상을 계산한다.

\[
q_t^\dagger U^{t-n}k_n
=\sum_a|q_{t,a}||k_{n,a}|
e^{i(\phi^k_{n,a}-(t-n)\theta_a-\phi^q_{t,a})}
\]

\(q_t^\dagger S_t\)는 \(S_t\)의 key 축을 contraction하고 value 축을
남긴다.

\[
q_t^\dagger S_t
=\sum_n\left(q_t^\dagger U^{t-n}k_n\right)v_n^\dagger
\]

query는 하나의 slot을 고르는 index가 아니라 모든 과거 write에 동시에
복소 검색 계수를 부여하는 선형 측정자다.

원문의 식을 문자 그대로 보면 \(x_tW_Q\)와 \(x_tW_K\)는 실수이고
\(\phi_q,\phi_k\)는 별도 위상 벡터다. 따라서 입력은 각 채널의 크기와
부호를 바꾸지만 임의의 연속적인 입력 의존 위상을 직접 만들지는 않는다.
양수일 때 위상은 \(\phi\), 음수일 때는 \(\phi+\pi\)다. 원문의
“의미론적 위상”은 이 식 안에서는 채널별 기준 위상과 실수 projection의
부호·크기가 결합된 표현이다.

원래 순서는 memory를 갱신한 뒤 같은 시점에 읽는 것이다.

\[
q_t^\dagger S_t
=q_t^\dagger US_{t-1}+(q_t^\dagger k_t)v_t^\dagger
\]

따라서 출력은 과거 memory read와 현재 write의 self-read를 모두 포함한다.

## 5. \(\operatorname{Re}\): 위상차의 signed 실수 관측

과거 사건 하나의 복소 검색 계수를

\[
c_{t,n}=q_t^\dagger U^{t-n}k_n
=|c_{t,n}|e^{i\Delta\phi_{t,n}}
\]

라고 하면

\[
\operatorname{Re}(c_{t,n})
=|c_{t,n}|\cos\Delta\phi_{t,n}
\]

이다. 실수 \(v\)에서는 전체 출력이

\[
o_t=\sum_n
\operatorname{Re}(c_{t,n})v_n
\]

으로 분해된다.

- \(\Delta\phi=0\): \(+v_n\)으로 최대 보강
- \(\Delta\phi=\pi/2\): 해당 quadrature에서 관측되지 않음
- \(\Delta\phi=\pi\): \(-v_n\)으로 기여

따라서 \(\operatorname{Re}\)는 의미 위상차와 시간 위상차를 cosine
형태의 signed attention weight로 바꾼다. \(U\)의 복소 회전이 실제 출력의
시간 진동으로 나타나는 지점도 이 측정이다.

절댓값을 사용하지 않는 이유는 보강과 반대 위상을 구분하고 선형
superposition을 유지하기 위해서다.

\[
\operatorname{Re}\left(\sum_n y_n\right)
=\sum_n\operatorname{Re}(y_n)
\]

이지만 일반적으로

\[
\left|\sum_n y_n\right|\ne\sum_n|y_n|
\]

이다. 원문의 간섭은 intensity \(|\sum y|^2\)의 pairwise cross term이
아니라, homodyne식 선형 quadrature에서 나타나는 signed 보강과 상쇄다.

Re와 Im을 모두 downstream에 전달하면 사건 하나는

\[
|c_{t,n}|
\begin{bmatrix}
\cos\Delta\phi_{t,n}\\
\sin\Delta\phi_{t,n}
\end{bmatrix}
\]

로 남는다. 이 경우 \(\pi/2\) 위상차의 사건도 버려지지 않고 downstream
projection이 측정 기준을 학습할 수 있다. 표현 정보는 늘지만, 위상 정렬된
cosine quadrature만 관측한다는 원래 homodyne 의미와는 다른 read다.

## 6. 실수 value와 복소 value의 차이

원래 문서는

\[
v_t\in\mathbb R^{d_v}
\]

를 명시한다. 이때

\[
\operatorname{Re}(c_{t,n}v_n)
=\operatorname{Re}(c_{t,n})v_n
\]

이므로 시간과 query 위상은 payload의 세기와 부호만 바꾸며 \(v_n\)의
실수 방향은 고정된다.

복소 value

\[
v_{n,b}=|v_{n,b}|e^{i\psi_{n,b}}
\]

를 사용하면 write의 한 원소는

\[
(W_n)_{a,b}
=|k_{n,a}||v_{n,b}|
e^{i(\phi^k_{n,a}-\psi_{n,b})}
\]

가 되고, 실수 관측은

\[
\operatorname{Re}(c_{t,n}\overline{v_{n,b}})
=|c_{t,n}||v_{n,b}|
\cos(\angle c_{t,n}-\psi_{n,b})
\]

가 된다. value 채널마다 위상이 다르면 query나 시간에 따라 각 출력 채널이
서로 다르게 변하므로 같은 write가 다른 실수 내용 방향으로 관측될 수 있다.

- 실수 \(v\): 위상은 주소·시간·선택을 담당하고 \(v\)는 고정 payload다.
- 복소 \(v\): 주소 선택과 관측되는 payload 내용이 value 위상으로 결합된다.

직접적인 \(S\mapsto US\)의 unitary 안정성은 둘 다 동일하다. 차이는 주로
관측 동역학과 inductive bias다. 복소 \(v\)는 실수 \(v\)를 특수한 경우로
포함하므로 표현력 손실을 뜻하지 않지만, 원래의 주소와 내용 분업을 바꾸는
구조적 변경이다. 자세한 단독 비교는
[real_vs_complex_value_dynamics_ko.md](./real_vs_complex_value_dynamics_ko.md)에
있다.

## 7. 원래 단일 레이어의 gradient 범위

원래 입력열 \(x_t\)가 주어져 write가 이전 memory와 독립일 때 직접 memory
Jacobian은

\[
\frac{\partial S_t}{\partial S_{t-1}}=U
\]

이다. 따라서 memory carrier 자체에는 반복적인 spectral 증폭이나 감쇠가
없다.

그러나 이것만으로 전체 layer의 모든 gradient가 일정하다는 뜻은 아니다.
write와 read에는 다음 크기 의존성이 있다.

\[
\lVert k_tv_t^\dagger\rVert_F
=\lVert k_t\rVert_2\lVert v_t\rVert_2,
\]

\[
\lVert\delta(q^\dagger S)\rVert
\le\lVert q\rVert_2\lVert\delta S\rVert_F,
\qquad
\lVert\delta(q^\dagger S)\rVert
\le\lVert S\rVert_F\lVert\delta q\rVert_2.
\]

즉 memory로 돌아가는 read gradient는 \(\lVert q\rVert\), query로 돌아가는
gradient는 \(\lVert S\rVert\)에 비례할 수 있다. \(\operatorname{Re}\)는
직교 projection이므로

\[
\lVert\operatorname{Re}(r)\rVert\le\lVert r\rVert
\]

이며 자체적으로 노름을 증폭하지 않는다. phase 미분도 cosine의 미분인
sine이므로 절댓값이 1 이하이지만, 위상 정렬점에서는 phase gradient가
0에 가까워질 수 있다.

따라서 원래 구조가 보장하는 것은 다음으로 한정된다.

> 주어진 write tape에 대한 직접 memory transport는 unitary다. additive
> accumulation, Q/K/V 크기, read의 bilinear coupling 및 여러 loss에서
> 합쳐지는 gradient까지 자동으로 일정하게 만들지는 않는다.

## 8. \(x_t\)를 committed AR state \(z_t\)로 바꾸는 해석

write 의미에 필요한 것은 입력이 관측 token 그 자체라는 조건이 아니라,
AR 시간축에서 하나의 새롭고 확정된 사건이라는 조건이다. 따라서

\[
z_t=\text{prefix로부터 만들어져 }x_t\text{를 추론하는 committed latent event}
\]

로 정의하고

\[
q_t=Q(z_t),\qquad k_t=K(z_t),\qquad v_t=V(z_t)
\]

로 바꾸는 것은 write의 기본 의미와 양립한다.

- \(z_t\): 현재 AR 시점의 commit된 사건
- \(k_t\): 그 사건을 미래에 찾을 주소
- \(v_t\): 미래에 반환할 내용
- \(W_t\): 사건의 지속적인 memory 흔적
- \(q_t\): 현재 사건이 과거와 현재 memory에 던지는 질의

원래 순서를 유지하면 \(z_t\)로 write한 \(W_t\)를 같은 시점의 query가
즉시 self-read한다. \(z_t\)가 target \(x_t\)를 보지 않은 prefix-only
상태라면 이것은 target 누출이 아니다.

commit은 forward 시간에서 \(z_t\)를 하나의 새 사건으로 취급한다는
뜻이다. 미래 loss가 BPTT를 통해 그 상태를 생성한 규칙에 신용을 할당하는
것과 detach는 별개의 문제다.

다만 \(z_t\)가 이전 \(S\)의 read로 만들어지면 write tape가 더 이상
외생적이지 않다.

\[
S\longrightarrow z\longrightarrow(k,v)\longrightarrow S
\]

라는 feedback loop가 생기므로 전체 결합 Jacobian은 \(U\) 하나가 아니다.
조건부 직접 memory edge \(S_{t-1}\mapsto US_{t-1}\)는 여전히 unitary지만,
\(S\rightarrow z\rightarrow W\)와 \(S\rightarrow z\rightarrow q\rightarrow
\text{read}\)의 간접항을 함께 분석해야 한다. 이것이 원래 단일 레이어의
안정성에서 hidden recurrence의 안정성으로 넘어갈 때 새로 추가되는 문제다.

## 9. 원래 구조의 최소 의미 계약

1. \(U\)는 기존 기록의 크기를 바꾸지 않고 시간 위상만 이동한다.
2. write는 commit된 사건을 rank-one key-value 관계로 새로 주입한다.
3. \(S\)는 사건들을 지우지 않고 선형 중첩하되, 개별 slot 복원을
   보장하지 않는다.
4. \(q\)는 현재 상태가 중첩 memory에 사용하는 측정 방향이다.
5. 실수 \(v\)에서는 phase가 검색을 담당하고 payload 내용은 고정된다.
6. \(\operatorname{Re}\)는 상대 위상을 cosine signed weight로 관측한다.
7. 직접 memory transport의 Jacobian만 unitary이며 전체 결합 recurrence의
   안정성은 별도 문제다.

이 계약을 기준으로 하면 이후 재귀 확장에서 어떤 변경이 단순한 수치
안정화이고 어떤 변경이 memory의 의미 자체를 바꾸는지 구분할 수 있다.
