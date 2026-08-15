# URM식 carry 경계와 복소 메모리 재귀의 대응

## 1. 범위

이 문서는 현재 `unitary-detached-input-raw-read-postnorm-residual`
중앙층에서 재귀 carry가 무엇인지, hidden-only detach와 전체 carry
detach가 어떻게 다른지, 그리고 전체 carry의 forward scale을 제한하려면
왜 hidden뿐 아니라 complex memory에도 경계 연산이 필요한지를 정리한다.

여기서 말하는 `URM식 경계`는 반복 블록의 이전 상태를 다음 블록에
주어지는 입력값으로 취급하는 계산 경계를 뜻한다. 특정 구현의 모든
연산을 그대로 복제한다는 뜻은 아니다.

## 2. 현재 중앙층의 실제 재귀 상태

재귀 스텝을 `j`라고 하면 현재 중앙층은 다음 두 값을 다음 스텝으로
전달한다.

```math
z_j \in \mathbb R^d,
\qquad
S_j \in \mathbb C^{H\times d_k\times d_v}.
```

hidden과 memory의 갱신은 다음 형태다.

```math
u_j = Rz_{j-1},
```

```math
(q_j,k_j,v_j)=\operatorname{QKV}(u_j),
```

```math
W_j=k_jv_j^\dagger,
\qquad
S_j=US_{j-1}+W_j,
```

```math
r_j=\frac{q_j^\dagger S_j}{\sqrt{d_k}},
\qquad
z_j=N_z\!\left(u_j+O\,\operatorname{flat}(r_j)\right).
```

`R`과 `U`는 각각 hidden과 memory에 작용하는 unitary rotation이고,
`N_z`는 fixed non-affine RMS post-normalization이다.

따라서 중앙층의 완전한 Markov state는 `z_j` 하나가 아니라

```math
c_j=(z_j,S_j)
```

이다. `S_j`는 단순한 현재 스텝의 임시 attention 출력이 아니다.
`S_{j-1}`가 `S_j`에 직접 들어가고, 갱신된 `S_j`를 읽은 값이 다시
`z_j`를 만든다.

```math
\left.\frac{\partial S_j}{\partial S_{j-1}}\right|_{z_{j-1}}=U,
\qquad
\frac{\partial z_j}{\partial S_j}\ne 0.
```

그러므로 `z`와 `S`는 독립적인 병렬 경로가 아니라 하나의 결합된
재귀 상태다.

## 3. 시간에 따라 변하는 attention과 직접 carry되는 memory의 차이

반복 Transformer 블록의 attention map도 스텝마다 변한다.

```math
A_j=\operatorname{Attention}(z_j),
\qquad
z_{j+1}=B(z_j,A_j).
```

그러나 현재 hidden을 고정했을 때 이전 attention map이 다음 attention
map에 직접 더해지는 것은 아니다.

```math
\left.\frac{\partial A_j}{\partial A_{j-1}}\right|_{z_j}=0.
```

이전 attention의 효과는 갱신된 residual stream `z_j`를 통하여 다음
attention 계산에 간접적으로 들어간다. 반면 현재 복소 메모리는

```math
S_j=US_{j-1}+W_j
```

이므로 이전 memory가 명시적인 상태로 직접 전달된다. 두 구조 모두
attention 관련 값이 스텝에 따라 진화하지만, 현재 구조에는
`S_{j-1}\rightarrow S_j`라는 추가적인 직접 carrier가 있다.

MLP의 부재는 이 차이의 원인이 아니다. MLP를 추가하더라도 위 memory
갱신식을 유지하면 `(z,S)` 공동 carry는 그대로 남는다. MLP는 residual
stream을 다음 attention 호출 전에 추가로 변환하지만,
`S_{j-1}`의 직접 전달 여부를 바꾸지는 않는다.

## 4. 세 가지 detach 경계

### 4.1 현재 구현: hidden-only detach

현재 mode는 스텝 입구에서 이전 hidden만 stop-gradient한다.

```math
\tilde z_{j-1}=\operatorname{sg}(z_{j-1}),
\qquad
\tilde S_{j-1}=S_{j-1}.
```

따라서 hidden temporal edge는 끊기지만 memory temporal edge는 살아
있다.

```text
H4 CE -> S4 -> S3 -> S2 -> S1 -> 과거 K/V
```

이 경로 때문에 H4 CE가 이전 재귀 use의 write 파라미터까지 도달하는
것은 현재 구현과 일치한다.

### 4.2 `(z,S)` 전체 carry detach

현재 중앙층의 완전한 재귀 상태를 이전 스텝의 외부 입력으로
취급하려면 두 필드를 함께 stop-gradient해야 한다.

```math
\tilde z_{j-1}=\operatorname{sg}(z_{j-1}),
\qquad
\tilde S_{j-1}=\operatorname{sg}(S_{j-1}).
```

그 뒤 현재 스텝 내부의 Q/K/V, write, read, output projection과 postnorm은
그대로 미분한다. 이 경우 미래 loss가 predecessor carrier를 통하여
이전 use로 들어가는 경로는 사라진다.

```math
\frac{\partial (z_j,S_j)}
{\partial (z_{j-1},S_{j-1})}=0
```

은 autograd 경계에 대한 식이다. forward 값은 바뀌지 않으며, `S_j`는
계속 이전 memory 값에 현재 write를 더해 계산된다.

현재 공통 모델의 `rollout(..., detach_state_between_horizons=True)`는
스텝 사이에서 hidden과 memory를 함께 detach하는 이 경계를 이미
표현한다. 현재 학습 mode는 이 옵션을 사용하지 않고 hidden만 내부에서
detach한다.

### 4.3 inverse-decoder의 과거 branch tape

`(z,S)` carry detach와 decoder의 causal tape는 별개의 경로다. H4
decoder가 `p_1,p_2,p_3`을 attention할 때 그 tape가 attached라면 H4 CE는
decoder를 통하여 이전 branch state에 도달할 수 있다. 모든 horizon의
신용할당까지 use-local로 만들려면 decoder의 과거 branch state 경계도
별도로 stop-gradient해야 한다.

## 5. 전체 carry detach 뒤에도 남는 `N` 의존성

memory carry가 attached일 때 memory adjoint는 다음처럼 누적된다.

```math
G_{S,j-1}=U^\dagger G_{S,j}+B_{j-1}.
```

서로 비정렬된 항은 대략 `sqrt(N)`, 정렬된 항은 최악의 경우 `N`에
비례할 수 있다. `(z,S)` 전체 carry detach는 이 스텝 간 adjoint 재귀를
제거한다.

그러나 다음 두 효과는 detach만으로 제거되지 않는다.

첫째, forward memory 값은 그대로 누적된다.

```math
S_j=\sum_{n=1}^{j}U^{j-n}W_n.
```

따라서 write가 비정렬이면 `||S_j||`가 대략 `sqrt(j)`, 정렬되면 최악의
경우 `j`에 비례할 수 있다. raw read에서는

```math
\left\|\frac{\partial r_j}{\partial q_j}\right\|
=\frac{\|S_j\|}{\sqrt{d_k}}
```

이므로 현재 스텝의 local query gradient가 memory의 forward 크기에
의존한다.

둘째, 같은 파라미터를 모든 스텝에서 공유한다.

```math
g_\theta=\sum_{j=1}^{N}g_{\theta,j}.
```

loss를 합산하면 비정렬 gradient는 `sqrt(N)`, 정렬 gradient는 `N`까지
증가할 수 있다. `N`개 CE를 평균하면 각각 대략 `1/sqrt(N)`과 `1`
스케일이 된다. 이는 temporal Jacobian 곱이 아니라 tied parameter의
일반적인 gradient 합산이다.

## 6. 공동 carry와 S post-update normalization

`c_j=(z_j,S_j)` 전체가 재귀 carry이고, post-normalized block 출력만
다음 재귀 단위로 전달한다는 경계 조건을 적용한다면 두 성분 모두에
forward scale 조건이 있어야 한다. z만 fixed-RMS로 제한하고 S를 raw
누적하면 공동 carry 전체가 bounded라는 명제는 성립하지 않는다.

memory scale이 현재 read의 local Jacobian에도 들어가므로 S의 경계
연산은 write 뒤, read 전에 위치해야 한다.

```math
S_j^{\mathrm{raw}}=U\tilde S_{j-1}+k_jv_j^\dagger,
```

```math
S_j=N_S(S_j^{\mathrm{raw}}),
```

```math
r_j=\frac{q_j^\dagger S_j}{\sqrt{d_k}}.
```

read 이후 또는 다음 carry 직전에만 S를 제한하면 현재 스텝의
`\partial r_j/\partial q_j`는 raw memory 크기에 계속 노출된다.

### 6.1 fixed-RMS memory postnorm

현재 공통 모델에 존재하는 memory postnorm은 다음 형태다.

```math
N_S(X)=\frac{X}
{\sqrt{\operatorname{mean}|X|^2+\epsilon}}.
```

이 연산은 memory RMS를 고정하지만, `X`가 작거나 과거 memory와 현재
write가 상쇄될 때 tangent Jacobian gain이

```math
\frac{1}{\sqrt{\operatorname{mean}|X|^2+\epsilon}}
```

까지 커질 수 있다. 따라서 fixed-RMS라는 forward 조건만으로
non-expansive Jacobian이 보장되지는 않는다. 이 옵션은 현재
`unitary-detached-input-raw-read-postnorm-residual` 실험에는 적용되지
않았다.

### 6.2 비증폭 radial post-cap

작은 memory를 확대하지 않으면서 큰 memory만 제한하는 한 가지 연산은
다음과 같다.

```math
C_c(X)=\frac{X}
{\sqrt{1+\left(\operatorname{rms}(X)/c\right)^2}}.
```

이때

```math
\operatorname{rms}(C_c(X))<c.
```

radial 방향의 singular value는
`(1+(r/c)^2)^{-3/2}`, tangent 방향은
`(1+(r/c)^2)^{-1/2}`이므로 둘 다 1 이하이다. 이 연산은 fixed-RMS
normalization이 아니라 bounded radial post-cap이다.

두 memory 경계 연산 모두 하나의 양의 실수 배율만 적용하므로 그
스텝에서 complex 방향, 채널 사이 상대 위상과 상대 크기는 보존한다.

## 7. S 경계 연산이 바꾸는 forward 의미

memory 경계 연산은 detach와 달리 forward 계산을 변경한다. 경계 연산이
없을 때는

```math
S_j=\sum_{n=1}^{j}U^{j-n}W_n
```

이라는 정확한 unitary 선형 중첩식이 성립한다. 매 write 뒤 `N_S` 또는
`C_c`를 적용하면 과거 write 전체가 이후 스텝마다 하나의 radial
계수로 다시 스케일된다. 그 결과:

- memory의 방향과 내부 위상 관계는 해당 스텝에서 보존된다.
- memory의 절대 에너지는 더 이상 lossless하게 보존되지 않는다.
- 과거 write와 현재 write의 상대 계수는 이후 경계 배율의 영향을 받는다.
- 상태 의존 비선형성이 들어가므로 원래 affine associative scan 식은
  그대로 유지되지 않는다.

따라서 다음 두 선택은 서로 다른 개입이다.

```text
(z,S) carry detach   : backward temporal edge만 변경, forward 동일
S postnorm/post-cap  : forward memory 크기와 중첩 계수를 변경
```

정확한 unitary 중첩을 보존해야 한다면 S carrier postnorm은 사용할 수
없으며, write scale이나 결정론적 read scale처럼 선형 누적식과 양립하는
별도 스케일 규칙이 필요하다.

## 8. 공동 경계를 적용한 계산 형태

전체 carry detach와 memory 경계 연산을 동시에 표현하면 한 스텝은 다음
형태다.

```math
\tilde z_{j-1}=\operatorname{sg}(z_{j-1}),
\qquad
\tilde S_{j-1}=\operatorname{sg}(S_{j-1}),
```

```math
u_j=R\tilde z_{j-1},
\qquad
(q_j,k_j,v_j)=\operatorname{QKV}(u_j),
```

```math
S_j=N_S\!\left(U\tilde S_{j-1}+k_jv_j^\dagger\right),
```

```math
r_j=\frac{q_j^\dagger S_j}{\sqrt{d_k}},
```

```math
z_j=N_z\!\left(u_j+O\,\operatorname{flat}(r_j)\right),
\qquad
c_j=(z_j,S_j).
```

여기서 `N_S`를 fixed-RMS postnorm으로 둘지, 비증폭 post-cap으로 둘지,
또는 unitary 중첩 보존을 위해 항등으로 둘지는 서로 다른 forward
아키텍처다. 전체 carry detach만 선택하는 경우에는 `N_S`가 항등이고,
forward memory 누적은 현재 구조와 동일하다.

## 9. carry와 normalization에 대한 중간 결론

현재 중앙층의 재귀 상태는 `(z,S)`다. hidden-only detach는 이 상태의
일부만 끊으므로 memory를 통한 미래-to-past 신용할당을 유지한다.
`(z,S)` 전체 detach는 그 temporal adjoint 누적을 제거하지만, raw S의
forward 성장과 공유 파라미터 gradient 합산까지 제거하지는 않는다.

공동 carry 전체를 post-update bounded state로 취급하려면 z뿐 아니라
S에도 경계 연산이 필요하다. 다만 S postnorm은 detach와 달리 unitary
선형 중첩의 forward 의미를 바꾸므로 별도 아키텍처 개입으로 구분해야
한다.

## 10. 하나의 4-token 문제라는 해석

prefix를 `x`, 다음 네 token을

```math
Y=(y_1,y_2,y_3,y_4)
```

라고 두면 H1--H4는 서로 무관한 네 문제가 아니라 하나의 구조화된
continuation을 이룬다. 각 좌표는 다음과 같은 순차 제약을 가진다.

```text
y1: x와 양립해야 한다.
y2: x와 y1에 양립해야 한다.
y3: x, y1, y2에 양립해야 한다.
y4: x, y1, y2, y3에 양립해야 한다.
```

표준 teacher-forced autoregressive model에서는 이 구조가 정확히 다음
chain rule로 표현된다.

```math
-\log p(Y\mid x)
=\sum_{h=1}^{4}-\log p(y_h\mid x,y_{<h}).
```

현재 latent rollout은 정답 `y_{<h}`를 다음 중앙 스텝에 다시 넣지
않는다. 대신 이전에 생성한 latent와 complex memory를 carry한다.
따라서 현재 H1--H4 CE 합은 표준 teacher-forced chain rule과 완전히 같은
계산은 아니며,

```math
-\log p_h(y_h\mid x,c_{<h}),
\qquad c_h=(z_h,S_h)
```

형태의 internally generated trajectory supervision이다. 그래도 네 CE가
하나의 4-token continuation에 공동 제약을 준다는 해석은 유지된다.

문장 매니폴드를

```math
\mathcal M(x)=\{Y:\;Y\text{가 prefix }x\text{와 양립하는 continuation}\}
```

이라고 부르면, 중앙 재귀는 `\mathcal M(x)` 위의 한 부분해를 순차적으로
구성하는 과정으로 볼 수 있다.

## 11. URM과 현재 모델에서 재귀축의 의미

URM의 공식 구현에서 한 recurrent block은 non-causal self-attention,
attention residual post-RMSNorm, ConvSwiGLU residual post-RMSNorm으로
구성된다. 고정된 puzzle input embedding은 각 low-level cycle에 다시
주입되고, carry는 전체 위치의 `current_hidden`이다. 각 outer call의
출력 logits은 동일한 puzzle label 전체와 비교된다.

- [URM 공식 모델 코드](https://github.com/UbiquantAI/URM/blob/main/models/urm/urm.py)
- [URM 공식 loss 코드](https://github.com/UbiquantAI/URM/blob/main/models/losses.py)
- [URM 논문](https://arxiv.org/abs/2512.14693)

URM에는 해답 좌표 `i`와 reasoning iteration `r`이라는 서로 다른 두
축이 있다.

```math
\hat Y^{(r)}=(\hat y_1^{(r)},\ldots,\hat y_m^{(r)}),
```

```math
\hat Y^{(r+1)}=F_\theta(\hat Y^{(r)},x),
```

```math
L_r=\sum_i \operatorname{CE}
\left(\hat y_i^{(r)},Y_i\right).
```

`r`이 증가해도 target `Y`와 각 출력 좌표의 의미는 고정된다. 뒤
iteration은 앞 iteration이 틀린 좌표를 다시 수정할 수 있다.

현재 모델에서는 중앙 재귀 인덱스 `h`가 미래 token 위치와 동시에
사용된다.

```math
c_h=F_\theta(c_{h-1}),
\qquad
L_h=\operatorname{CE}(D_h(c_{\le h}),y_h).
```

H1에서 H2로 이동하면 같은 token 후보를 다시 정제하는 것이 아니라
다음 token 좌표로 이동한다. H2가 H1의 예측을 내부 조건으로 사용할 수는
있지만, H1 출력 자체를 다시 H1 정답에 맞게 수정하지는 않는다.

| 구분 | URM | 현재 H1--H4 모델 |
|---|---|---|
| 재귀 인덱스 | reasoning iteration `r` | 미래 위치 `h`와 동일 |
| 한 스텝의 출력 | 전체 solution grid | 현재 horizon의 token |
| 스텝별 target | 항상 동일한 전체 `Y` | `y_1`부터 `y_4`로 이동 |
| 이전 좌표 수정 | 뒤 iteration에서 가능 | 이미 지난 horizon은 고정 |
| 입력 제약 전달 | 전체 위치의 global attention | causal latent/memory tape |

따라서 URM은 동일한 전체 후보를 반복 정제하는 solver이고, 현재 모델은
부분해를 순서대로 확장하는 constructive solver다.

## 12. 제약충족 문제로서 같은 점과 다른 점

두 모델은 모두 하나의 구조화된 출력에 존재하는 여러 제약을 shared
recurrent computation으로 전파한다는 점에서 같은 문제 계열로 볼 수
있다. Sudoku도 정해진 변수 순서대로 값을 배치하면 순차 constructive
solver로 풀 수 있고, 네 token continuation도 전체 block의 양립 조건을
만족시키는 부분 할당으로 볼 수 있다.

그러나 고전적 Sudoku와 자연어 continuation 사이에는 다음 차이가
있다.

1. Sudoku는 보통 입력에 의해 정답이 결정되는 단일해 문제다.
2. 자연어 prefix에는 서로 다른 여러 continuation이 동시에 타당할 수
   있다.
3. 데이터의 `Y`는 가능한 매니폴드 전체가 아니라 그중 관측된 샘플
   하나다.
4. 따라서 자연어 CE는 단일 satisfying assignment의 복원뿐 아니라
   조건부 확률질량의 배분을 학습한다.

그러므로 현재 문제는 결정론적 CSP와 완전히 같지는 않지만,
`\mathcal M(x)` 위에서 causal compatibility를 만족하는 구조적 해를
구성한다는 의미에서는 제약충족 관점이 성립한다.

## 13. 이 해석이 carry detach에 주는 결과

URM은 각 supervision 지점에서 동일한 전체 solution label을 다시 본다.
carry가 outer 경계에서 detach되어도 현재 hidden은 그 지점의 전체 grid
CE로 직접 학습된다. 반면 현재 모델의 H1 state가 직접 받는 target은
기본적으로 `y_1`이다.

따라서 `(z,S)`와 decoder의 과거 branch tape까지 모두 끊으면 H2--H4 CE가
H1 state에 다음 정보를 가르치는 경로도 함께 사라진다.

```text
H1에서 무엇을 보존해야 H2를 맞힐 수 있는가
H1/H2에서 무엇을 보존해야 H3를 맞힐 수 있는가
H1/H2/H3에서 무엇을 보존해야 H4를 맞힐 수 있는가
```

즉 같은 full carry detach라도 감독의 범위가 다르다.

```text
URM: 각 detached 구간의 끝에서 전체 해답을 직접 감독
현재: 각 horizon state에 서로 다른 한 token을 직접 감독
```

현재 decoder tape를 attached로 유지하면 미래 CE가 이전 branch state에
도달하는 우회 경로는 남는다. memory carry를 attached로 유지하면 미래
CE가 이전 K/V write에 도달한다. 두 경로를 모두 끊은 완전한 use-local
학습은 URM의 전체-grid deep supervision과 동일하지 않다.

따라서 네 token을 하나의 공동 제약계로 해석할수록, full detach의
안정성 이득과 미래 constraint credit의 손실을 별개로 측정해야 한다.

## 14. URM과 같은 반복 정제 문제로 만드는 조건

URM과 목표뿐 아니라 풀이 축까지 직접 대응시키려면 미래 token 위치와
reasoning iteration을 분리해야 한다.

```text
h = 해답 좌표: token 1..4
r = 같은 4-token 후보를 정제하는 iteration 1..R
```

각 iteration은 네 token 후보 전체를 유지하고 갱신한다.

```math
C^{(r+1)}=F_\theta(C^{(r)},x),
```

```math
\hat Y^{(r)}=D(C^{(r)}),
```

```math
L_r=\frac14\sum_{h=1}^{4}
\operatorname{CE}(\hat y_h^{(r)},y_h).
```

모든 `r`에서 입력은 prefix뿐이고 정답 token을 recurrent input으로 넣지
않으면, 전체 block supervision을 사용해도 label leakage는 생기지 않는다.
후보 slot 사이의 attention을 global로 둘지 causal compatibility mask로
둘지는 별도의 생성 의미 선택이다.

이 형태에서는:

- `h`는 해답의 위치만 나타낸다.
- `r`은 동일한 문장 후보를 반복 정제하는 계산 깊이만 나타낸다.
- 매 `r`마다 target은 동일한 `(y_1,y_2,y_3,y_4)`다.
- carry detach가 있더라도 각 경계에서 전체 block supervision을 줄 수
  있다.

현재 구조는 이 2축 구조가 아니라 `r=h`로 합쳐진 1축 구조다.

## 15. 전체 판정

4-token block을 하나의 문장 매니폴드 또는 causal constraint system으로
보는 것은 타당하다. 그러므로 H1--H4를 네 개의 완전히 독립된 과제로
보는 것은 현재 recurrent carry의 역할을 충분히 설명하지 못한다.

다만 현재 모델은 동일 후보를 반복 수정하는 URM식 solver가 아니라
부분해를 되돌리지 않고 확장하는 solver다. 재귀 깊이가 4라는 점은
URM의 긴 reasoning loop보다 Jacobian 길이를 짧게 만들지만, 명시적인
`S` 누적과 horizon별로 이동하는 supervision은 URM에 없는 구조다.

따라서 URM의 학습 안정성은 현재 모델에 자동으로 이전되지 않는다.
두 모델의 목표는 같은 구조적 문제 계열로 해석할 수 있지만, 현재 가장
큰 차이는 목표 자체보다 reasoning iteration과 output position을 같은
축으로 사용한다는 점이다.
