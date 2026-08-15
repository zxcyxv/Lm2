# 현재 121M residual 모델 아키텍처

대상 mode는 \(\texttt{unitary\text{-}branch\text{-}normalized\text{-}residual}\)이다.
아래에는 모델의 순방향 연산만 적는다.

## 1. 크기와 기호

| 기호 | 값 | 의미 |
|---|---:|---|
| \(B\) | 가변 | batch 크기 |
| \(V\) | \(256\) | byte vocabulary |
| \(T\) | \(256\) | prefix 길이 |
| \(D\) | \(4352\) | hidden 폭 |
| \(d\) | \(2176\) | reversible block의 절반 폭 |
| \(L\) | \(2\) | encoder block 수 |
| \(A\) | \(64\) | anchor 수 |
| \(J\) | \(4\) | recurrence horizon |
| \(H_e,d_e\) | \(4,544\) | encoder attention head 수와 head 폭 |
| \(H_m,K,U\) | \(8,16,31\) | memory head, key 폭, value 폭 |
| \(\varepsilon\) | \(10^{-6}\) | normalization 상수 |

\[
\begin{aligned}
X&:[B,256],& Z&:[B,256,4352],& z^{(0)}&:[B,64,4352],\\
S^{(j)}&:[B,64,8,16,31]\ {\rm complex},&
P,Y&:[B,64,4,4352],&
\Lambda&:[B,64,4,256].
\end{aligned}
\]

전체 경로:

\[
X\rightarrow E[X]\rightarrow \operatorname{Enc}\rightarrow
\{z^{(0)}_a\}\rightarrow
\operatorname{Central}^{\,4}\rightarrow P\rightarrow
\operatorname{Enc}^{-1}\rightarrow Y\rightarrow\Lambda.
\]

## 2. Fixed simplex embedding

byte \(r\in\{0,\ldots,255\}\), 표준기저 \(e_r\), all-one vector
\(\mathbf 1\in\mathbb R^{256}\)에 대해

\[
s_r=\sqrt{\frac{256}{255}}
\left(e_r-\frac1{256}\mathbf1\right),\qquad
E_r=[s_r,0_{4096}]\in\mathbb R^{4352}.
\]

\[
E\in\mathbb R^{256\times4352},\qquad
X^{(0)}_{b,t}=E_{X_{b,t}}\in\mathbb R^{4352}.
\]

\(E\)는 고정 parameter다. 별도 positional embedding은 없다.

## 3. Reversible causal encoder

### 3.1 RMSNorm

각 block의 두 RMSNorm은 서로 다른 학습 gain
\(g_1,g_2\in\mathbb R^{2176}\)를 가진다.

\[
\operatorname{RMS}_g(x)=
g\odot x\left(
\frac1{2176}\sum_{c=1}^{2176}x_c^2+\varepsilon
\right)^{-1/2}.
\]

### 3.2 Four-head causal attention

\(U\in\mathbb R^{B\times T\times2176}\)에 대해

\[
[\widetilde Q,\widetilde K,\widetilde V]
=UW_{\rm qkv}^{\mathsf T},\qquad
W_{\rm qkv}\in\mathbb R^{6528\times2176}.
\]

각 결과를 \([B,4,T,544]\)로 reshape한다. \(m=0,\ldots,271\)번째
채널쌍의 RoPE 주파수와 회전은

\[
\omega_m=10000^{-m/272},\qquad
R(\alpha)\binom ab=
\binom{a\cos\alpha-b\sin\alpha}{a\sin\alpha+b\cos\alpha}.
\]

\[
Q'_{t,m}=R(t\omega_m)Q_{t,m},\qquad
K'_{t,m}=R(t\omega_m)K_{t,m}.
\]

RoPE는 \(Q,K\)에만 적용한다. head별 causal attention은

\[
\sigma_{t,s}=
\begin{cases}
\langle Q'_t,K'_s\rangle/\sqrt{544},&s\le t,\\
-\infty,&s>t,
\end{cases}
\quad
\alpha_{t,:}=\operatorname{softmax}(\sigma_{t,:}),
\quad
o_t=\sum_{s\le t}\alpha_{t,s}V_s.
\]

\[
\operatorname{Attn}(U)=
\operatorname{Concat}(o^{(1)},\ldots,o^{(4)})
W_{\rm proj}^{\mathsf T},\qquad
W_{\rm proj}\in\mathbb R^{2176\times2176}.
\]

### 3.3 FFN과 reversible block

\[
\operatorname{FFN}(x)=
W_2\operatorname{GELU}(W_1x+b_1)+b_2,
\]

\[
W_1:[8704,2176],\quad b_1:[8704],\quad
W_2:[2176,8704],\quad b_2:[2176].
\]

GELU는 정확한 GELU다. \([x_1,x_2]\), \(x_1,x_2\in\mathbb R^{2176}\)에
대한 한 block은

\[
\boxed{
\begin{aligned}
y_1&=x_1+\operatorname{Attn}(\operatorname{RMS}_{g_1}(x_2)),\\
y_2&=x_2+\operatorname{FFN}(\operatorname{RMS}_{g_2}(y_1)).
\end{aligned}}
\]

두 residual coefficient는 \(1\)이다. 서로 다른 parameter를 가진 block
두 개를 순서대로 적용하여

\[
Z=\operatorname{Enc}(X^{(0)})\in\mathbb R^{B\times256\times4352}
\]

를 만든다.

## 4. Anchor와 초기 state

\[
\mathcal A=\{a_n=4n\mid n=0,\ldots,63\}
=\{0,4,\ldots,252\}.
\]

\[
z^{(0)}_{b,n}=Z_{b,a_n},\qquad
S^{(0)}_{b,n,h,k,u}=0\in\mathbb C.
\]

\(z^{(0)}\)에는 추가 normalization이 없다.

## 5. Four-step complex-memory recurrence

아래 연산을 \(j=1,2,3,4\)에 대해 같은 parameter로 반복한다.

### 5.1 Hidden rotation과 QKV 입력

학습 angle \(\phi\in\mathbb R^{2176}\)로 hidden 채널쌍을 회전한다.

\[
\binom{\bar z^{(j)}_{2c}}{\bar z^{(j)}_{2c+1}}
=R(\phi_c)
\binom{z^{(j-1)}_{2c}}{z^{(j-1)}_{2c+1}}.
\]

QKV에만 들어가는 non-affine fixed RMS는

\[
u^{(j)}=
\bar z^{(j)}
\left(
\frac1{4352}\sum_{c=1}^{4352}(\bar z^{(j)}_c)^2+\varepsilon
\right)^{-1/2}.
\]

\(\bar z^{(j)}\) 자체는 normalization 없이 residual 경로에 남는다.

### 5.2 Complex Q, K, V

\[
\begin{aligned}
q_{\rm raw}&=u^{(j)}W_Q^{\mathsf T},&W_Q&:[256,4352],\\
k_{\rm raw}&=u^{(j)}W_K^{\mathsf T},&W_K&:[256,4352],\\
v_{\rm raw}&=u^{(j)}W_V^{\mathsf T},&W_V&:[496,4352].
\end{aligned}
\]

구현에서는 세 weight를 \([1008,4352]\)로 연결해 한 linear 연산으로
계산한다. 각 출력의 앞 절반은 실수부, 뒤 절반은 허수부다.

\[
\begin{aligned}
q&=\operatorname{reshape}(q_{\Re}+i q_{\Im})\in\mathbb C^{8\times16},\\
k&=\operatorname{reshape}(k_{\Re}+i k_{\Im})\in\mathbb C^{8\times16},\\
v&=\operatorname{reshape}(v_{\Re}+i v_{\Im})\in\mathbb C^{8\times31}.
\end{aligned}
\]

### 5.3 Memory rotate, write, read

학습 angle \(\theta\in\mathbb R^{8\times16}\)에 대해

\[
\bar S^{(j)}_{h,k,u}
=e^{-i\theta_{h,k}}S^{(j-1)}_{h,k,u}.
\]

현재 key와 value의 rank-one write 및 memory update는

\[
W^{(j)}_{h,k,u}=k^{(j)}_{h,k}\overline{v^{(j)}_{h,u}},
\qquad
S^{(j)}=\bar S^{(j)}+W^{(j)}.
\]

updated memory를 같은 step에서 읽는다.

\[
r^{(j)}_{h,u}
=\frac1{\sqrt{16}}\sum_{k=1}^{16}
\overline{q^{(j)}_{h,k}}S^{(j)}_{h,k,u}
\in\mathbb C^{8\times31}.
\]

anchor 하나의 전체 memory Frobenius norm과 normalized read는

\[
\rho^{(j)}
=\left(\sum_{h=1}^{8}\sum_{k=1}^{16}\sum_{u=1}^{31}
|S^{(j)}_{h,k,u}|^2\right)^{1/2},
\qquad
m^{(j)}=\frac{r^{(j)}}{\rho^{(j)}+\varepsilon}.
\]

이 division은 read \(r^{(j)}\)에만 적용한다. state memory \(S^{(j)}\)는
raw 값으로 유지된다. write gate, damping, write-count scaling은 없다.

### 5.4 496차원 readout과 residual successor

\[
8\times31=248\ {\rm complex}
\quad\Longrightarrow\quad
2\times248=496\ {\rm real}.
\]

\[
c^{(j)}
=\left[
\operatorname{vec}\Re m^{(j)},
\operatorname{vec}\Im m^{(j)}
\right]\in\mathbb R^{496}.
\]

\[
\delta^{(j)}=c^{(j)}W_O^{\mathsf T},\qquad
W_O\in\mathbb R^{4352\times496}.
\]

현재 residual successor는

\[
\boxed{z^{(j)}=\bar z^{(j)}+\delta^{(j)}}.
\]

별도 scalar와 gate는 곱하지 않는다. 등록된
\(\texttt{residual\_step\_scale}=0.1\)도 이 mode에서는 사용하지 않는다.

decoder-facing branch만 fixed RMS를 통과한다.

\[
p^{(j)}=
z^{(j)}
\left(
\frac1{4352}\sum_{c=1}^{4352}(z^{(j)}_c)^2+\varepsilon
\right)^{-1/2}.
\]

다음 recurrence에는 normalized \(p^{(j)}\)가 아니라 raw
\(z^{(j)}\)와 raw \(S^{(j)}\)가 들어간다.

\[
P=\operatorname{stack}(p^{(1)},p^{(2)},p^{(3)},p^{(4)})
\in\mathbb R^{B\times64\times4\times4352}.
\]

## 6. Shared exact-inverse decoder

별도 decoder block은 없다. encoder block을 \(2\rightarrow1\) 순서로,
동일한 parameter를 사용해 역산한다. 한 block 출력
\([y_1,y_2]\)의 inverse는

\[
\boxed{
\begin{aligned}
x_2&=y_2-\operatorname{FFN}(\operatorname{RMS}_{g_2}(y_1)),\\
x_1&=y_1-\operatorname{Attn}(\operatorname{RMS}_{g_1}(x_2)).
\end{aligned}}
\]

prefix \(Z\)와 branch \(P\)를 함께 역산한다. branch \((n,j)\)의 위치와
허용 source 집합은

\[
\pi_{n,j}=a_n+j,\qquad
\mathcal S_{n,j}
=\{\text{prefix }s:0\le s\le a_n\}
\cup\{\text{branch }(n,r):1\le r\le j\}.
\]

즉 다른 anchor의 branch와 \(r>j\)인 미래 branch는 보지 않는다.
branch query와 허용된 prefix/branch key에 encoder와 같은 QKV weight와
RoPE를 적용한다.

\[
\sigma_{n,j,s}
=\frac{\langle q'_{n,j},k'_s\rangle}{\sqrt{544}}.
\]

prefix score와 같은 anchor의 branch score를 연결한 뒤 하나의 source
축에서 softmax한다.

\[
\alpha_{n,j,:}
=\operatorname{softmax}
\left([
\sigma^{\rm prefix}_{n,j,:},
\sigma^{\rm branch}_{n,j,:}
]\right).
\]

\[
o_{n,j}
=\sum_{s=0}^{a_n}\alpha^{\rm prefix}_{n,j,s}v_s
+\sum_{r=1}^{j}\alpha^{\rm branch}_{n,j,r}v_{n,r}.
\]

네 head를 연결한 뒤 encoder의 같은 \(W_{\rm proj}\)를 적용한다.
두 inverse block을 지난 branch 출력은

\[
Y\in\mathbb R^{B\times64\times4\times4352}.
\]

## 7. Fixed simplex raw-tied head

decoder hidden을 normalize하지 않고 고정 embedding과 내적한다.

\[
\Lambda_{b,n,j,r}
=16\langle Y_{b,n,j},E_r\rangle,
\qquad
\Lambda\in\mathbb R^{B\times64\times4\times256}.
\]

\(E_r\)의 뒤 \(4096\)개 좌표는 \(0\)이므로 내적에는 \(Y\)의 앞
\(256\)개 좌표만 직접 들어간다. 별도 vocabulary weight, bias,
head RMSNorm은 없다.

## 8. Parameter 구성

| 구성 | parameter 수 | 상태 |
|---|---:|---|
| simplex embedding | \(1{,}114{,}112\) | 고정 |
| reversible block 2개 | \(2\times56{,}834{,}944\) | 학습 |
| encoder head RMS gain | \(4{,}352\) | 고정, 현재 head에서 미사용 |
| \(W_Q,W_K\) | \(2\times1{,}114{,}112\) | 학습 |
| \(W_V,W_O\) | \(2\times2{,}158{,}592\) | 학습 |
| \(\phi,\theta\) | \(2{,}176+128\) | 학습 |

\[
\text{전체 parameter}=121{,}336{,}064,\qquad
\text{학습 parameter}=120{,}217{,}600.
\]

exact-inverse decoder는 encoder parameter를 공유하므로 별도 parameter가 없다.
