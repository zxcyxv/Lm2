# 병렬 스캔 중앙 재귀 명세

## 기호

| 기호 | 의미 |
|---|---|
| \(h=1,\ldots,H\) | horizon 인덱스 (등록된 실험에서는 \(H=16\)) |
| \(z_0\in\mathbb R^{2d}\) | 중앙 recurrence의 root state |
| \(c_h\in\mathbb R^{2d}\) | Q/K/V를 미리 계산하기 위한 forcing state |
| \(z_h\in\mathbb R^{2d}\) | 실제 중앙 successor state |
| \(S_h\in\mathbb C^{A\times d_k\times d_v}\) | \(A\)개 head의 complex KV memory |
| \(R_\alpha\) | 실수 hidden pair마다 적용하는 직교 회전 |
| \(U_\phi\) | complex memory의 unit-modulus 회전 |
| \(N(\cdot)\) | fixed non-affine RMS normalization |
| \(\mathcal Q,\mathcal K,\mathcal V,\mathcal O\) | 중앙 Q/K/V projection 및 complex-read output projection |
| \(\epsilon\) | memory Frobenius 분모의 수치 안정화 상수 |

## 1. 고정 unitary transport

\[
R_\alpha
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}
=
\begin{bmatrix}
\cos\alpha_i & -\sin\alpha_i\\
\sin\alpha_i & \cos\alpha_i
\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix},
\qquad i=1,\ldots,d.
\]

\[
[U_\phi(S)]_{a,k,v}=e^{-\mathrm i\phi_{a,k}}S_{a,k,v},
\qquad
a=1,\ldots,A,\;k=1,\ldots,d_k,\;v=1,\ldots,d_v.
\]

\[
R_\alpha^{\mathsf T}R_\alpha=I,
\qquad
|e^{-\mathrm i\phi_{a,k}}|=1.
\]

## 2. 병렬 forcing tape

\[
c_h=R_\alpha^h z_0.
\]

\[
x_h=N(c_h).
\]

\[
q_h=\mathcal Qx_h\in\mathbb C^{A\times d_k},
\qquad
k_h=\mathcal Kx_h\in\mathbb C^{A\times d_k},
\qquad
v_h=\mathcal Vx_h\in\mathbb C^{A\times d_v}.
\]

\[
W_h=k_hv_h^\ast\in\mathbb C^{A\times d_k\times d_v}.
\]

\[
\bigl(c_1,\ldots,c_H\bigr),
\quad
\bigl(q_1,k_1,v_1\bigr),\ldots,\bigl(q_H,k_H,v_H\bigr),
\quad
\bigl(W_1,\ldots,W_H\bigr)
\]

은 모두 \(z_1,\ldots,z_H\)를 계산하기 전에 horizon 축으로 병렬 계산한다.

## 3. complex-memory prefix scan

\[
S_0=0,
\qquad
S_h=U_\phi(S_{h-1})+W_h.
\]

각 horizon의 affine memory map을

\[
M_h(S)=U_\phi(S)+W_h
\]

로 두면, 두 map의 시간 순서 합성은

\[
M_b\circ M_a
=
\bigl(U_\phi^2,\;U_\phi(W_a)+W_b\bigr)
\]

이고, 일반적으로

\[
(U_b,W_b)\circ(U_a,W_a)
=
\bigl(U_bU_a,\;U_bW_a+W_b\bigr).
\]

따라서 prefix scan의 \(h\)번째 결과가 정확히 \(S_h\)다.

## 4. memory read와 innovation tape

\[
r_h=\frac{q_h^\ast S_h}{\sqrt {d_k}\,\bigl(\lVert S_h\rVert_F+\epsilon\bigr)}
\in\mathbb C^{A\times d_v}.
\]

\[
\delta_h
=
\mathcal O\!\left(
\operatorname{concat}\bigl(\operatorname{Re}r_h,\operatorname{Im}r_h\bigr)
\right)
\in\mathbb R^{2d}.
\]

\[
\bigl(\delta_1,\ldots,\delta_H\bigr)
\]

은 memory prefix scan이 끝난 뒤 확정되는 latent innovation tape다.

## 5. latent prefix scan

\[
z^{\mathrm{scan}}_0=z_0,
\qquad
z_h=R_\alpha z_{h-1}+\delta_h.
\]

각 horizon의 affine latent map을

\[
T_h(z)=R_\alpha z+\delta_h
\]

로 두면,

\[
T_b\circ T_a(z)
=
R_\alpha^2z+R_\alpha\delta_a+\delta_b.
\]

pairwise-rotation parameterization으로는

\[
(\alpha_b,\delta_b)\circ(\alpha_a,\delta_a)
=
\left(
\alpha_a+\alpha_b,
R_{\alpha_b}\delta_a+\delta_b
\right).
\]

prefix composition을 \((\bar\alpha_h,\bar\delta_h)\)라 쓰면,

\[
z_h=R_{\bar\alpha_h}z_0+\bar\delta_h.
\]

최종 중앙 read state는

\[
\tilde z_h=N(z_h).
\]

## 6. 전체 중앙 계산 순서

\[
z_0
\xrightarrow{\;R_\alpha^h\;}
\{c_h\}_{h=1}^{H}
\xrightarrow{\;N,\mathcal Q,\mathcal K,\mathcal V\;}
\{q_h,k_h,v_h,W_h\}_{h=1}^{H}
\xrightarrow{\;\text{memory prefix scan}\;}
\{S_h\}_{h=1}^{H}
\xrightarrow{\;q_h^\ast S_h,\mathcal O\;}
\{\delta_h\}_{h=1}^{H}
\xrightarrow{\;\text{latent prefix scan}\;}
\{z_h,\tilde z_h\}_{h=1}^{H}.
\]
