# CE-only AR의 병렬 고정점 복원

## 질문

표준 teacher forcing CE만으로 학습한

\[
\text{encoder}\rightarrow K\rightarrow\text{exact-inverse decoder}
\]

모델을 그대로 두고, 추론에서만 미래 token block을 병렬 갱신하면
동일 모델의 greedy AR 궤적을 복원할 수 있는가? 가능하다면 필요한
iteration은 몇 번인가?

## 실험 계약

`EXP-20260727-k1-exact-inverse-ce-only-solver-replication-13m`은
width 896, 두 reversible causal block, 하나의 bias-free 선형 \(K\),
exact-inverse decoder와 RMS-tied head를 표준 next-token CE만으로
1000 step 학습했다. MSE, closure, KL, branch, noise, distillation은
사용하지 않았다.

검증에서는 동일 checkpoint의 literal greedy AR 출력을 정답 궤적으로
정했다. 먼저 한 실제 prefix의 마지막 state \(h_A\)에서

\[
Kh_A,K^2h_A,\ldots,K^nh_A
\]

를 만들고 joint inverse decode하여 미래 block의 초기 token
\(y^{(0)}\)을 얻었다. 이후 한 iteration은 후보 block 전체를 causal
teacher-forced pass 하나로 평가하고 모든 위치를 동시에 갱신했다.

\[
y_t^{(r+1)}
=
\arg\max p_\theta(\cdot\mid A,y_{<t}^{(r)}).
\]

이는 새 loss나 학습된 solver가 아니라, 원래 AR 조건부 분포가
정의하는 Jacobi fixed-point iteration이다.

## 관측

32개의 고정 validation prompt에서 얻은 결과는 다음과 같다.

| block \(n\) | latent 초기 일치율 | 99% 일치 | 전체 exact |
|---:|---:|---:|---:|
| 4 | 25.00% | 3 | 3 |
| 8 | 14.84% | 7 | 7 |
| 16 | 7.42% | 15 | 15 |
| 32 | 3.71% | 30 | 31 |

마지막 token 반복 초기값 control은 전체 exact에 각각
4, 8, 16, 32 iteration이 필요했다. 따라서 latent orbit은 정확히
한 iteration 정도를 절약했지만, iteration 수의 block 길이에 대한
선형 scaling은 바꾸지 못했다.

원 metric과 해석은
[solver audit record](../../experiments/records/EXP-20260727-k1-exact-inverse-ce-only-solver-replication-13m/solver_audit/)
에 있다.

## 왜 정확히 \(n-1\)인가

latent initializer의 첫 token은 근사가 아니라 같은 계산 계약 때문에
greedy AR의 첫 token과 동일하다.

\[
y_1^{(0)}
=\arg\max D(Kh_A)
=y_1^{AR}.
\]

이후 이전 iteration에서 앞의 \(m\)개 token이 AR과 같다면, 다음
iteration에서 \(m+1\)번째 위치가 보는 전체 prefix도 AR과 같다.
그러므로 그 위치의 argmax 역시 반드시 AR과 같다.

\[
y_{1:m}^{(r)}=y_{1:m}^{AR}
\quad\Longrightarrow\quad
y_{1:m+1}^{(r+1)}=y_{1:m+1}^{AR}.
\]

귀납적으로 iteration 0에서 한 token, iteration \(r\)에서 적어도
\(r+1\)개 token이 확정된다. 따라서 길이 \(n\) block은 늦어도
\(n-1\)회에 정확히 복원된다. control이 \(n\)회를 요구한 것도 첫
token이 초기 상태에서 보장되지 않기 때문이다.

먼 위치가 더 이른 iteration에 우연히 AR token과 같을 수는 있다.
하지만 잘못된 앞 prefix에서 얻은 일치이므로 앞 token이 바뀔 때 다시
바뀔 수 있다. 이를 causal frontier를 넘어선 확정 일치로 해석하면
안 된다.

## 정립된 근간과 현재 한계

이 실험은 다음을 직접 보였다.

1. 표준 CE AR 모델의 greedy 생성은 병렬 token block 위의 결정론적
   고정점 문제로 재표현할 수 있다.
2. 별도 distillation, closure 또는 latent target 없이도 반복 병렬
   갱신으로 원래 AR 궤적을 정확히 복원할 수 있다.
3. exact 복원 가능성과 계산 가속은 별개의 명제다.
4. 단순 Jacobi 갱신에서는 causal dependency가 정답 frontier를
   iteration마다 한 칸만 확정하므로 유의미한 asymptotic speedup이
   없다.

따라서 다음 설계의 핵심 지표는 최종 \(n-1\) 수렴 여부가 아니다.
iteration \(r\)에서 구조적으로 보장되는 앞 \(r+1\)개를 제외하고도,
먼 위치의 예측이 앞 prefix 수정에 안정적으로 유지되는지, 즉
**causal frontier를 한 번에 여러 칸 전진시키는가**를 측정해야 한다.

현재 관측의 범위는 fixed global linear \(K\), one-step CE,
Jacobi token update다. 입력 의존적 transition, spectral coupling,
multigrid 또는 parallel-prefix solver의 가능성을 반증하지 않는다.
