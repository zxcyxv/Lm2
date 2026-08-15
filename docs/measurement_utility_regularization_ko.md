# Measurement Utility Regularization

## 상태

이 문서는 현재 branch-normalized unitary H16 모델에 추가할 수 있는
사전학습 정규화 제안이다. 아직 구현하거나 학습으로 검증한 결과가 아니다.

## 문제

현재 기본 목적함수는 각 horizon의 정답 byte에 대한 CE 평균이다.

\[
L_{\mathrm{CE}}
=
\frac{1}{H}\sum_{r=1}^{H}
\operatorname{CE}(\ell_r,y_r).
\]

이 목적은 각 recurrent measurement가 정답 형성에 유용해야 한다고 직접
요구하지 않는다. 이전 latent carrier가 이미 제공한 정보나 corpus의 고빈도
출력만으로 CE를 낮출 수 있다면, measurement correction이 새로운 정보를
제공하지 않거나 동일한 공통 mode를 반복해도 허용된다.

Step-6000 감사에서는 H8--H16의 여러 write amplitude가 강하게 보강되지만
출력은 공백으로 수렴했다. 따라서 보강·상쇄의 양 자체보다, 각 measurement가
정답 예측에 추가로 기여했는지를 직접 제약할 필요가 있다.

## 실제 상태와 counterfactual 상태

현재 한 reasoning step의 latent update를 다음과 같이 쓴다.

\[
\bar z_r=Rz_r,
\]

\[
\delta_r=O(m_r),
\]

\[
z_{r+1}=\bar z_r+\delta_r.
\]

여기서 `measurement utility`는 실제 successor와 correction을 제거한
counterfactual successor를 비교한다.

\[
z_{r+1}^{\mathrm{actual}}=\bar z_r+\delta_r,
\]

\[
z_{r+1}^{\mathrm{no\text{-}write}}=\bar z_r.
\]

두 상태는 같은 이전 carrier와 decoder를 사용하며, 차이는 현재
measurement correction의 존재뿐이다.

## 정답 logit margin

상태 \(z\)를 decode한 logits를 \(\ell(z)\)라 하고 정답 byte를 \(y\)라
한다. 정답 margin은 다음과 같이 정의할 수 있다.

\[
M(z,y)
=
\ell_y(z)-\max_{j\ne y}\ell_j(z).
\]

현재 correction의 utility는 실제 상태가 no-write 상태보다 정답 margin을
얼마나 개선했는지다.

\[
G_r
=
M(z_{r+1}^{\mathrm{actual}},y_r)
-
M(z_{r+1}^{\mathrm{no\text{-}write}},y_r).
\]

- \(G_r>0\): correction이 정답 선택을 개선했다.
- \(G_r<0\): correction이 기존 정답 방향을 손상했다.
- \(G_r\approx0\): correction이 현재 예측에 거의 기여하지 않았다.

## Utility loss

각 step이 최소 margin \(\mu\)만큼 유용하도록 hinge loss를 둘 수 있다.

\[
L_{\mathrm{utility}}
=
\frac{1}{H}\sum_{r=1}^{H}
\left[
\mu-
\left(
M(z_{r+1}^{\mathrm{actual}},y_r)
-
\operatorname{sg}
M(z_{r+1}^{\mathrm{no\text{-}write}},y_r)
\right)
\right]_+.
\]

전체 학습 목적은 다음과 같다.

\[
L
=
L_{\mathrm{CE}}
+\lambda_{\mathrm{utility}}L_{\mathrm{utility}}.
\]

여기서 `sg`는 stop-gradient다. No-write margin을 detach하지 않으면 모델이
실제 상태를 개선하는 대신 counterfactual baseline을 일부러 악화시켜 두
상태의 차이를 키울 수 있다. Detach는 no-write 경로를 고정 비교 기준으로
만들고, gradient가 실제 correction 경로를 개선하도록 제한한다.

## 현재 구조에서의 의미

이 항은 correction의 크기, memory norm, 보강간섭 또는 상쇄간섭을 직접
규제하지 않는다. 요구하는 것은 하나뿐이다.

> 현재 measurement correction은 같은 carrier를 그대로 운반했을 때보다
> 현재 horizon의 정답 byte를 더 잘 선택하게 해야 한다.

따라서 다음 경우를 구분한다.

- 유용한 보강간섭: 실제 정답 margin을 높이므로 허용·강화된다.
- 공백 공통 mode의 무차별 보강: 비공백 정답 margin을 높이지 못하면
  penalty를 받는다.
- 유용한 상쇄간섭: 경쟁 오답 logit을 낮춰 정답 margin을 높이면 허용된다.
- 필요한 정보를 제거하는 상쇄: 정답 margin을 낮추므로 penalty를 받는다.

보강은 좋고 상쇄는 나쁘다는 고정된 부호 규칙이 아니라, 정답에 대한
실제 기능으로 간섭을 평가한다.

## Decode 방법

Actual H1--H16 states는 현재처럼 하나의 causal exact-inverse tape로 decode할
수 있다. No-write states도 같은 prefix와 position을 사용하는 두 번째
causal tape로 묶어 병렬 decode할 수 있다.

```text
actual tape:   zbar_1 + delta_1, ..., zbar_H + delta_H
no-write tape: zbar_1,           ..., zbar_H
```

두 tape 사이에는 정보가 섞이면 안 된다. 각 tape 내부에서만 horizon-causal
mask를 사용해야 한다. 추가 decoder pass가 필요하므로 학습 연산량은
증가한다. 첫 실험에서는 utility를 모든 anchor가 아니라 일부 anchor 또는
H4 이후 horizon에만 적용하는 비용 절감도 비교할 수 있다.

## 초기 실험 권고

첫 ablation에서는 구조 변경을 최소화한다.

- 기존 CE와 H16 recurrence는 그대로 유지한다.
- actual/no-write 두 tape의 정답 margin만 비교한다.
- no-write margin은 detach한다.
- \(\lambda_{\mathrm{utility}}\)와 \(\mu\)는 작게 시작한다.
- CE-only control과 seed, split, batch, schedule을 맞춘다.
- 전체 block NLL뿐 아니라 horizon별 margin gain \(G_r\)을 TSV로 기록한다.
- H8--H16 공백률, Born cross/diagonal ratio와 실제 문장 생성을 함께
  감사한다.

성공은 단순히 utility loss가 낮아지는 것이 아니다. 다음이 동시에 필요하다.

1. H1 성능을 크게 손상하지 않는다.
2. H4 이후의 평균 \(G_r\)이 양수로 증가한다.
3. H8--H16 공백 collapse가 감소한다.
4. 전체 block NLL과 실제 생성 품질이 control보다 개선된다.
5. gradient norm과 recurrent scale이 안정적으로 유지된다.

## 실패 가능성

- 모델이 정답 margin을 과도하게 키워 logits만 날카롭게 만들 수 있다.
- 모든 step에 동일한 양의 margin을 요구하면 쉬운 step에도 불필요한
  correction을 만들 수 있다.
- 초기의 부정확한 no-write baseline 때문에 utility gradient가 noisy할 수
  있다.
- 추가 decode tape가 학습 비용과 VRAM을 늘린다.
- 각 horizon을 개별적으로 개선하려다 장기 공통 plan 성분을 약화할 수 있다.

따라서 이 항은 CE를 대체하지 않고 작은 보조항으로 사용해야 한다. 필요하면
H1--H4에는 낮은 weight, H5--H16에는 높은 weight를 두거나, no-write보다 이미
충분히 좋은 step에는 hinge가 자동으로 0이 되게 하는 방식으로 적용한다.

