# Parallel-scan H1--H16 curvature and cascade interpretation

## NLL 곡률의 부호

이 분석에서 `v_e = L_e - L_(e-1)`이므로 NLL이 감소할 때 속도는
음수다. 감소가 점점 느려지는 convex-up 곡선은
`a_e = v_e - v_(e-1) > 0`이다. 반대로 양수로 정의한 감소량
`I_e = -v_e` 자체의 변화는 `Delta I_e = -a_e`이므로 음수다.

epoch 1--10의 전체 추세에서는 flattening이 우세하다. H1--H16 모두
평균 2차 차분, interval velocity의 OLS slope, Theil--Sen slope가
양수다. block NLL은 `2.954417`에서
`2.835063`로 변했고, block의 8개
국소 2차 차분 중 `6`개가
양수였다. 전체 horizon-center 128개 중 양수는 `77`개
(`60.2%`), horizon별 median도 양수인 것은
16개 중 `14`개다. 따라서 장기적인 양의 NLL 가속도는
관측되지만 모든 checkpoint 사이에서 pointwise하게 양수라는 주장은
성립하지 않는다. H16은 9개 interval 중
`5`개에서만 감소해
깊은 horizon의 작은 후반 차분은 특히 진동성이 크다.

## 얕은 horizon에서 깊은 horizon으로의 cascade

순서 있는 학습 파동은 뚜렷하지 않다. 최대 개선 interval이 E1->E2인
horizon은 16개 중 `13`개이며 H2와 H16도 모두 여기에
포함된다. horizon 번호와 positive-improvement 시간 중심의 Spearman
상관은 `+0.126`, peak
interval midpoint와의 상관은
`-0.240`다. H2와
H16의 시간 중심은 각각
`3.507`와
`3.814`
epoch이다.

interval 개선량의 Pearson 상관에서 H2--H16 동시 상관은
`+0.449` (`n=9`), H2가 한 interval 선행할 때
`+0.271` (`n=8`), 반대 방향은
`+0.604` (`n=8`)다. 첫 개선 interval을 빼면 동시
상관은 `-0.030`이다. 따라서 이 tape는 H2가 H16을
선행한다고 raw interval 상관만으로 지지하지 않는다. H1의 동시 H16 상관
`+0.688`도 첫 interval 제거 후
`+0.052`로 약해지고, H1의 한-interval 선행 H16
상관은 `-0.021`이다.

가까운 horizon에는 제한적인 동조가 있다. H1의 한-interval 선행
상관은 H2에서 `+0.934`, H4에서
`+0.923`이지만 mean H13--H16에서는
`+0.006`이다. cosine learning-rate 면적으로 나눈
heuristic에서 H2->H16 선행 상관도 `+0.354` (`n=8`)에
그친다.

“얕은 개선이 다음 깊은 개선률의 가속을 만든다”는 직접 지표도
민감하다. `I_H1[t]`와 `I_H16[t+1]-I_H16[t]`의 상관은 전체 8개
transition에서 `-0.472`이지만 첫 transition을 빼면
`+0.224` (`n=7`)로 부호가 바뀐다. 이는 cascade의
양성 증거가 아니라 표본 수와 초기 interval에 대한 민감도다.

## 공통 diminishing-return envelope 제거

raw improvement의 early/late ratio 자체도 깊은 horizon에서 조금 더
크다는 약한 신호가 있다. 이 ratio와 horizon 번호의 Spearman 상관은
`+0.456`이고, H2는
`0.092`, H16은
`0.192`이다.
cosine-LR exposure로 나눈 peak-LR-equivalent 개선량은 H2가
`0.10480`에서
`0.06023`로
줄지만, H16은
`0.00511`에서
`0.00719`로
소폭 커진다.

여섯 common envelope에 대한 signed ratio `s_h/g`에서도 H16의 endpoint
late-minus-early와 OLS slope는 각각
`6/6`,
`6/6`에서
양수다. block envelope 대비 H16 ratio는
`0.115`에서
`0.324`로, shallow-H1--H4 mean 대비로는
`0.051`에서
`0.236`로 변한다. 이는 raw 감소율이 작아져도 H16의 상대적 비중이
커진다는 방향의 evidence다.

하지만 robust slope인 Theil--Sen이 양수인 envelope는
`3/6`뿐이다.
observed block, shallow mean, shallow median denominator의 최대/최소 비는
각각 `19.4x`,
`41.4x`,
`57.8x`여서 후반 ratio가 작은
분모와 checkpoint 진동을 증폭한다. all-horizon median도
`14.8x` 변한다. 9개 interval에
비해 자유도가 큰 power-law fit은 추가하지 않았고, smooth 대안은
log-linear block exponential 하나만 보존했다.

positive-only compositional share는 같은 방향을 더 직접적으로 보인다.
H1--H4의 early/late share는
`0.637`에서
`0.386`로 줄고, H5--H16은
`0.363`에서
`0.614`로 커진다. H9--H16은
`0.143`에서
`0.320`다. 다만 이 통계는 regression interval을 0으로 clip하는
구성비이므로 signed 개선을 대체하지 않는다. H13--H16도 endpoint
share는
`0.051`에서
`0.093`로 늘지만 Theil--Sen slope는
`-0.0009`여서 가장 깊은 group의 증가는
마지막 두 interval에 민감하다.

공통 envelope에 대해 각 series를 회귀한 residual의 one-interval
partial lead에서는 H2->H16이 여섯 envelope 모두 양수이고 범위는
`+0.282`--`+0.679` (`n=8`)다.
첫 transition을 빼도 범위는
`+0.287`--`+0.606`
(`n=7`)다. 이것은 raw envelope를 제거한 뒤의 일관된 부호라는 약한
힌트다. 그러나 H1->H16의 같은 범위는
`-0.439`--`-0.153`로 음수이며,
표본은 매우 작고 일부 envelope가 source 또는 target horizon을
포함한다. 따라서 residual 결과는 H2가 H16을 구동한다는 causal proof가
아니라 다음 matched ablation을 정당화할 수 있는 탐색적 신호다.

## 증거 범위

이 결과는 seed 1337의 단일 training trajectory와 고정 64-example
validation aggregate의 10개 epoch 지점에 한정된다. 인접 차분은 같은
checkpoint를 공유하고 horizon도 같은 모델과 validation token을
공유한다. cosine learning-rate 감소와 공통 초기 학습이 상관의
confound이며, 원 TSV에는 per-example NLL이 없어 표준오차를 복원할 수
없다. 따라서 상관은 기술통계이고 causal direction 또는 H2가 H16을
구동한다는 증거가 아니다. 정확한 interval, 곡률, horizon 요약과 모든
민감도 행은 각각의 TSV에 보존한다.
