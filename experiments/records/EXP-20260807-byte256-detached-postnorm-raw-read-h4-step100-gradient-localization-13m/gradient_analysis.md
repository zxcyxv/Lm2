# Step-100 gradient localization

이 문서는 고정된 training-split 진단 배치만 보고한다. 실제 optimizer batch의 학습 지표는 `metrics.tsv`에 분리되어 있다.
각 Hk CE는 64 examples x 64 anchors = 4,096 labels의 평균이고, `mean_h1_h4`는 총 16,384 labels의 평균이다. 따라서 학습 gradient는 `(g_H1 + g_H2 + g_H3 + g_H4) / 4`이며, 각 Hk 표는 그 1/4 계수를 적용하지 않은 개별 손실이다.

## Separate loss sources

| loss | CE | full-model parameter gnorm | tied-central gnorm |
|---|---:|---:|---:|
| H1 | 4.66242903 | 154.41868974 | 104.60406951 |
| H2 | 4.68452847 | 87.85084487 | 57.06954137 |
| H3 | 4.62286222 | 100.39286164 | 67.95568399 |
| H4 | 4.85385883 | 135.24325756 | 95.10766734 |
| mean_h1_h4 | 4.70591986 | 89.52557644 | 59.67692380 |

## Central parameter gradient by recurrence use

값이 동일한 감사용 사본 네 개의 gradient다. 각 행에서 네 gradient vector를 파라미터별로 더하면 실제 tied gradient가 된다.

| loss | step 1 | step 2 | step 3 | step 4 |
|---|---:|---:|---:|---:|
| H1 | 104.59993252 | 0.00000000 | 0.00000000 | 0.00000000 |
| H2 | 23.97401934 | 50.18036165 | 0.00000000 | 0.00000000 |
| H3 | 21.63042102 | 16.52625781 | 56.99226530 | 0.00000000 |
| H4 | 24.47589135 | 18.96516539 | 15.81506680 | 76.80571002 |
| mean_h1_h4 | 31.74039607 | 16.63127451 | 15.61921957 | 19.20142750 |

관측된 causal support는 정확히 삼각형이다: H1은 use 1만, H2는 use 1--2, H3는 use 1--3, H4는 use 1--4에 도달했다. 따라서 H4가 네 재귀 use 모두에 영향을 준다는 가정은 맞다. hidden detach는 이 사실을 없애지 않는다. 이전 write에서 미래 read로 이어지는 live complex-memory 경로와 causal inverse decoder tape 경로가 남아 있기 때문이다.

## H4 central components by recurrence use

| use | all central params | Q | K | V | O |
|---:|---:|---:|---:|---:|---:|
| 1 | 24.47589135 | 0.13655283 | 4.58980465 | 24.03223801 | 0.53102458 |
| 2 | 18.96516539 | 0.10821849 | 5.61784840 | 18.09856224 | 0.66296095 |
| 3 | 15.81506680 | 0.16336782 | 5.26327705 | 14.88326740 | 0.87998718 |
| 4 | 76.80571002 | 13.98105335 | 5.25433683 | 14.52599144 | 73.92456055 |

use 4의 local Q/O gradient가 크고, 이전 use의 Q/O는 작다. 반대로 과거 write를 구성하는 K/V에는 H4 gradient가 계속 도달한다. 이는 hidden feedback edge는 잘렸지만 memory credit-assignment edge는 살아 있다는 구현과 일치한다.

## H4 recurrent-state adjoints

| use | rotated hidden | successor hidden | write memory | successor memory |
|---:|---:|---:|---:|---:|
| 1 | 0.0192005712 | 0.0020572396 | 0.0060129751 | 0.0060129751 |
| 2 | 0.0141721561 | 0.0025582598 | 0.0060130130 | 0.0060130130 |
| 3 | 0.0117182113 | 0.0032838891 | 0.0060155163 | 0.0060155163 |
| 4 | 0.0170561080 | 0.2572076369 | 0.0060194615 | 0.0060194615 |

H4 successor-memory adjoint는 use 1--4에서 0.0060129751--0.0060194615, 즉 최대/최소 차이가 0.1079%다. unitary memory carry를 거슬러 갈 때 누적 증폭은 관측되지 않았다. successor-hidden adjoint가 use 4에서 크고 이전 use에서 작아지는 것은 hidden recurrent edge가 detach되어 있기 때문이다. 이전 use의 작은 nonzero 값은 decoder tape의 직접 causal attention 경로다.

## Layer localization

| objective | encoder block 1 | encoder block 2 | central tied | full model |
|---|---:|---:|---:|---:|
| H1 | 19.41855774 | 111.91979270 | 104.60406951 | 154.41868974 |
| H2 | 11.46764070 | 65.79765657 | 57.06954137 | 87.85084487 |
| H3 | 13.13651373 | 72.71989886 | 67.95568399 | 100.39286164 |
| H4 | 16.95324640 | 94.64596012 | 95.10766734 | 135.24325756 |
| mean_h1_h4 | 11.59001834 | 65.72035512 | 59.67692380 | 89.52557644 |

학습과 같은 mean objective에서 큰 항은 encoder block 2 attention (`65.65860747`), central output (`47.84637791`), central value (`34.29835561`)다. H4만 보면 encoder block 2 attention `94.47067920`, central output `73.94878203`, central value `55.57123016`이다. 세부 named-parameter 값은 `parameter_gradients.tsv`에 있다.

## Interpretation

H1 단독 global gnorm은 154.41868974, H4 단독은 135.24325756다. 한 번만 쓰는 H1이 H4보다도 크므로, 이 체크포인트의 큰 global gnorm을 네 번의 recurrent Jacobian 곱 자체로 설명할 수 없다. H4에서도 과거 use로 갈수록 central-use gnorm이 커지는 패턴은 없고, local use 4가 가장 크다.

H4 use별 central norm의 root-sum-square는 84.30883812, 실제 tied vector-sum norm은 95.10766734 (`1.1281x`)다. 공유 파라미터에 여러 use의 gradient가 합쳐지는 양의 정렬 효과는 있지만, 네 scalar norm을 그대로 더한 값과는 다르며 이것만으로 Jacobian 폭발이라고 부를 수 없다.

네 개별 horizon global norm의 산술평균은 119.47641345지만 실제 mean-loss gnorm은 89.52557644 (`0.7493x`)다. 서로 다른 horizon gradient의 방향이 완전히 정렬되지 않아 평균 과정에서 일부 상쇄되었다.

결론적으로 step 100에서 큰 norm의 주 위치는 마지막 encoder attention과 local central output/value projection이다. H4의 credit는 네 use 모두에 도달하지만, memory adjoint는 거의 등척이고 hidden adjoint의 역방향 연쇄는 detach로 끊겨 있다. 따라서 이 한 지점에서는 recurrent Jacobian explosion의 증거보다 고차원 공유 파라미터에 대한 정상적인 gradient 합산과 local projection scale의 증거가 강하다. 단일 checkpoint 감사만으로 장기 학습 안정성을 증명하지는 않는다.

## Mechanical checks

- Untied/tied forward-logit maximum absolute error: 0
- Target mismatches: 0
- Maximum absolute error between a tied central gradient and the sum of four use gradients: 9.53674316e-07
- Maximum relative max-element error for that check: 1.98578233e-07

상세 named-parameter, layer-group, recurrence-use, intermediate-tensor 값은 인접 TSV에 있다.
