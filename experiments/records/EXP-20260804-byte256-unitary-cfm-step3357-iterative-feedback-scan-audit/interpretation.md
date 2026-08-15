# Iterative feedback-scan audit 해석

## 결론

추가 rescan은 순차 feedback을 매우 빠르게 근사했다. 세 checkpoint
모두에서 H4--H16 raw-state 오차, feedback logit KL, fixed-point defect가
`compiled_scan -> rescan_1 -> rescan_2` 순서로 감소했다. `rescan_2`의
tail top-1은 모든 checkpoint에서 순차 feedback과 100% 일치했다.

그러나 이 solver 정합성은 validation NLL 개선으로 이어지지 않았다.
세 checkpoint 모두 feedback 방향으로 이동한 NLL 차이의 paired 95%
구간이 0을 포함했다. 따라서 이 audit은 “적은 rescan으로 feedback을
복원할 수 있다”는 계산적 결과는 지지하지만, “현재 scan-trained
weights에서 feedback 복원이 언어 품질을 높인다”는 주장은 지지하지
않는다.

## 집계 결과

| checkpoint | mode | block NLL | tail state rel. RMS | tail KL to feedback | tail defect | median ms | scan 대비 |
|---|---|---:|---:|---:|---:|---:|---:|
| CFM 3357 | scan | 2.952514 | 1.184e-2 | 1.355e-4 | 1.204e-2 | 11.862 | 1.000x |
| CFM 3357 | +1 | 2.952568 | 3.175e-4 | 1.134e-7 | 3.223e-4 | 13.097 | 1.104x |
| CFM 3357 | +2 | 2.952568 | 5.618e-6 | 9.096e-11 | 5.717e-6 | 14.406 | 1.214x |
| CFM 3357 | feedback | 2.952568 | 0 | 0 | 0 | 20.449 | 1.724x |
| CE 3357 | scan | 2.954462 | 8.432e-3 | 2.513e-4 | 8.810e-3 | 11.999 | 1.000x |
| CE 3357 | +1 | 2.954469 | 4.453e-4 | 5.823e-7 | 4.616e-4 | 13.129 | 1.094x |
| CE 3357 | +2 | 2.954463 | 1.690e-5 | 8.921e-10 | 1.737e-5 | 14.442 | 1.204x |
| CE 3357 | feedback | 2.954464 | 0 | 0 | 0 | 20.542 | 1.712x |
| CE 33570 | scan | 2.835094 | 8.606e-4 | 7.617e-5 | 8.838e-4 | 11.831 | 1.000x |
| CE 33570 | +1 | 2.835268 | 2.977e-5 | 5.602e-8 | 3.018e-5 | 13.186 | 1.115x |
| CE 33570 | +2 | 2.835264 | 6.640e-7 | 3.057e-11 | 6.253e-7 | 14.518 | 1.227x |
| CE 33570 | feedback | 2.835264 | 0 | 0 | 0 | 20.786 | 1.757x |

세 checkpoint 평균 full-forward latency는 scan `11.897 ms`, +1
`13.137 ms`, +2 `14.456 ms`, sequential feedback `20.592 ms`였다. +1과
+2는 scan보다 각각 약 10.4%, 21.5% 느리지만, 순차 feedback보다는
각각 약 36.2%, 29.8% 빨랐다. 이 timing은 고정 4-example GPU-resident
입력의 encoder-central-decoder 전체 호출이며 checkpoint load와 Triton
compile은 제외한다.

## NLL 판정

Feedback의 scan 대비 block-NLL 점 추정 변화는 CFM-3357
`+0.0000544`, CE-3357 `+0.0000017`, CE-33570 `+0.0001694`였다. 각각의
paired 95% 구간은 `[-0.0002329,+0.0003417]`,
`[-0.0003963,+0.0003997]`, `[-0.0001003,+0.0004392]`로 모두 0을
포함한다. horizon별 차이도 부호가 섞였다. 현재 weights는 scan 실행
계약으로 학습됐으므로 feedback에 가까워지는 것과 NLL이 낮아지는 것은
별개의 결과다.

## 구조 검사와 수치 범위

등록된 exact-prefix max-logit 기준 `2e-4`는 CFM-3357과 CE-3357에서는
통과했지만, CE-33570에서 `2.4414e-4`로 소폭 실패했다. 별도 확인에서
같은 mature checkpoint의 eager scan도 H1 max error `2.8992e-4`였고,
mean/RMS error는 각각 `1.43e-5`/`2.26e-5`였다. 따라서 이는 rescan의
인덱싱 실패 증거가 아니라, 성숙한 decoder가 작은 float32 연산순서
차이를 일부 logit에서 증폭한 범위다. 사전등록 기준은 사후에 완화하지
않으며 이 한계와 함께 보존한다.

## 보존할 상충 evidence

CFM step-3357 checkpoint의 재평가 scan NLL `2.952514`는 기존 record의
`2.952489`와 `2.5e-5` 차이로 일치한다. 반면 CE step-3357 checkpoint의
재평가 `2.954462`는 기존 metrics row `2.951368`과 다르다. 동일
checkpoint를 기존 공통 evaluator로 microbatch 4/16/64에서 다시
확인한 값은 각각 `2.954462`, `2.954594`, `2.954524`였으므로 단순
microbatch 차이는 아니다. 기존 row를 덮어쓰지 않고, 본 audit의 모든
arm은 실제 보존 checkpoint에서 동일한 roots/targets로 계산되었다는
범위로 해석한다.

## 다음 판단

현재 단순 Picard rescan은 이미 +1에서 대부분의 feedback 차이를 없애고
+2에서 실질적으로 수렴한다. 따라서 이 checkpoint들에는 full
quasi-Newton Jacobian을 추가할 실익이 작다. 다음 유효한 실험은
`rescan_1`을 학습 forward contract로 사용해 모델이 feedback edge를
실제로 활용하도록 학습한 뒤, 같은 계산량의 scan-only 모델과 NLL을
비교하는 것이다. 지금 결과만으로 noise, control-energy, PMP loss를
추가할 근거는 없다.
