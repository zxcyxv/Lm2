# Epoch-2 H4 품질 비교: 13M 대 121M

## 결론

121M checkpoint는 epoch 2에서 13M보다 아직 나쁘다. 16-token gold 경계
rollout CE와 정확도가 모두 열세이고, 128-byte 자유생성 12개를 직접
읽어도 문장 품질이 좋아졌다고 판단할 수 없다.

121M은 13M보다 다양한 byte를 쓰고 같은 4-gram을 덜 반복한다. 그러나
그 차이는 더 좋은 문장으로 이어지지 않았다. 대부분 짧은 단어 조각 뒤에
`@`, 숫자, 공백 및 잘못된 UTF-8 byte 패턴으로 무너진다. 따라서 현재
결과는 "문장 생성 개선"이 아니라 "붕괴 형태가 덜 단조로워짐"으로
해석해야 한다.

## 16-token gold 경계 rollout

각 anchor는 literal gold prefix에서 시작한다. 그 뒤 16개 target 안에서는
모델이 네 token씩 open-loop로 예측하고, 그 네 greedy token을 다시 encode해
다음 네 token을 예측한다. 16-token 구간이 끝나면 생성 이력을 버리고 다음
gold anchor로 복원한다. 모델마다 16,384개 label을 동일하게 평가했다.

| model | H1--H16 평균 rollout CE | 평균 정확도 |
|---|---:|---:|
| 13M | 3.494414 | 20.3552% |
| 121M | 3.638637 | 18.9758% |

121M의 CE는 `+0.144224`, 정확도는 `-1.3794` percentage point다. 이 값은
생성한 token에 조건화된 gold-aligned rollout CE이므로 저장소의 관례적인
"block NLL"에 해당하지만, 엄밀한 teacher-forced likelihood는 아니다.

## 128-byte 자유생성

| metric | 13M | 121M | 판정 |
|---|---:|---:|---|
| gold-aligned byte accuracy | 14.0137% | 11.9873% | 121M 열세 |
| corpus byte entropy | 2.0738 bits | 2.3589 bits | 121M이 더 다양함 |
| sample당 unique bytes | 10.09 | 12.27 | 121M이 더 다양함 |
| repeated 4-gram fraction | 72.65% | 62.85% | 121M이 덜 반복함 |
| 평균 동일-byte 최장 run | 3.06 | 5.73 | 121M 열세 |
| 최대 동일-byte run | 4 | 118 | 121M에 심한 outlier 붕괴 |
| strict UTF-8 sample fraction | 100% | 90.625% | 121M 열세 |

121M은 prompt 간 출력의 획일성은 줄였지만, 국소 예측 entropy는 오히려
`2.3652 -> 2.2238` nats로 낮아졌다. 즉 더 넓은 조건별 문장 분포를 배운
것이라기보다 prompt마다 서로 다른 저품질 attractor에 들어가는 모습에
가깝다. 보고한 12개 sample 가운데 어느 쪽도 128 byte 동안 정상 문장을
유지하지 못했고, 121M 역시 명확한 가독성 개선이 없다.

## float32 prefix-shape audit

checkpoint는 훈련 때 사용한 full-context evaluator를 다시 실행했을 때
등록된 H1--H4 NLL을 최대 오차 `0.0`으로 재현했다. 따라서 잘못된
checkpoint나 모델 차원을 읽은 결과가 아니다.

반면 실제 block 재주입처럼 anchor마다 literal prefix를 다시 encode하면,
첫 네 horizon CE가 full-context encode-and-select 경로와 최대 `0.05257049`
달라졌다. rollout microbatch를 8에서 4로 바꿔도 차이는
`0.05327323`으로 유지됐다. 원인은 batch 크기가 아니라 width-4352
float32 exact-inverse 경로가 sequence-length에 따른 GEMM 모양에 민감한
것이다. 실제 생성은 prefix 재인코딩 경로를 사용하므로 본 비교에서는 그
경로의 값을 최종 결과로 채택하고, 차이 자체도 상충 evidence로 보존한다.

## 범위

이 결과는 seed 1337, validation prompt 64개, step 6,714, greedy block-4에
한정된다. 121M 모델이 더 학습되면 순위가 바뀔 수 있으며, 이 비교만으로
일반적인 parameter scaling 법칙을 주장하지 않는다. 현재 말할 수 있는
결론은 "같은 epoch 2에서는 121M이 13M을 따라잡지 못했다"까지다.
