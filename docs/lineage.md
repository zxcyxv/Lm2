# MSE+CE 이후 실험 계보

이 문서는 새 저장소에 보존한 producer만 요약한다. 수치는
`experiments/records/`의 원 기록을 기준으로 한다.

## 1. Query K=1 MSE+CE

- `train_query_k1_mse_ce_joint_13m.py`
- `train_query_k1_mse_ce_joint_nodetach_13m.py`
- `train_query_k1_orthogonal_mse_ce_joint_13m.py`
- Householder/ActNorm 및 fixed/learned-noise 변형

Query-branch state와 실제 token-prefix state의 domain 차이, target
stop-gradient, 가역 operator 선택을 분리한 단계다.

## 2. hA-domain skew operator

- `train_k1_ha_skew_learned_noise_mse_ce_13m.py`
- `train_k1_ha_skew_fixed_noise_sigma001_13m.py`
- `train_k1_ha_skew_iresnet_spectral_learned_noise_13m.py`

`K`의 입력을 실제 causal state `h_A`로 옮기고
`K=exp(A-A^T)` 및 exact inverse decoder를 사용했다. i-ResNet식
spectral constraint와 noise tolerance 분석도 이 단계에 속한다.

## 3. Window-3 trajectory supervision

- `train_k3_ha_skew_clean_window3_mse_ce_13m.py`
- stride-1/batch-64 wrapper
- target stop-gradient와 truncated-gradient ablation
- learned compounding-noise ablation

`K h_A`, `K^2 h_A`, `K^3 h_A`를 한 causal inverse tape에서
학습하고 anchor stride와 horizon 간 gradient 경로를 비교했다.

## 4. Three-trajectory CE competition

- `train_k4_ha_skew_three_trajectory_detached_ce_13m.py`
- `train_k4_ha_skew_three_trajectory_detached_ce_tau005_shared_13m.py`
- `train_k4_ha_skew_three_trajectory_detached_ce_tau005_shared_h1_online_mse_13m.py`

세 noisy trajectory 각각에 대해 네 토큰 CE 합을 계산한다.

\[
C_i=\sum_{j=1}^{4}\operatorname{CE}_{ij},\qquad
w_i=\operatorname{sg}\operatorname{softmax}_i(-C_i/0.05)
\]

\[
L=\mathbb{E}_{A}\left[\sum_i w_i C_i/4\right].
\]

실제 prefix, clean K orbit, sigma predictor와 inverse-prefix 계산은
trajectory 사이에 공유된다. literal prefix expansion과 logits 및
gradient가 일치하는지 단위 테스트로 검증한다.

마지막 wrapper는 동일한 tau-0.05 trajectory CE에 clean h1 정합만
추가한다.

\[
L=L_{\mathrm{trajectory\ CE}}+
\frac{\lVert Kh_A-h_{B,\mathrm{online}}\rVert^2}
{\lVert h_{B,\mathrm{online}}\rVert^2}.
\]

`hB_online`은 `hA`와 같은 현재 encoder pass에서 만들며 detach하지
않는다. h2--h4 MSE와 EMA target은 사용하지 않는다. 보존된
microbatch-32 step-1000 record의 best validation marginal NLL은
`5.884858`이다. CE-only matched record는 `5.787555`다.

## 5. Initial-condition trajectory와 prefix prior

- `train_k4_ha_skew_initial_condition_three_trajectory_prior_tau005_h1_online_mse_13m.py`
- `src/rotlm/models/trajectory_initializer.py`

독립 horizon noise를 하나의 learned context-conditioned initial
condition으로 바꿨다.

\[
\eta_i=R(\operatorname{sg}(h_A),z_i),\qquad
u_{i,1}=Kh_A+\eta_i,\qquad
u_{i,j}=K^{j-1}u_{i,1}.
\]

prefix-only prior \(\pi_i\)를 추가하고 네 token CE 합과 prior에서
detached posterior를 만든다.

\[
q_i=\operatorname{sg}
\operatorname{softmax}_i(\log\pi_i-C_i/0.05).
\]

loss는 posterior-weighted path CE, clean h1 attached online MSE,
\(\operatorname{KL}(q\|\pi)\)의 합이다. branch-state h2--h4 MSE와
후속 innovation은 없다.

보존된 microbatch-16/effective-batch-64 step-1000 record에서:

- validation marginal NLL: `5.638737`
- prior winner accuracy: `0.428070`
- clean h1 relative MSE: `0.013768`
- branch-distance preservation error: `2.24e-6`

step-1000 greedy generation에서는 initial condition이 clean orbit의
동일 token fixed-point collapse를 줄였지만, `prior_block4`의
period-4 repeat가 `0.4951`로 높아 4-position phase cycle이
관측됐다. 또한 후속 audit에서 기록된 `prior_ar`는 매 token prior와
residual을 다시 계산하고 `prior_block4`는 네 token 동안 유지하므로,
둘의 비교는 순수 re-anchoring ablation이 아님이 확인됐다.

설계 논쟁과 evidence 범위는
[research_debate/README.md](research_debate/README.md)에 별도로
보존한다.

## 6. CE-only exact-inverse AR fixed-point audit

- `train_k1_exact_inverse_ce_only_solver_replication_13m.py`
- `eval_k1_exact_inverse_ce_only_parallel_solver.py`
- record:
  `EXP-20260727-k1-exact-inverse-ce-only-solver-replication-13m`

width-896 reversible encoder, bias-free linear \(K\), exact-inverse decoder,
RMS-tied head를 표준 next-token CE만으로 1000 step 학습했다. 별도의
latent MSE, closure, branch, KL 또는 distillation은 없다.

추론에서는 \(K^1h_A,\ldots,K^nh_A\)의 joint latent decode를 초기
미래 block으로 삼고, 원래 AR conditional을 모든 위치에서 동시에
적용하는 Jacobi fixed-point update를 반복했다. 32 validation
prompt에서 greedy AR와 전체 block이 정확히 같아지는 iteration은
block 4, 8, 16, 32에 대해 각각 3, 7, 15, 31이었다.

이는 첫 latent token이 AR 첫 token과 구조적으로 같고, causal
triangular update가 확정 prefix를 iteration마다 한 token씩 늘리는
귀납적 결과다. 따라서 AR trajectory의 병렬 fixed-point exact 복원은
확인됐지만, 현재 solver는 block 길이에 선형인 iteration을 요구해
속도 이득은 확인되지 않았다.

## 7. Self-predicted complex KV와 EMA latent target

- `train_byte256_complex_self_predicted_kv_h1_online_mse_ce_13m.py`
- `train_byte256_complex_self_predicted_kv_full_innovation_h1_online_mse_ce_13m.py`
- `train_byte256_complex_self_predicted_kv_h1_online_sg_mse_ce_13m.py`
- `train_byte256_complex_self_predicted_kv_h1_ema_sg_mse_ce_13m.py`
- `train_byte256_complex_self_predicted_kv_post_update_full_read_h1_ema_sg_mse_ce_13m.py`
- `eval_byte256_complex_self_predicted_kv_post_update_residual_shapley.py`
- `eval_byte256_complex_self_predicted_kv_decode_sharpness.py`

중앙 recurrence는 실제 또는 생성 token을 받지 않고 자신의 continuous
prior에서 rank-one complex KV innovation을 만든다. 현재 CE readout은
innovation-free prior만 decode하며, full innovation state는 다음 recurrent
state와 H1 latent 회귀 예측으로 사용한다. beta와 별도 vocabulary head는
없다.

attached online target은 step 1000에서 mean central/AR state cosine
`0.945871`을 얻었지만 H2--H4 token agreement는 `0.261149`였다. target
encoder 출력에 stop-gradient를 적용하자 cosine은 step 50부터 `0.650909`
으로 낮아져, 기존 높은 cosine에 target-coordinate co-adaptation이
포함됐음을 드러냈다.

후속 EMA-SG run은 step 100까지 exact-copy하고 이후 full-model decay
`0.99`를 적용했다. 추론은 같은 EMA snapshot의 encoder, transition,
exact inverse decoder를 사용했다. step 1000 EMA H1 NLL은 `1.393542`,
H2--H4 agreement는 `0.272135`, exact four-token agreement는 `0.024658`다.
평균 agreement는 attached reference를 소폭 넘었지만 H4 agreement는
`0.120605`에 그쳐 exact AR equivalence는 성립하지 않았다.

후속 post-update full-read run은 current CE state `P`는 그대로 두고 다음
recurrent hidden만 새 query가 전체 updated memory를 다시 읽도록 바꿨다.
step 1000 EMA H2--H4 agreement는 `0.242269`, exact block agreement는
`0.009277`로 direct EMA control보다 낮았다. step 200 이후 aggregate
agreement도 단조증가하지 않았지만 H2 alone은 모든 등록 지점에서
증가했다.

read-only residual/Shapley audit에서는 H1 innovation이 canonical-AR hidden
오차를 `99.73%`의 행에서 줄였지만, H2에서는 평균적으로 거의 중립,
H3/H4에서는 방향적으로 해로웠다. 같은 weights를 old recurrent 식으로
되돌리면 H2--H4 agreement가 `0.242269`에서 `0.193604`로 더 낮아져,
coherent full reread 자체보다 반복 innovation closure가 병목임을
확인했다.

별도 decode-sharpness audit에서 H1 `P`/`Znext` top-1 agreement는
`0.400879`였다. local pre-innovation state 대비 innovation의 net P-basin
gain은 약 25.0%p였으나, Znext mean max-softmax probability는 `0.460469`,
99% 이상 비율은 `0.130615`에 그쳤다. canonical hidden의 mean confidence
`0.999973`과 달리 Znext는 아직 realization-sharp future hidden으로 볼 수
없다. H2--H4 sharpness는 보조지표로만 보존한다.

## 8. H16 boundary recurrence와 initial-root RMSNorm

- `train_byte256_complex_self_predicted_kv_urm_boundary_postnorm_no_qkv_prenorm_h16_attached_stride16_ce_only_13m.py`
- `train_byte256_complex_self_predicted_kv_urm_boundary_postnorm_no_qkv_prenorm_initial_root_rmsnorm_h16_attached_stride16_ce_only_13m.py`
- `eval_byte256_complex_self_predicted_kv_initial_root_rmsnorm_anchor0_h_monitor.py`

no-QKV-pre-norm H16 control은 재사용 직전의 complete successor `Z/S`만
fixed non-affine RMS 정규화하고, initial encoder-root 복사본과 내부
`P/W/Sraw/Zraw`는 raw로 두었다. 이 control은 step 900에서 raw global
gradient norm `1563.952271`, block validation NLL `11.598250`의 transient를
보였고, step-1000 감사가 H5/H6의 정렬된 민감도와 anchor-0 byte 104
(`h`) 경로를 좁혔다.

후속 단독 개입은 raw encoder 출력과 exact inverse 경로는 유지한 채
private recurrent initialization만
`Z0=fixed_RMS(encoder_root)`, `S0=write(Z0)`로 바꿨다. 같은 step 900에서
gradient norm은 `2.388309`, block NLL은 `3.102862`였고, step 1000은 각각
`1.824580`, `3.095199`로 끝나 모든 등록 기준을 통과했다. matched raw-root
control의 최종 block NLL은 `3.167410`이었다. 다만 step 600 candidate는
control보다 gradient와 NLL이 모두 나빴다가 step 700에 회복했으므로,
전 구간의 단조 우월성은 주장하지 않는다.

고정 validation 256-window monitor에서 step-1000 control의 byte-h H6
innovation-W와 raw-successor-Z는 non-h 행보다 `3.79x`, `9.86x` 컸지만,
candidate는 `0.955x`, `0.976x`였다. candidate의 H2--H16 P/W/Zraw/Sraw
byte-h/non-h mean ratio 최대는 `1.053x`여서 등록된 late-horizon outlier가
다른 horizon으로 이동한 증거는 없었다. 동시에 candidate 전체 trajectory
scale도 크게 낮아졌으므로, 이 monitor만으로 유일한 인과 경로를 확정하지
않는다. count-dependent `1/sqrt(write_count)` read scaling은 이 실험에
추가하지 않았으며 별도 ablation 범위로 남긴다.

## 9. Fused H16 stochastic value-write와 ray velocity

- `train_byte256_unitary_stochastic_value_ray_flow_h16_300.py`
- `eval_byte256_unitary_stochastic_value_ray_flow_corrected_noise.py`
- record:
  `EXP-20260803-byte256-unitary-stochastic-value-ray-flow-h16-300step`

완료된 fused time-varying scan step-33570 checkpoint에서 complex process
noise를 value 쪽 저랭크 write
`k(v + 0.05 epsilon)^dagger`로 넣었다. noise tape를 scan 전에 전부
샘플링하므로 literal recurrence와 rotating-frame associative scan은
바뀌지 않는다. deterministic CE, stochastic CE, stochastic CE plus
H1--H4 detached sample-path ray-velocity loss를 동일 데이터 순서로 300
step 비교했다.

세 팔의 최종 raw gradient norm은 모두 `0.59` 미만이었다. ray 보조항은
stochastic-CE 대비 H1--H4 ray cosine을 `0.078722`에서 `0.086485`로
올렸고 deterministic block NLL 차이는 `0.000079`였다. 그러나 corrected
same-full-encode audit에서 noise logit RMS는 parent `0.051129`에서 CE-only
`0.049792`, ray 보조항 `0.049362`로 감소했다. 따라서 이 규모에서는
process noise가 안정적인 stochastic write로 작동했지만 미래 mode를
선택하는 변수로 사용되지는 않았다. 분포적 FM 주장은 명시적
noise-to-future coupling 또는 bridge loss 전까지 보류한다.

## 범위 밖

MSE+CE 이전 rotation toy model, 중앙 planner, Mamba/ScanLM,
random-tape compiler, paper용 단발 스크립트는 이 저장소에서 제외했다.
