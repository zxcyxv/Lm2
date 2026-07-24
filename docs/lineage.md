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

## 범위 밖

MSE+CE 이전 rotation toy model, 중앙 planner, Mamba/ScanLM,
random-tape compiler, paper용 단발 스크립트는 이 저장소에서 제외했다.
