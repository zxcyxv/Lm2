# Repository maintenance rules

이 저장소는 MSE+CE 이후 latent-orbit 실험의 최소 재현본이다.

- 변경 전 `README.md`, `docs/lineage.md`, 관련 experiment manifest를
  읽는다.
- 공통 모델·loss·평가 로직을 단발 experiment wrapper에 복제하지
  않는다.
- 실행 전 질문, 비교군, seed, split, 성공 기준을 manifest에 기록한다.
- metric은 TSV, 해석은 Markdown으로 분리한다.
- checkpoint, tokenized data, smoke artifact는 Git에 넣지 않는다.
- 기존 record와 producer는 대체 코드로 재현되기 전 삭제하지 않는다.
- 상충하는 evidence는 삭제하지 않고 상태와 범위를 명시한다.
