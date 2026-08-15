# 실수 value와 복소 value의 동역학적 차이

## 범위

이 문서는 [제미나이대화내용.txt](./제미나이대화내용.txt)의 원래 식을
기준으로 한다.

\[
S_t=US_{t-1}+k_tv_t^\dagger,
\qquad
o_t=\operatorname{Re}(q_t^\dagger S_t)
\]

여기서는 \(U\)가 key 축에만 작용하고, 최종 관측이 실수부만 취하는
경우에 한정해 \(v\)가 실수일 때와 복소수일 때를 비교한다.

과거 write 하나의 현재 read 계수를 다음과 같이 둔다.

\[
c_{t,n}=q_t^\dagger U^{t-n}k_n
\]

## 실수 value

\(v_n\in\mathbb R^{d_v}\)이면 해당 write의 출력 기여는

\[
\operatorname{Re}(c_{t,n}v_n^\top)
=\operatorname{Re}(c_{t,n})v_n^\top
\]

이다. 시간에 따라 변하는 것은 하나의 실수 검색 계수뿐이다. 따라서
\(v_n\)의 방향과 내용은 고정되고, \(U\)는 그 내용을 얼마나 강하게 또는
어떤 부호로 읽을지만 바꾼다.

즉 동역학은 **고정된 payload의 가시성**에 작용한다.

## 복소 value

\(v_{n,b}=|v_{n,b}|e^{i\psi_{n,b}}\)이면 출력 채널 \(b\)의 기여는

\[
\operatorname{Re}
\left(c_{t,n}\overline{v_{n,b}}\right)
=|c_{t,n}||v_{n,b}|
\cos\left(\angle c_{t,n}-\psi_{n,b}\right)
\]

이다. value 채널마다 위상 \(\psi_{n,b}\)가 다르므로
\(\angle c_{t,n}\)가 시간에 따라 변할 때 각 출력 채널이 서로 다르게
변한다. 따라서 같은 write도 시점에 따라 다른 실수 벡터로 관측될 수
있다.

즉 동역학이 **검색 강도뿐 아니라 관측되는 payload의 방향과 내용**에도
작용한다.

## 의미 차이

- 실수 \(v\): 복소 위상은 주소·시간·간섭을 담당하고, \(v\)는 고정된
  내용이다.
- 복소 \(v\): 주소 선택과 payload 내용이 value 위상을 통해 결합된다.

두 경우 모두 \(S\)의 직접 전이 \(S\mapsto US\)는 동일하게 unitary이고,
write 하나의 Frobenius norm도 보존된다. 달라지는 핵심은 상태 전이의
노름 안정성이 아니라 **read에서 발생하는 관측 동역학과 메모리의
의미**다. 따라서 원래 식에서 복소 \(v\)는 단순한 표현력 확장이 아니라,
고정된 내용을 위상으로 선택하는 메모리에서 시간에 따라 관측 내용까지
변하는 메모리로의 구조적 변경이다.
