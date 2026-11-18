# 한 픽셀의 correlation lookup 검증

DEFOM의 첫 번째 delta update에서 좌영상 픽셀 `(y=0, x=41)`의 correlation 값을 조회한 사례다. 두 피라미드 레벨에서 각각 아홉 위치를 선형 보간한 뒤 INT8로 재양자화한다. 시차 입력은 Q8 표현으로 235다. 유효 채널 18개와 정렬용 0바이트 46개가 합쳐져 128비트 워드 네 개가 된다.

## CPU에서 실행

저장소 루트에서 Python 3으로 실행한다. Python 표준 라이브러리만 사용한다.

```sh
python hardware/replay_corr.py
```

[재생 코드](replay_corr.py)는 원래 검증 코드 `native_corr.py`의 `lookup_integer`에서 이 픽셀에 해당하는 정수 연산을 옮겼다. 두 correlation 행으로 결과를 새로 계산하고, 각 바이트를 당시의 Python 참조값 및 XSim이 출력한 네 워드와 대조한다.

```text
PASS: 18 INT8 correlation channels and 46 padding bytes match the reference and the recorded four XSim beats.
Channel 9: RTL arithmetic = 21, frozen model = 20.
```

두 번째 레벨은 좌표를 절반으로 줄인다. 이 사례의 Q8 좌표 10261은 당시 RTL의 양수 반올림 규칙으로 5131, 이전 모델의 내림 규칙으로 5130이 된다. 이 차이는 채널 9에서 INT8 값 21과 20의 차이로 나타난다.

## 입력과 기록의 출처

2026년 10월 27일의 단일 픽셀 RTL 검증 패키지 `rtl_corr_pixel`에서 다음 데이터를 가져왔다.

| 파일 | 역할 |
| --- | --- |
| [레벨 0의 입력](fixtures/row0.hex) · [레벨 1의 입력](fixtures/row1.hex) | 이 픽셀의 correlation 행. 각 줄은 signed INT32 값 네 개를 담은 128비트 워드다. |
| [RTL 연산 규칙의 참조값](fixtures/expected.hex) · [이전 모델의 참조값](fixtures/frozen.hex) | 같은 픽셀의 채널별 출력을 비교하는 원본 바이트열이다. |
| [XSim 출력](fixtures/xsim_output.hex) | 당시 로그의 `OUTPUT_BEAT_PASS` 네 줄에서 추출한 128비트 출력이다. |
| [SystemVerilog 테스트벤치](tb_corr_lookup_pixel.sv) | correlation 블록의 포트로 입력을 구동하고, 모의 DDR 응답과 핸드셰이크·출력값을 확인한 당시 벤치다. |

당시 Vivado XSim 실행은 `corr_lookup`의 실제 FSM, 연산부와 RAM을 사용했고 DDR에는 트랜잭션 응답 모델을 연결했다. DDR 요청 2회와 반환 6워드, 출력 4워드가 기록됐다. 해당 RTL의 원본 SHA-256은 `bcfb28e4577932ed2fc3bd9d12a60f19a7ba56f15c4d638d8f3f9fcc05486ec1`이며, 시뮬레이션용 사본은 `interp_cnt` 선언 위치만 조정했다.

이 예제의 실행 범위는 CPU 정수 연산과 기록 재생이다. SystemVerilog 벤치를 다시 실행하려면 당시 `corr_lookup`와 `SimpleDP_Block_RAM` 소스를 별도로 준비하고 `fixtures` 파일을 시뮬레이터의 작업 디렉터리에 둔다. 전체 NPU, 실제 DDR 및 스테레오 시스템의 보드 처리량은 이 사례의 측정 범위 밖이다.
