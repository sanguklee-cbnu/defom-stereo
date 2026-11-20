# DINO ViT 가속기 · DEFOM 401토큰 RTL

이상욱이 설계한 `DINO_vcu128_norm3`를 DEFOM의 401토큰 입력에 맞춘 Verilog 구현이다. 한 영상의 패치 토큰 400개와 CLS 토큰 1개가 인코더 12개 블록을 통과한다. DEFOM은 블록 3·6·9·12의 Norm3 출력을 읽어 DPT에 전달한다. Vivado 블록 디자인에 연결하는 최상위 모듈은 `DEFOM_USER_VIT_TOP`이며, 내부의 `vit_axi_top`이 AXI-Lite 제어와 256비트 DDR 인터페이스를 제공한다. AXI master 모듈은 Sangbo Park가 설계했다. [최상위 모듈](rtl/DEFOM_USER_VIT_TOP.v#L204) · [AXI master 원저자 표기](rtl/DEFOM_USER_VIT_AXI_Master.v#L2)

## FC1 결과를 칩 안에서 FC2에 전달하는 방법

MLP는 같은 타일의 FC1과 FC2를 차례로 실행한다. FC1은 양자화와 GELU를 거친 값을 온칩 `SCOREBUF`의 bank 0에 쓰고, FC2는 같은 주소에서 그 타일을 읽는다. 이 데이터 경로는 확장된 FC1 활성값 전체를 외부 DRAM에 쓴 뒤 FC2에서 다시 읽는 전송을 생략한다. [FC1 → FC2 실행 순서](rtl/vit_controller.v#L1008) · [FC1 타일의 쓰기 주소와 데이터](rtl/writeback_controller.v#L2116) · [FC2 입력의 SCOREBUF 연결](rtl/vit.v#L263)

## 실보드와 RTL 검증

- **실보드** · 2026년 10월 29일 14:03 실행에서는 ViT 한 개로 좌우 영상을 차례로 처리하고 DPT와 후단을 NPU 4개에서 실행했다. 같은 영상 한 쌍의 전체 추론 시간은 **2,809.116 / 2,847.427 / 2,823.743 ms**였고 세 회차 모두 사후 수치 검사를 통과했다. 이 시간에는 호스트의 첫 상관 볼륨 계산과 최종 시차 업샘플링이 포함된다. [보드 실행 기록과 측정 범위](../../docs/fpga-board-runs.md)
- **원본 릴리스의 XSim 검증** · 고정된 좌우 입력 각각에 대해 AXI 제어와 DDR 모델을 포함한 RTL로 12개 블록을 실행했다. 패치 단계의 Add0·Norm1, 각 블록의 Add2, 네 개의 Norm3에서 비교한 유효 INT8 데이터 **5,543,424바이트**가 모두 참조 출력과 일치했다.

공개본의 30개 RTL·헤더·ROM 파일은 원본 릴리스와 SHA-256이 일치한다. Vivado 2022.2에서 등록 스크립트를 시험하고, XSim으로 `DEFOM_USER_VIT_TOP`을 컴파일·엘라보레이션했다. 원본과 같은 미연결 경고가 하나 발생하는 `controller_top.m_axi_wlast`는 상수 출력이며, 외부 DDR의 WLAST는 AXI master가 구동한다. [외부 WLAST 연결](rtl/vit.v#L658)

## Vivado 프로젝트에 추가

Vivado에서 대상 프로젝트를 연 뒤, Tcl Console에서 저장소 루트를 현재 디렉터리로 두고 실행한다.

```tcl
source hardware/dino-vit/add_sources.tcl
```

스크립트는 `rtl/`의 Verilog 소스와 헤더, `mem/`의 ROM 초기화 파일 15개를 등록하고 헤더 검색 경로를 지정한다. 기존 프로젝트의 최상위 모듈 설정은 유지된다. `mem/`에는 Softmax·GELU·LayerNorm 등의 고정 테이블이 들어 있다. 실제 보드 추론에는 호스트가 DDR에 적재하는 모델 가중치(`weight256.bin`), 파라미터(`parameter256.bin`), 양자화 계수(`quant256.bin`), 좌우 입력이 별도로 필요하다. [Vivado 등록 스크립트](add_sources.tcl) · [ROM 초기화 파일](mem/) · [DDR 주소 배치](rtl/addr_map.vh#L78)
