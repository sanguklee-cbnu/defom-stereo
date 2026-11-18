# DEFOM-Stereo · 시차 추정의 GPU 분석과 FPGA 검증

**2026 · 이상욱**  
좌우 카메라 영상에서 물체까지의 거리를 추정하려면 같은 지점이 두 영상에서 얼마나 어긋나는지, 즉 시차를 구해야 한다. 이 프로젝트에서는 DEFOM-Stereo의 깊이 인코더 입력을 줄였을 때의 속도와 오차를 측정하고, FPGA 이식을 위해 INT8 추론 결과와 하드웨어 연산을 검증했다.

![KITTI 장면 000001_10의 왼쪽 영상, 정답 시차, FP32와 INT8 예측 및 오차 지도](assets/kitti-000001-10.jpg)

**그림을 읽는 법** · 첫 줄은 왼쪽 영상, KITTI 정답, FP32 예측이다. 둘째 줄에서는 INT8 조정 전·후 결과를, 마지막 줄에서는 정답과의 절대 오차를 비교한다. 오차 지도에서 회색은 유효한 정답이 없는 픽셀이다. 이 장면의 평균 시차 오차는 INT8 조정 전 **3.358 px**, 조정 후 **0.797 px**였다. 예측은 하드웨어 산술을 소프트웨어에서 모사해 얻었다. [그림 출처와 사용 조건](docs/experiments.md#sources)

[실험 요약 PDF](docs/defom-project-brief.pdf) · [실행 환경과 측정 조건](docs/experiments.md)

## 깊이 인코더 입력을 어디까지 줄일 수 있을까?

KITTI 2015 영상에서 Depth Anything의 입력 배율을 1.0–3.5로 바꿔 ViT-S와 ViT-L을 각각 측정했다. **ViT-S**에서는 배율을 3.5에서 2.0으로 내리자 전체 추론 시간이 **253.4 → 234.8 ms**, 평균 시차 오차(EPE)는 **1.077 → 1.110 px**가 됐다. 배율을 1.0까지 더 낮춰도 전체 시간은 235.0 ms였고 오차는 1.191 px로 커졌다. 같은 변경에 대한 **ViT-L**의 전체 시간은 441.5 → 287.9 ms였다.

![인코더 입력 배율에 따른 ViT-S와 ViT-L의 전체 지연시간 및 시차 오차](assets/kitti15-resize-sweep.png)

발표의 입력 크기 표기는 370×1224이며 혼합 정밀도는 껐다. 그래프의 시간은 전체 모델을 한 번 실행한 값들의 평균이다. [12개 조건의 측정값](results/kitti15-resize-sweep.csv) · [GPU 프로파일링 코드](evaluate_kitti_profile.py)

## INT8로 바꾼 뒤의 시차 오차

전단의 양자화 파라미터를 다시 보정하고 후단을 미세학습했다. KITTI 2015 학습 분할의 200쌍 중 이번 보정·미세학습에 사용한 32쌍을 제외하고, 동일한 **168쌍**에서 세 구성을 비교했다. 이전 전단 최적화에서도 KITTI를 사용했으므로 이 수치의 범위는 해당 장면들이다.

| 소프트웨어 예측 | 평균 시차 오차 EPE (px) ↓ | Out3 (%) ↓ | D1 (%) ↓ |
| --- | ---: | ---: | ---: |
| 기준 DEFOM FP32 | 1.3233 | 7.4102 | 7.2303 |
| 조정 전 INT8 | 3.8179 | 28.0948 | 28.0233 |
| 조정 후 INT8 | **1.2104** | **4.8217** | **4.6038** |

EPE는 장면별 평균에 같은 가중치를 줬다. Out3와 D1은 유효한 정답 픽셀 15,358,905개를 기준으로 계산했다. [장면별 결과](results/kitti-int8-per-image.csv) · [집계 코드](scripts/summarize_int8.py)

## FPGA에서 확인한 연산 범위

- **Correlation lookup** · 첫 시차 갱신의 한 픽셀에서 두 피라미드 레벨의 데이터를 보간하고 INT8로 다시 양자화했다. 유효 채널 18개와 패딩 46바이트를 다시 계산하면 이전 Vivado XSim이 기록한 128비트 출력 네 개와 일치한다. 채널 9에서 이전 참조값 20과 RTL 산술 결과 21이 갈린 지점도 확인할 수 있다. [재현 코드·테스트벤치·입력 벡터](hardware/README.md)
- **DINO 인코더** · VCU128용 호스트 코드는 PCIe로 패치 임베딩과 첫 인코더 블록의 Q·V·최종 출력을 읽어 참조값과 비교한다. 이 코드는 한 블록의 검증 범위를 보여준다. 전체 DEFOM 파이프라인의 실보드 처리량은 별도 측정 항목이다. [보드 검증 코드](board/dino_block_check.py) · [보드 실행 조건](docs/experiments.md#board)

## 바로 확인하기

저장소 루트에서 아래 두 명령을 실행하면 Python 표준 라이브러리만으로 위의 INT8 표와 한 픽셀의 correlation 결과를 재현할 수 있다.

```sh
python scripts/summarize_int8.py
python hardware/replay_corr.py
```

기반 모델은 CVPR 2025의 *DEFOM-Stereo: Depth Foundation Model Based Stereo Matching*이다. [원 논문](https://arxiv.org/abs/2501.09466) · [원 모델 구현](https://github.com/antigravity-tech/DEFOM-Stereo) · [KITTI Stereo 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
