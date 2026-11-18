# 실험 조건과 실행 방법

## 입력 배율 실험

`results/kitti15-resize-sweep.csv`의 12개 행은 2026년 프로젝트 발표 «DEFOM Stereo» 3장에 기록된 측정표에서 가져왔다. 같은 발표의 640×640 단일 입력 실험(1장)은 이 CSV에 포함하지 않았다. KITTI15의 370×1224 표기와 깊이 인코더가 실제로 받는 크기는 다르다. 예를 들어 배율 2.0에서 깊이 인코더 입력은 196×630이다. 전체 모델 경로와 세부 모듈은 각각 따로 측정하므로 세부 시간의 합이 전체 시간과 달라질 수 있다.

| 항목 | 조건 |
| --- | --- |
| 비교한 인코더 | ViT-S, ViT-L |
| 깊이 인코더 입력 배율 | 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 |
| 측정 지표 | 전체·Depth Anything·ViT·DPT 지연시간, EPE, Out3 |
| 정밀도 | 혼합 정밀도 비활성 |

[전체 측정표](../results/kitti15-resize-sweep.csv) · [그래프 생성 코드](../scripts/plot_resize_sweep.py)

그래프를 다시 그리려면 Matplotlib를 설치한 Python 환경에서 저장소 루트 기준으로 실행한다.

```sh
python scripts/plot_resize_sweep.py
```

## GPU에서 평가 코드 실행

DEFOM-Stereo 모델 소스, 해당 버전의 CUDA·PyTorch 환경, 사용할 인코더에 맞는 체크포인트, KITTI 2015 학습 분할이 필요하다. [모델 저장소](https://github.com/antigravity-tech/DEFOM-Stereo)의 환경을 준비하고, [평가 스크립트](../evaluate_kitti_profile.py)를 해당 소스 트리의 `core/`와 같은 위치에 둔다. 사용한 KITTI 로더는 실행 디렉터리의 `../datasets/KITTI/training/` 아래 `image_2`, `image_3`, `disp_occ_0`를 읽는다. 공개 패키지에는 모델 가중치나 데이터셋을 동봉하지 않았다.

```sh
python evaluate_kitti_profile.py --help
python evaluate_kitti_profile.py --restore_ckpt ./checkpoints/model.pth --dinov2_encoder vits --kitti_split 15 --da_factor 2.0 --valid_iters 32 --scale_iters 8
```

두 번째 명령의 경로와 모델 구조 옵션은 실제 체크포인트에 맞춰야 한다. 위 값은 한 조건의 실행 예다. 기본 프로파일러는 앞 50개 장면 뒤부터 전체 지연시간을 집계하고, 그중 20개 장면에서 세부 모듈을 따로 측정한다. 저장한 CSV는 발표 당시의 결과이며, 초안을 준비하면서 GPU 전체 평가를 다시 실행하지 않았다.

## INT8 결과의 집계 기준

전체 200개 장면의 [CSV](../results/kitti-int8-per-image.csv)에서 `backend_calibration_seen=False`인 168개를 선택한다. 각 장면의 EPE는 유효 픽셀에 대한 시차 절대 오차의 평균이다. 표의 EPE는 이 장면별 값을 평균한 결과다. Out3은 절대 오차가 3 px를 초과하는 비율이고, D1은 3 px와 정답 시차의 5%를 모두 초과하는 비율이다. Out3과 D1의 장면별 비율은 유효 픽셀 수로 가중한다. 전단 보정과 후단 미세학습을 함께 적용한 비교이므로 구성 간 차이에는 두 변경의 효과가 모두 들어 있다.

## <a id="board"></a>VCU128 단일 블록 확인

[호스트 스크립트](../board/dino_block_check.py)를 실행하려면 해당 비트스트림을 올린 VCU128, PCIe 드라이버, 보드용 Python 모듈 `PCIeAccess`·`NPUv4`, 입력 및 참조 벡터가 필요하다. `DINO_VEC_DIR`에 입력·가중치·양자화 벡터 디렉터리, `DINO_GOLD_DIR`에 참조 출력 디렉터리, `DINO_VERIFY_ROOT`에 결과 덤프 경로를 지정한다. 실행하면 실제 장치에 데이터를 기록한다. 이 초안에서 CPU로 재실행한 하드웨어 관련 검증은 [correlation lookup 사례](../hardware/README.md)다.

## <a id="sources"></a>그림·모델 출처

- **대표 이미지** · KITTI Stereo 2015 학습 분할 장면 `000001_10`의 좌영상·정답 시차 위에 이 프로젝트의 FP32·INT8 예측, 차이 및 오차 지도를 함께 배치했다. 원 데이터는 Moritz Menze와 Andreas Geiger의 *Object Scene Flow for Autonomous Vehicles* (CVPR 2015)에 기반한다. KITTI 파생 비교 이미지에도 CC BY-NC-SA 3.0을 적용한다. [데이터셋](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) · [KITTI 저작권·인용 안내](https://www.cvlibs.net/datasets/kitti/) · [이미지 라이선스](https://creativecommons.org/licenses/by-nc-sa/3.0/)
- **기반 모델** · DEFOM-Stereo의 모델 구현과 원 논문은 원 연구팀의 작업이다. 이 프로젝트의 입력 배율 실험, 양자화 비교, FPGA 연산 검증은 그 위에서 수행했다. [DEFOM-Stereo 원본](https://github.com/antigravity-tech/DEFOM-Stereo) · [원본 트리의 MIT 고지](../LICENSE.upstream)
