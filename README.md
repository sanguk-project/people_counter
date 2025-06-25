# People Counter using YOLOv5/YOLOv8

이 프로젝트는 YOLOv5/YOLOv8과 TensorRT를 활용한 실시간 통행량 계수 시스템입니다.

## 🔄 **[2024년 3월 업데이트]** 프로젝트 구조 리팩토링 완료

기존의 복잡하고 중복이 많았던 폴더 구조를 깔끔하게 정리했습니다.

### ✅ **완료된 정리 작업**
- **Legacy 폴더 통합**: `main/code/legacy/` 폴더의 모든 내용을 새로운 구조로 이동 후 삭제
- **코드 모듈화**: 공통 코드를 `common/` 디렉토리로 통합
- **Git 설정 최적화**: 모델 파일, 설정 파일, 결과 파일을 .gitignore에 추가

## 📁 **새로운 프로젝트 구조**

```
people_counter/
├── main/
│   ├── code/
│   │   ├── common/                    # 🔧 공통 모듈들
│   │   │   ├── __init__.py
│   │   │   ├── centroidtracker.py     # 객체 추적 알고리즘
│   │   │   ├── trackableobject.py     # 추적 가능한 객체 클래스
│   │   │   └── utils/                 # 유틸리티 모듈들
│   │   │       ├── __init__.py
│   │   │       ├── onnx2tensorrt.py   # ONNX → TensorRT 변환
│   │   │       ├── onnx2int8.py       # INT8 양자화 변환
│   │   │       └── onnx2int8_edge.py  # Edge용 INT8 변환
│   │   │
│   │   ├── models/                    # 🤖 모델 관련 파일들 (Git 제외)
│   │   │   ├── yolov4/               # YOLOv4 모델 파일들
│   │   │   ├── yolov5/               # YOLOv5 모델 파일들
│   │   │   └── yolov8/               # YOLOv8 모델 파일들
│   │   │
│   │   ├── apps/                     # 🚀 메인 애플리케이션들
│   │   │   ├── edge_ai/              # Edge AI 최적화 버전
│   │   │   ├── people_counter_yolov4/ # YOLOv4 기반 구현
│   │   │   ├── people_counter_yolov5/ # YOLOv5 기반 구현 (메인)
│   │   │   └── people_counter_yolov8/ # YOLOv8 기반 구현
│   │   │
│   │   ├── tests/                    # 🧪 테스트 파일들
│   │   │
│   │   ├── config/                   # ⚙️ 설정 파일들 (Git 제외)
│   │   │   ├── settings.yaml         # 메인 설정 파일
│   │   │   └── requirements.txt      # 의존성 패키지 목록
│   │   │
│   │   └── analysis/                 # 📈 분석 및 시각화
│   │       └── results_analysis.ipynb # 결과 분석 노트북
│   │
│   ├── results/                      # 실행 결과 저장
│   └── srcs/                        # 샘플 데이터
├── docs/                            # 기술 문서
└── README.md                        # 이 파일
```

## 🎯 **Git 관리 최적화**

다음 폴더들은 .gitignore에 추가되어 Git에 업로드되지 않습니다:
- `main/code/config/` - 설정 파일들
- `main/code/models/` - 대용량 모델 파일들
- `main/code/apps/*/results/` - 실행 결과 파일들
- `main/code/apps/*/models/` - 개별 앱의 모델 파일들

## 🚀 **주요 기능**

- **실시간 객체 탐지**: YOLOv5/YOLOv8 모델을 사용한 사람 탐지
- **TensorRT 최적화**: GPU 추론 성능 최적화 (FP32, FP16, INT8 지원)
- **통행량 계수**: Centroid Tracking 알고리즘을 사용한 양방향 통행량 계수
- **Edge AI 지원**: NVIDIA Jetson 플랫폼 최적화
- **모듈화된 구조**: 공통 코드 재사용 및 유지보수성 향상

## 🔧 **설치 및 설정**

### 환경 요구사항

- Python 3.8+
- CUDA 11.8+
- cuDNN 8.9.0+
- TensorRT 8.6.1+
- Numpy 1.23.4 (필수)
- OpenCV 4.7.0 
- torch 2.0.1+cu118
- tensorflow 2.10.0 (필수)
- ONNX 1.12 (필수)
- ONNXruntime 1.14.1
- ONNXruntime-gpu 1.14.1
- protobuf 3.19.6



## 📊 **개발 진행 과정**

1. **Yolov5 모델 개발** → **AI to ONNX format 변환** → **ONNX format to TensorRT engine으로 변환** → **Object Detection 확인** → **통행량 계수 기능 개발**

### 주요 단계별 설명

- **모델 개발**: 수집한 데이터셋을 활용하여 모델 개발(train-set 123,201장, valid-set 5,927장)
  - Batch-size, Epochs, Patience(Earlystopping), Activation Function, Learning-rate 등 하이퍼파라미터 조정
- **ONNX 변환**: yolov5에서 제공하는 export.py를 활용하여 AI 모델을 ONNX format으로 변환
- **TensorRT 최적화**: ONNX format을 TensorRT engine으로 변환하는 코드 작성(onnx2tensorrt.py)
- **객체 탐지**: TensorRT engine 기반 동영상에서 객체 탐지 확인
- **통행량 계수**: 기존 yolov4 기반 people counting 코드를 참고하여 통행량 계수 기능 추가

## 📊 **성능 지표 및 정밀도**

### 성능 테스트 기준

1. 평균 지연율(ms)
2. 초당 평균 프레임 처리량(fps)
3. 객체를 탐지하는 횟수
4. 카운트 된 객체수
5. 전체 카운팅 정확도(%)
6. 기준(ex. 좌우이동 등) 가운팅 정확도(%)

### 정밀도(Precision) 모드별 특성

#### 1) FP32 (32-bit floating point)
- IEEE 754 표준에 따라 32bit를 사용하여 실수 표현
- **높은 정확도, 느린 추론 속도**

#### 2) FP16 (16-bit floating point)
- 16bit를 사용하여 실수를 표현하는 데이터 형식
- **균형잡힌 정확도와 속도**

#### 3) INT8 (Quantization, 양자화)
- 8bit 정수 형식으로 -128 ~ 127까지의 범위를 표현
- **빠른 추론 속도, 양자화로 인한 약간의 정확도 손실**
- 전력 절약 및 메모리 효율성 향상

##### TensorRT의 양자화 방법
- **Post-Training Quantization(PTQ)**: 모델 학습이 완료된 후 모델을 양자화
- **Quantization Aware Training(QAT)**: 모델 학습 과정에서 양자화를 포함

## 🔄 **마이그레이션 가이드**

### 기존 코드 수정 시 체크리스트

- [ ] Import 경로 수정 (`from common.` 사용)
- [ ] 설정 파일 경로 수정 (`config/settings.yaml`)
- [ ] 모델 파일 경로 수정 (`models/` 하위)
- [ ] 출력 파일 경로 확인

### 권장 작업 순서

1. **설정 확인**: `config/settings.yaml` 파일 검토 및 수정
2. **공통 모듈 테스트**: `common/` 디렉토리 모듈들이 정상 작동하는지 확인
3. **단계별 실행**: 각 YOLO 버전별로 테스트
4. **결과 분석**: analysis 도구를 활용한 성능 검증

## ✨ **개선 사항**

### 1. 중복 제거
- ✅ `centroidtracker.py` → `common/` 디렉토리로 통합
- ✅ `trackableobject.py` → `common/` 디렉토리로 통합
- ✅ `onnx2tensorrt.py` → `common/utils/` 디렉토리로 통합

### 2. 명확한 분류
- ✅ **공통 모듈**: `common/` - 모든 버전에서 공통 사용
- ✅ **모델 파일**: `models/` - 각 YOLO 버전별 구분
- ✅ **메인 앱**: `apps/` - 실제 실행 가능한 애플리케이션
- ✅ **테스트 코드**: `tests/` - 테스트 및 실험용 코드
- ✅ **설정 파일**: `config/` - 모든 설정 파일 통합 관리

### 3. 유지보수성 향상
- ✅ 단일 책임 원칙 적용
- ✅ 의존성 관계 명확화
- ✅ 버전별 구분 체계화

## 🎯 **지원 플랫폼**

- **개발 환경**: NVIDIA GeForce RTX 30/40 시리즈 (테스트: RTX 3090 Ti)
- **Edge 환경**: NVIDIA Jetson Xavier NX, Jetson AGX Xavier
- **OS**: Ubuntu 18.04+, Windows 10+

## ⚠️ **사용 시 주의사항**

1. **모델 파일**: 본 저장소에는 모델 파일이 포함되지 않습니다. 직접 학습하거나 공개 모델을 사용하세요.
2. **설정 파일**: `config/` 폴더의 설정 파일들을 환경에 맞게 수정해야 합니다.
3. **데이터셋**: 저작권이 있는 데이터셋은 개별적으로 확보하세요.
4. **상업적 사용**: 상업적 용도로 사용 시 관련 라이선스를 확인하세요.
5. **정밀도 선택**: 작업 상황 및 어플리케이션의 필요에 따라 적절한 데이터 형식 선택 필요
