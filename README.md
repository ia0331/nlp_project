# GLoRIA 기반 흉부 X선 판독문 생성

흉부 X선 한 장을 입력받아 **정상 · 폐렴 · 기타 이상**으로 분류하고, 영문 판독문의 **FINDINGS** 초안을 생성하는 프로젝트다. 폐렴으로 분류된 영상에서는 모델이 주목한 폐 영역을 계산한다. GLoRIA와 검색 인덱스를 연결하면 영상과 유사한 과거 판독 문장도 함께 제시한다.

이 프로젝트에는 역할이 다른 두 모델이 있다. **GLoRIA**는 영상과 판독문을 같은 임베딩 공간에 정렬하도록 학습한다. **3분류 ResNet-50**은 이전 프로젝트에서 학습한 가중치를 불러와 영상의 클래스를 예측한다. [3분류 모델의 이전 프로젝트](https://github.com/lgh0829/AWS-SAY-1-Team4)

## 구현한 기능

| 단계 | 구현 내용 | 결과 |
| --- | --- | --- |
| 데이터 준비 | 정상·폐렴 CSV를 병합하고 영상 경로를 통일한다. `label` 비율을 유지하며 행 단위로 데이터를 나눈다. | 학습·검증·테스트 CSV (기본 70·15·15%) |
| 영상–텍스트 표현 학습 | ResNet-50과 BioClinicalBERT의 가중치를 고정하고 네 개의 투영 헤드를 학습한다. 영상 전체와 판독문 전체의 유사도, 영상 패치와 단어 토큰의 어텐션을 손실 계산에 사용한다. | GLoRIA 체크포인트, 테스트 배치 내 양방향 검색 Recall@1·5 |
| 분류 및 위치 추정 | 별도 ResNet-50으로 3분류한다. 폐렴이면 Grad-CAM을 폐 마스크와 겹쳐 네 분면의 점수와 활성 집중도를 계산한다. | 클래스, 확률, 가장 활성도가 높은 폐 분면 |
| 판독 문장 검색 | 학습 판독문을 문장 단위로 임베딩해 ChromaDB에 저장한다. 폐렴 영상의 전역 임베딩으로 유사 문장을 찾고, 검색 문장의 소견 키워드와 영상 패치 사이의 어텐션으로 위치를 추정한다. | 검색 문장과 키워드별 위치가 담긴 추론 JSON |
| 판독문 생성 및 평가 | 정상·기타 이상은 정해진 문구를 사용한다. 폐렴은 JSON에서 문장을 조합하거나 OpenAI 또는 Amazon Bedrock으로 초안을 생성한다. 생성문과 참조문을 파일명으로 짝지어 SBERT와 PubMedBERT의 코사인 유사도를 계산한다. | FINDINGS 텍스트, 영상별 유사도 CSV, 분포 히스토그램 |

## 폴더 구조

```text
nlp_project/
├── prepare_cxr_dataset.ipynb          # 판독문·영상 매칭, 라벨 정리 실험 기록
├── configs/
│   ├── gloria_training.json          # 표현 학습 설정 및 체크포인트 저장 경로
│   └── binary_classifier_training.json  # 별도 이진 분류 실험 설정
├── scripts/
│   ├── merge_cxr_manifests.py        # 정상·폐렴 CSV 병합 및 영상 경로 정리
│   ├── build_splits.py               # 라벨 비율을 유지한 데이터 분할
│   ├── train_representation.py        # GLoRIA 학습 및 검색 평가
│   ├── infer_cxr.py                  # 단일 영상 추론 · 배치 실행의 기본값
│   ├── infer_cxr_baseline.py         # 집중도 지표 추가 전의 추론 구현
│   ├── generate_report.py            # 추론 JSON → 영문 FINDINGS
│   ├── batch_infer_and_report.py     # CSV의 영상별 추론 및 판독문 생성
│   └── evaluate_report_similarity.py # 생성문과 참조문의 유사도 평가
├── src/
│   ├── datasets/                     # 영상·판독문 쌍을 읽는 PyTorch Dataset
│   ├── models/                       # 두 인코더, 투영 헤드, 패치–토큰 어텐션
│   ├── losses/                       # 전역·지역 대조 손실
│   ├── rag/                          # index.py로 문장 색인, retrieve.py로 텍스트 질의
│   └── utils/                        # 소견 텍스트 분석 및 폐 분면 마스크 유틸리티
├── requirements.txt
└── README.md
```

## 실행 준비

아래 명령의 기준 디렉터리는 `nlp_project/`다. 영상, 원본 판독문 CSV, 모델 가중치, ChromaDB 인덱스는 별도로 준비한다. 노트북에는 실험 당시의 로컬 절대 경로가 있어 셀을 재사용할 때 데이터 위치를 수정해야 한다.

```shell
python -m pip install -r requirements.txt
python -m pip install chromadb opencv-python
```

유사도 평가에는 `sentence-transformers`와 `matplotlib`이 추가로 필요하다. 외부 LLM을 사용하는 경우 선택한 제공자에 맞춰 `openai`와 API 키 또는 `boto3`와 Bedrock 접근 자격을 준비한다. 기본값인 `--provider none`은 로컬에서 판독문을 조합한다.

### 1. 단일 영상에서 판독문까지

`checkpoints/model.pth`에는 3분류 ResNet-50의 `state_dict`가 필요하다. 폐 분면 계산은 배경이 검은색인 폐 영역 분할 영상을 전제로 한다. 아래 예시는 검색 인덱스 없이 실행하는 경로다.

```shell
python -m scripts.infer_cxr --image data/imdat/example.jpg --model checkpoints/model.pth --out_json outputs/example.json
python -m scripts.generate_report --in_json outputs/example.json --out_report outputs/example.txt --provider none
```

`outputs/example.json`의 `prediction`과 `probs`에서 분류 결과를 확인할 수 있다. 폐렴으로 예측한 경우 `gen_payload.findings.global`에 분면 점수와 집중도가 기록된다. `outputs/example.txt`에는 영문 FINDINGS 초안이 저장된다.

### 2. GLoRIA 학습 및 문장 검색 연결

학습과 분할에 사용할 CSV에는 `file_name`(실제 영상 경로), `report`(영문 판독문), `label`(분할 기준)이 필요하다. 정상·폐렴 CSV가 따로 있다면 첫 명령으로 병합한다. 이미 같은 형식의 CSV가 있다면 분할부터 시작한다.

```shell
python -m scripts.merge_cxr_manifests --normal data/final_normal.csv --pneumo data/final_pneumo.csv --prefix data/imdat --out data/merged.csv
python -m scripts.build_splits --manifest data/merged.csv --out_dir data
python -m scripts.train_representation --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv --cfg configs/gloria_training.json
python -m src.rag.index --csv data/train.csv --ckpt checkpoints/rep_best.pt --persist_dir outputs/chroma --collection pneumo_reports
```

학습 스크립트는 모델 가중치를 `rep_best.pt`의 `model` 키에 저장한다. 색인 스크립트는 이 파일을 그대로 읽는다. 영상 추론 로더는 가중치 딕셔너리를 직접 받으므로, 아래처럼 한 번 추출한 뒤 같은 컬렉션을 지정한다.

```shell
python -c "import torch; ckpt = torch.load('checkpoints/rep_best.pt', map_location='cpu'); torch.save(ckpt['model'], 'checkpoints/gloria_weights.pt')"
python -m scripts.infer_cxr --image data/imdat/example.jpg --model checkpoints/model.pth --gloria_ckpt checkpoints/gloria_weights.pt --chroma_path outputs/chroma --chroma_collection pneumo_reports --out_json outputs/example.json
```

같은 영상의 판독문을 다시 생성하면 검색 문장을 반영할 수 있다. 추론 JSON의 `support_sentences`와 `localized_findings`에서 어떤 문장과 위치가 사용됐는지 확인할 수 있다.

### 3. 배치 생성 및 참조 판독문과 비교

배치 실행기는 `file_name` 열의 영상을 하나씩 처리해 결과 CSV에 `report`와 `label`을 기록한다. 예측 라벨은 정상 `0`, 폐렴 `1`, 기타 이상 `2`다. 아래 PowerShell 예시는 자식 프로세스에서도 프로젝트의 `src`를 찾을 수 있도록 경로를 지정한다.

```powershell
$env:PYTHONPATH = (Get-Location).Path + [IO.Path]::PathSeparator + $env:PYTHONPATH
python -m scripts.batch_infer_and_report --csv data/test.csv --model checkpoints/model.pth --out_csv outputs/generated.csv --provider none
python -m scripts.evaluate_report_similarity --gen_csv outputs/generated.csv --ref_csv data/test.csv
```

기본 출력은 영상별 결과를 담은 `outputs/report_similarity.csv`와 모델별 분포 그림 `outputs/similarity_hist.png`, `outputs/similarity_pub.png`다. 이 점수는 같은 영상에 대한 두 판독문의 표현이 임베딩 공간에서 얼마나 가까운지 나타낸다.
