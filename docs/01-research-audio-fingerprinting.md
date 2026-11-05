# 오디오 핑거프린트 기술 리서치

> 작성일: 2026-11-05
> 목적: 휴대폰 실시간 녹음, 3초 이내 인식, 10만~100만 곡 서버 검색이 가능한 핑거프린트("DNA") 구조를 설계하기 위한 기술 조사

---

## 0. 요약 (TL;DR)

| 항목 | 결론 |
|---|---|
| **검증된 기본 기술** | Shazam 방식(스펙트럼 피크 → 피크 쌍 해시 → 역색인 → 시간 오프셋 투표). 구현이 단순하고 CPU만으로 100만 곡까지 확장 가능 |
| **최신 기술 흐름** | 딥러닝 기반 핑거프린트(NAFP 계열): 0.5~1초 구간마다 128차원 임베딩을 만들고 대조학습(contrastive learning)으로 잡음·잔향에 강하게 학습. ANN(FAISS IVF-PQ, HNSW) 검색 |
| **3초 인식** | 두 방식 모두 가능. 조용한 환경은 피크 해시로 충분하고, 시끄러운 환경(카페, 차 안)은 신경망 방식이 확실히 더 강함 |
| **100만 곡 인덱스** | 피크 해시: 수십 GB 역색인(샤딩). 신경망: 약 4억 개 벡터, PQ 압축 시 약 20~30 GB |
| **권장 방향** | **1단계** 피크 해시 방식 기준선 구현 → **2단계** 신경망 임베딩 추가 → **최종** 하이브리드(빠른 해시 후보 + 임베딩 재검증) |

---

## 1. 요구사항 정리

1. **입력**: 휴대폰 마이크로 실시간 녹음 (주변 소음, 잔향, 마이크 주파수 특성, AGC·노이즈 억제 등 OS 처리 포함)
2. **지연**: 녹음 시작 후 **3초 분량의 오디오로 인식**. 스트리밍으로 1초 단위로 보내고, 확신도가 기준을 넘으면 바로 종료(early stop)하는 구조가 유리
3. **규모**: 서버 DB 10만 곡 → 100만 곡. 검색 지연 수십~수백 ms, 메모리·디스크 비용 관리 가능해야 함
4. **DNA 구조**: 곡을 표현하는 압축된 데이터 형식. 조건:
   - 짧은 구간만으로 검색 가능 (곡의 어디서든 시작 가능 = 국소성)
   - 왜곡에 강건함 (잡음, 압축 코덱, EQ, 잔향)
   - 색인 가능 (해시 테이블 또는 벡터 인덱스)
   - 작은 크기 (클라이언트→서버 전송량, 서버 저장량)

---

## 2. 기술 분류

| 계열 | 대표 기술 | DNA 형태 | 검색 방식 |
|---|---|---|---|
| **A. 스펙트럼 피크 랜드마크** | Shazam(Wang 2003), Audfprint, Dejavu | 32비트 해시 + 시간 | 역색인 + 시간 오프셋 히스토그램 |
| **B. 비트 패턴(서브 핑거프린트)** | Philips(Haitsma & Kalker 2002) | 프레임마다 32비트 | 해시 정확 일치 + 비트 오류율(BER) 비교 |
| **C. 시간·피치 변형 불변 피크 조합** | Panako, QFP(Quad), Olaf | 피크 3~4개 조합의 불변량 해시 | 역색인 + 변형 파라미터 검증 |
| **D. 크로마/전곡 식별** | Chromaprint(AcoustID) | 크로마 기반 비트열 | 전곡 비교 (짧은 녹음엔 부적합) |
| **E. 신경망 임베딩** | Google Now Playing, NAFP, GraFPrint, PeakNetFP, Conformer FP | 구간당 실수 벡터(64~128차원) | ANN(FAISS 등) + 시퀀스 정렬 |

---

## 3. Shazam 방식 (Landmark / Constellation Map)

Avery Wang, *"An Industrial-Strength Audio Search Algorithm"*, ISMIR 2003.

### 3.1 처리 과정

```
오디오(8~11 kHz 모노) → STFT 스펙트로그램 → 국소 최대값(피크) 추출 → "별자리 지도"
   → 앵커 피크마다 앞쪽 목표 영역(target zone)의 피크 N개와 쌍을 만듦
   → 해시 = (f_anchor, f_target, Δt)   + 부가정보 t_anchor
   → 서버: hash → [(song_id, t_anchor), ...] 역색인 조회
   → 곡별로 (t_db − t_query) 차이 히스토그램 → 같은 오프셋에 몰린 개수 = 점수
```

1. **스펙트로그램**: 보통 8 kHz 또는 11,025 Hz로 다운샘플, 창 크기 약 64~128 ms, 홉 약 10~32 ms.
2. **피크 추출**: 시간·주파수 이웃 영역 안에서 가장 센 점만 남김. 초당 피크 수(밀도)를 조절하는 것이 핵심 튜닝 포인트. 피크는 잡음이 섞여도 살아남을 확률이 높고, 잡음은 원래 피크를 "가릴" 뿐 새 패턴을 맞추기 어렵다.
3. **조합 해싱(combinatorial hashing)**: 피크 하나는 정보량이 적어 충돌이 많다. 피크 2개 쌍으로 묶으면 해시 공간이 커져 특이성이 크게 높아지고, 쌍 중 하나만 살아도 다른 쌍이 남는다(fan-out).
4. **시간 오프셋 투표**: 진짜 곡이면 매칭된 해시들이 같은 시간 차이에 몰린다. 우연한 충돌은 흩어진다. 이 "정렬된 매칭 수"가 점수이며, 히스토그램 최고점이 배경 대비 충분히 높으면 인식 성공.

### 3.2 해시 비트 구성 예시 (32비트)

| 필드 | 비트 | 설명 |
|---|---|---|
| f_anchor | 9~10 | 주파수 빈 인덱스 |
| Δf (또는 f_target) | 8~10 | 목표 피크 주파수 차이 |
| Δt | 6~7 | 프레임 단위 시간 차이 |
| (여유) | 나머지 | 버전/대역 정보 등 |

역색인 엔트리(posting): `song_id(32비트) + t_anchor(16~32비트)` → 엔트리당 6~8바이트.

### 3.3 장단점

- **장점**: 단순하고 빠름, GPU 불필요, 잡음 섞임(additive noise)에 강함, 수 초 녹음으로 인식, 수천만 곡 상용 운영 검증됨, 오픈소스 구현 다수.
- **단점**:
  - **템포·피치 변화에 약함**: Δt, f가 그대로 해시에 들어가므로 1~2%만 속도가 바뀌어도 매칭이 무너짐 (라디오·DJ 믹스·라이브에서 문제).
  - **잔향·강한 소음**에서는 피크 자체가 바뀌어 성능 하락.
  - 인덱스가 큼 (아래 7장 계산 참고).
- **특허**: 원 특허(Wang/Shazam, 2000년대 초 출원)는 대부분 존속기간이 끝난 것으로 알려져 있지만, **상용화 전 법률 검토 필요**.

---

## 4. 기타 고전(비학습) 방식

### 4.1 Philips 방식 (Haitsma & Kalker, ISMIR 2002)
- 300~2000 Hz를 33개 로그 대역으로 나누고, 인접 대역·인접 프레임 에너지 차이의 부호로 **프레임당 32비트 서브 핑거프린트** 생성 (약 11.6 ms마다).
- 256개 서브 핑거프린트(약 3초)를 한 블록으로, **비트 오류율(BER) < 0.35** 이면 일치.
- 검색: 쿼리 블록 안에 **원본과 정확히 같은 32비트 값이 최소 하나는 있다**는 가정으로 해시 조회 후 BER 검증. 잡음이 심하면 이 가정이 깨져 비트 반전 후보를 추가 탐색.
- 의미: "3초 블록" 단위 설계의 고전적 근거. 휴대폰 녹음 같은 강한 왜곡에는 피크 방식보다 약한 편.

### 4.2 Panako (Joren Six) / QFP (Sonnleitner & Widmer)
- **Panako**: 피크 **3개(triplet)** 의 시간 비율·주파수 비율을 해시 → 시간 늘림(time-stretch)과 피치 변화에 일정 범위 불변. Panako 2.0(2021~)은 유지보수 중인 Java 오픈소스.
- **QFP (Quad-based Fingerprinting, IEEE/ACM TASLP 2016)**: 피크 **4개(quad)** 의 기하 관계를 정규화해 해시 → 속도·피치 ±수십%에 강함, 큰 DB에서도 검증.
- **Olaf (Overly Lightweight Acoustic Fingerprinting, Six 2020)**: C로 작성, ESP32 같은 임베디드·WebAssembly에서 동작. 클라이언트 쪽 경량 구현 참고용.
- 우리 용도: 휴대폰 녹음 자체에선 템포 변화가 없지만, **라디오(속도 조절), 리믹스, 라이브** 대응이 필요하면 고려.

### 4.3 Chromaprint / AcoustID
- 크로마(12음) 특징의 변화를 필터로 비트열화. **완성된 파일 전체**를 식별하는 용도(메타데이터 정리)에 최적화.
- 마이크 녹음 짧은 구간 인식에는 부적합 → 참고만.

### 4.4 오픈소스 구현
| 이름 | 언어 | 방식 | 비고 |
|---|---|---|---|
| Dejavu | Python | 랜드마크 | 학습용, MySQL/Postgres 저장, 대규모엔 느림 |
| audfprint (Dan Ellis) | Python | 랜드마크 | 깔끔한 레퍼런스, 해시 테이블 구조 참고 |
| Panako | Java | triplet | 시간/피치 변형 대응 |
| Olaf | C | 랜드마크 경량 | 임베디드·WASM |
| Chromaprint | C++ | 크로마 | 전곡 식별 |

---

## 5. 최신 기술: 신경망 오디오 핑거프린트

### 5.1 흐름 한눈에 보기

```
2017  Google "Now Playing" — 휴대폰 상시 저전력 인식, 기기 내 신경망 임베딩
2018  Google Sound Search — 같은 방식을 서버 대규모로 확장
2021  NAFP (Chang et al., ICASSP) — 대조학습 + FAISS, 공개 코드/데이터 → 사실상 표준 기준선
2022  Attention 기반 임베딩 (Singh et al.)
2023  Deezer: 피크 방식 + 음악 증강/잡음 제거 신경망
2024  GraFPrint (GNN), Holographic Reduced Representation(저장 개수 축소)
2025  PeakNetFP (Deezer, 피크 + 포인트넷), Conformer 사전학습 FP, 실환경 평가 프로토콜,
      합성 핑거프린트로 대규모 평가
2026  "Segment Length Matters" — 짧은 쿼리엔 짧은 세그먼트(0.5초)가 유리
```

### 5.2 Google Now Playing / Sound Search (2017~2018)
- Gfeller et al., *"Now Playing: Continuous low-power music recognition"* (NeurIPS 2017 workshop).
- 휴대폰에서 상시로, 서버 없이 기기 내에서 동작. 짧은 창마다 신경망이 **저차원 임베딩 시퀀스**를 출력하고, 기기 내 DB(수만 곡)에서 최근접 이웃 검색 후 시퀀스 일치로 확정.
- Sound Search는 같은 아이디어를 더 큰 모델·서버 DB로 확장.
- 의미: **"임베딩 시퀀스 = DNA"** 개념을 실서비스에서 처음 증명. 휴대폰 녹음 환경에서 신경망 방식이 실용적임을 보여줌.

### 5.3 NAFP — Neural Audio Fingerprint (Chang et al., ICASSP 2021) ★ 핵심 기준선
- 논문: *"Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrastive Learning"*, arXiv:2010.11910. 공개 코드: `mimbres/neural-audio-fp`.
- **구조**
  - 입력: 8 kHz, **1초 세그먼트**, **0.5초 홉**, 로그 멜 스펙트로그램(256×32 수준)
  - 인코더: 분리형 합성곱(separable conv) CNN → 투영층 → **128차원 L2 정규화 벡터**
  - 학습: **대조학습(SimCLR/NT-Xent)**. 원본 세그먼트와 "증강된" 같은 세그먼트(배경 잡음, 실내 잔향 IR, 마이크 IR, 시간 어긋남)를 가깝게, 다른 곡은 멀게
  - 검색: FAISS(IVF-PQ 등)로 세그먼트별 후보 → 쿼리의 연속 세그먼트들이 **같은 곡의 연속 위치**를 가리키는지 시퀀스 점수
- **데이터**: FMA 기반 약 10만 곡 규모 DB에서 평가. 1~3초 쿼리에서 Now Playing 재현 모델 대비 큰 폭으로 향상, PQ 압축 후에도 성능 유지.
- **의미**: 우리 요구(휴대폰, 3초, 10만~100만 곡)와 정확히 같은 문제를 풀었고, 코드·데이터가 공개돼 있어 **2단계 출발점으로 가장 적합**.

### 5.4 그 밖의 최신 연구

| 연구 | 연도 | 핵심 아이디어 | 우리에게 주는 시사점 |
|---|---|---|---|
| Attention-based embeddings for Query-by-Example (Singh et al., arXiv:2210.08624) | 2022 | 어텐션 인코더로 잡음·잔향 강건성 개선 | 기존 시스템이 5초 이상 쿼리가 필요하다는 문제 제기 |
| Music augmentation & denoising for peak-based FP (Deezer, 2023) | 2023 | 피크 추출 전에 신경망으로 잡음 제거 | **기존 피크 해시 인덱스를 그대로 쓰면서** 성능 향상 가능 |
| Holographic Reduced Representations (arXiv:2406.13139) | 2024 | 여러 세그먼트 임베딩을 하나로 합성해 **저장 벡터 수 감소** | 100만 곡 인덱스 크기 절감 |
| GraFPrint (arXiv:2410.10994) | 2024 | 시간-주파수 점들로 k-NN 그래프 → GNN 인코딩 | "별자리 지도"를 학습으로 해석. 코드 공개(`chymaera96/GraFP`) |
| PeakNetFP (Deezer, arXiv:2506.21086) | 2025 | **스펙트럼 피크 점 집합**을 PointNet++ 계열로 임베딩, 대조학습 | NAFP 대비 파라미터 약 1/100, 입력 약 1/11, 50~200% 시간 늘림에서도 Top-1 90% 이상. **휴대폰 경량 추출**에 유리. 코드 공개 |
| Contrastive & Transfer Learning + 실환경 평가 (arXiv:2507.06070) | 2025 | 10만 곡 이상, **5,600만 벡터** DB에서 실제 녹음 조건 평가 | 실제 규모 평가 방법론 참고 |
| Pretrained Conformers for FP (arXiv:2508.11609) | 2025 | Conformer + SimCLR 사전학습, **3초 오디오**로 SOTA 주장 | 3초 목표에 직접 부합, 서버 측 고성능 모델 후보 |
| Synthetic Latent Fingerprint Generation (arXiv:2509.18620) | 2025 | Rectified Flow로 가짜 핑거프린트를 만들어 **100만 곡 이상 규모를 저렴하게 시뮬레이션** | 실제 100만 곡을 확보하기 전 확장성 테스트에 활용 |
| Segment Length Matters (arXiv:2601.17690) | 2026 | 세그먼트 길이별 성능 비교 | **3초 미만 쿼리에서는 짧은 세그먼트(0.5초)가 유리** → 세그먼트 길이 설계 근거 |

### 5.5 신경망 방식 장단점
- **장점**: 잡음·잔향·마이크 왜곡에 크게 강함, 증강 설계로 원하는 환경(예: 차 안, 클럽)에 맞춰 학습 가능, 벡터 하나로 표현돼 구조가 깔끔함.
- **단점**: 학습 데이터·GPU 필요, 벡터 수가 많아 인덱스가 큼(PQ 필수), 모델을 바꾸면 **전체 DB 재색인**, 휴대폰에서 추론 시 모델 경량화(양자화, TFLite/Core ML) 필요, 거짓 양성 임계값 설계가 해시 방식보다 까다로움.

---

## 6. 상용 서비스 참고

| 서비스 | 알려진 특징 |
|---|---|
| Shazam (Apple) | 랜드마크 해시 원조. 수천만 곡 DB |
| Google (Now Playing / Sound Search) | 신경망 임베딩, 기기 내 + 서버 |
| ACRCloud | 음악·방송 모니터링 API, 커버/허밍 인식 별도 제공 |
| Gracenote (Nielsen) | 음악·TV 인식, 메타데이터 |
| Audible Magic | 저작권 콘텐츠 식별(업로드 필터링) |
| BMAT | 방송·공연장 음악 사용 모니터링(저작권 정산) |

> 참고: SoundHound의 허밍 검색이나 커버곡 인식은 "같은 녹음(동일 음원) 찾기"가 아니라 "같은 곡(멜로디) 찾기"라 다른 문제다. 이 프로젝트 범위는 **동일 음원 식별**로 본다.

---

## 7. 대규모 인덱스 설계 (10만 / 100만 곡)

가정: 평균 곡 길이 3.5분(210초).

### 7.1 방식 A: 피크 해시 역색인

| 항목 | 10만 곡 | 100만 곡 |
|---|---|---|
| 해시 밀도 (튜닝 값) | 초당 약 30~50개 | 초당 약 30~50개 |
| 전체 해시 수 | 약 6억~10억 | 약 63억~105억 |
| 저장량 (엔트리 8바이트) | 약 5~8 GB | **약 50~85 GB** |

- **구조**: `hash(32비트) → posting list [(song_id, t)]`. 해시 값으로 정렬된 배열 + 오프셋 테이블(CSR 형태)이 메모리 효율이 가장 좋음. posting list는 델타 인코딩·가변 길이 정수로 압축.
- **확장**: 해시 범위 또는 곡 ID 범위로 **샤딩**. 쿼리는 모든 샤드에 병렬 요청 후 병합.
- **튜닝 포인트**: 너무 흔한 해시(stop-hash)는 잘라내 posting 길이 상한 설정, 해시 밀도와 fan-out으로 정확도↔인덱스 크기 조절.
- **저장소 선택**: 메모리 상주 커스텀 구조(C++/Rust) 권장. Redis/MySQL 같은 범용 DB는 100만 곡 규모에서 느리고 비쌈.

### 7.2 방식 B: 신경망 임베딩 + ANN

| 항목 | 10만 곡 | 100만 곡 |
|---|---|---|
| 세그먼트 수 (0.5초 홉) | 약 4,200만 | **약 4.2억** |
| 원본 float32 128차원 | 약 21 GB | 약 215 GB |
| PQ 64바이트 압축 | 약 2.7 GB | **약 27 GB** |
| PQ 32바이트 압축 | 약 1.3 GB | 약 13 GB |

- **인덱스**: FAISS `IVF + PQ`(예: IVF 65,536 클러스터, PQ 64바이트) 또는 HNSW+PQ. GPU FAISS로 고속화 가능. 대안: ScaNN, Milvus, Qdrant(관리형 벡터 DB).
- **검색 절차**: 쿼리 3초 → 약 5개 세그먼트 → 각 세그먼트 top-k 후보 → `(song_id, 위치 − 쿼리 내 위치)` 기준으로 모아 **시퀀스 일관성 점수** → 상위 후보만 원본 벡터로 재채점.
- **크기 절감**: 세그먼트 홉 1초로 늘리기(절반), 차원 64로 축소, HRR처럼 여러 세그먼트를 합성, 인기 곡 우선 메모리·나머지 디스크(DiskANN 계열).

### 7.3 권장 최종 구조: 하이브리드

```
[휴대폰]
  녹음 1초마다 → (a) 피크 해시  (b) 경량 임베딩(선택)  → 서버로 전송 (원음은 보내지 않음)
[서버]
  1차: 피크 해시 역색인으로 후보 곡 + 시간 오프셋 빠르게 확보
  2차: 후보 곡의 해당 위치 임베딩과 비교해 재검증 (거짓 양성 차단)
  해시로 못 찾으면: 임베딩 ANN 검색으로 대체 경로
  점수가 임계값 넘으면 즉시 응답 → 클라이언트 녹음 중단 (보통 2~3초 안에 종료)
```

---

## 8. 휴대폰 실시간 녹음 실무 고려사항

1. **OS 음성 처리 끄기**: 기본 녹음 모드는 노이즈 억제·AGC·에코 제거가 음악을 망가뜨림.
   - Android: `AudioSource.UNPROCESSED`(지원 기기) 또는 `VOICE_RECOGNITION`
   - iOS: `AVAudioSession` 모드 `.measurement`
2. **샘플레이트**: 녹음은 44.1/48 kHz, 핑거프린트용으로 8~16 kHz로 다운샘플(대부분 정보는 4 kHz 이하에 있음).
3. **클라이언트에서 DNA 추출**: 원음 대신 해시/임베딩만 전송 → 전송량 수 KB, 개인정보 보호, 서버 부하 감소. 단, 알고리즘 업데이트 시 앱 업데이트 필요(버전 필드 포함 권장).
4. **스트리밍 + 조기 종료**: 0.5~1초 단위로 누적 전송, 서버는 누적 증거로 점수 갱신.
5. **학습·평가 시 왜곡 재현**: 배경 잡음(카페·거리·차량), 실내 잔향(IR), 휴대폰 마이크 IR, 코덱 압축, 음량 변화, 시작 위치 랜덤.

---

## 9. 평가 계획 제안

- **데이터**
  - 공개: FMA(Free Music Archive, full 약 10.6만 곡) — NAFP 등 대부분 논문이 사용, 비교 가능
  - 잡음: MUSAN, AudioSet / 잔향: 공개 RIR 데이터셋
  - **실녹음 셋**: 여러 기종 휴대폰으로 실제 장소에서 녹음한 쿼리 수백~수천 개 (가장 중요)
- **지표**
  - 쿼리 길이 1 / 2 / 3 / 5초별 Top-1 정답률
  - **거짓 양성률**(DB에 없는 곡을 다른 곡으로 착각) — 서비스 품질에 결정적
  - 인식까지 걸린 시간, 서버 검색 지연(p50/p99), 인덱스 크기·메모리, 단말 CPU/배터리 사용
  - DB 규모별 곡선(1만 → 10만 → 100만, 합성 핑거프린트 활용)

---

## 10. 다음 단계 제안 (로드맵)

| 단계 | 내용 | 결과물 |
|---|---|---|
| 1 | 피크 해시 방식 기준선 구현 (Python 프로토타입) + FMA 일부로 평가 | 3초 인식률, 인덱스 크기 실측 |
| 2 | 해시 역색인을 C++/Rust 메모리 구조로, 10만 곡 색인 | 검색 지연·메모리 실측 |
| 3 | NAFP 재현 + 증강 파이프라인, FAISS IVF-PQ | 잡음 환경 비교표 |
| 4 | 하이브리드 결합, 거짓 양성 임계값 설계 | 서비스용 판정 로직 |
| 5 | 모바일 SDK (Android/iOS) DNA 추출, 스트리밍 프로토콜 | 실녹음 테스트 |
| 6 | 100만 곡 규모 확장 (샤딩, 합성 디스트랙터 테스트) | 용량 산정서 |

다음 문서 후보: `02-dna-format-spec.md` (DNA 바이너리 형식·버전 정의), `03-baseline-design.md` (1단계 구현 설계).

---

## 참고 문헌

**고전 방식**
- A. Wang, "An Industrial-Strength Audio Search Algorithm," ISMIR 2003.
- J. Haitsma, T. Kalker, "A Highly Robust Audio Fingerprinting System," ISMIR 2002.
- R. Sonnleitner, G. Widmer, "Robust Quad-Based Audio Fingerprinting," IEEE/ACM TASLP, 2016.
- J. Six, M. Leman, "Panako – A Scalable Acoustic Fingerprinting System Handling Time-Scale and Pitch Modification," ISMIR 2014. / Panako 2.0, Olaf: https://github.com/JorenSix
- D. Ellis, audfprint: https://github.com/dpwe/audfprint
- Dejavu: https://github.com/worldveil/dejavu
- Chromaprint / AcoustID: https://acoustid.org/chromaprint

**신경망 방식**
- B. Gfeller et al., "Now Playing: Continuous low-power music recognition," NeurIPS 2017 Workshop.
- S. Chang et al., "Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrastive Learning," ICASSP 2021. https://ar5iv.arxiv.org/html/2010.11910 / 코드: https://github.com/mimbres/neural-audio-fp
- A. Singh et al., "Attention-Based Audio Embeddings for Query-by-Example," https://arxiv.org/html/2210.08624v2
- Holographic Reduced Representations for audio fingerprinting, https://arxiv.org/html/2406.13139v1
- GraFPrint: A GNN-Based Approach for Audio Identification, https://arxiv.org/pdf/2410.10994
- PeakNetFP: Peak-based Neural Audio Fingerprinting Robust to Extreme Time Stretching, https://arxiv.org/pdf/2506.21086
- Nikou & Giannakopoulos, "Contrastive and Transfer Learning for Effective Audio Fingerprinting through a Real-World Evaluation Protocol," https://arxiv.org/abs/2507.06070v1
- Pretrained Conformers for Audio Fingerprinting and Retrieval, https://arxiv.org/html/2508.11609v2
- Scalable Evaluation for Audio Identification via Synthetic Latent Fingerprint Generation, https://arxiv.org/html/2509.18620v1
- Segment Length Matters: A Study of Segment Lengths on Audio Fingerprinting Performance, https://arxiv.org/abs/2601.17690

**검색 인프라**
- FAISS: https://github.com/facebookresearch/faiss
- ScaNN: https://github.com/google-research/google-research/tree/master/scann

> 주의: 표의 인덱스 크기는 가정값으로 계산한 추정치다. 일부 최신 논문(2025~2026) 수치는 초록 기준이므로 구현 전 원문 확인이 필요하다.
