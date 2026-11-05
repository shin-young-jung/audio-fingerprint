# audio-fingerprint

휴대폰 실시간 녹음(3초 내외)으로 음원을 식별하고, 서버에서 10만~100만 곡 규모로 검색 가능한 오디오 핑거프린트 시스템.

## 문서
- [01. 오디오 핑거프린트 기술 리서치](docs/01-research-audio-fingerprinting.md)
- [02. 심층 기술 검토 (성능 수치·라이선스·수정 계획)](docs/02-deep-review.md)
- [03. 휴대폰 재녹음 테스트셋 절차서](docs/03-phone-test-protocol.md)
- [04. 알고리즘 선택 결정 문서 (승인 요청)](docs/04-algorithm-decision.md)
- [05. 상용 음원 검색 서비스 기술 조사 (샤잠·사운드하운드·구글·네이버 등)](docs/05-industry-research.md)
- [06. 알고리즘 맞대결 중간 결과 (진행 중)](docs/06-head-to-head-interim.md)

## 기준선 코드 (`afp/`)
샤잠식 피크 쌍 해시 구현과 평가 도구. 알고리즘 비교용 기준선이며 최종 제품 알고리즘이 아니다.

```bash
pip install numpy scipy pytest   # ffmpeg 필요
python -m pytest
python -m afp.evaluate build --out idx.npz refs/*.wav
python -m afp.evaluate recording --index idx.npz --recordings rec.wav --order order.csv --songs-dir refs
```
