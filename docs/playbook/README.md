# 플레이북 — 범용 설계·작업 기준

어느 프로젝트에나 그대로 복사해 쓰는 문서다. **무엇을 어떻게 만들지**(설계 절차, 문서 뼈대, 품질 기준)만 담고, 특정 제품·서버·명령은 담지 않는다.
**Claude 모델·세션을 어떻게 나눠 쓸지**(관리자 세션, 하위 개발 세션, 버그 세션)는 [../claude-ops/](../claude-ops/README.md) 에 따로 있다.

| 파일 | 언제 |
|---|---|
| [ARCHITECTURE_PROMPT.md](ARCHITECTURE_PROMPT.md) | 새 프로젝트·큰 모듈의 첫 단계(설계). 요구사항 질문 → **기능 정의(메뉴·기능·흐름, 사용자 확인)** → 기술·구조 결정(사용자 승인) → 아키텍처 문서·CLAUDE.md·저장소 뼈대 → 작업 분할 |
| [SPEC_TEMPLATE.md](SPEC_TEMPLATE.md) | `docs/SPEC.md` 기능 정의서 뼈대(역할별 메뉴 구조·기능 목록(ID·우선순위·완료 기준)·사용자 흐름·화면 정의·디자인 기준·하지 않는 것) — 무엇을 만들지 |
| [ARCHITECTURE_TEMPLATE.md](ARCHITECTURE_TEMPLATE.md) | `docs/ARCHITECTURE.md` 뼈대(목적·범위·구조·결정 기록·데이터·권한·화면·배포·작업 분할) — 어떻게 만들지 |
| [QA_CHECKLIST.md](QA_CHECKLIST.md) | 모든 작업의 완료 기준·화면 검수·검증 요령 |
| [CLAUDE_TEMPLATE.md](CLAUDE_TEMPLATE.md) | 새 프로젝트의 `CLAUDE.md` 뼈대 |

## 흐름
설계 `ARCHITECTURE_PROMPT`(기능 정의 `SPEC` → 아키텍처 `ARCHITECTURE` → 작업 분할) → 구축 → 유지보수. 구축·유지보수를 어떤 세션 구성으로 할지는 `claude-ops`. 모든 단계의 완료 기준은 `QA_CHECKLIST`.

## 새 프로젝트에 들이는 법
1. `playbook/` 과 `claude-ops/` 폴더를 새 프로젝트의 `docs/` 아래로 복사한다.
2. 새 대화에서 "`docs/playbook/ARCHITECTURE_PROMPT.md` 읽고 그대로 해. 요청: ○○" 로 설계부터 한다(모델은 `claude-ops/README.md`). 이 단계에서 `docs/SPEC.md`, `docs/ARCHITECTURE.md`, 루트 `CLAUDE.md`(모든 세션이 자동으로 읽음), 저장소 뼈대가 만들어진다.
3. 구축·버그 수정은 `claude-ops/README.md` 의 '쓰는 법'대로.

## 원칙 (이 모음 전체)
- 대화·보고는 사용자의 언어로, 결론부터 짧게.
- 보고서를 믿지 말고 직접 재현해서 검증한다. 해 보지 않은 것은 '미확인'으로 따로 말한다.
- 되돌릴 수 없는 작업(삭제·발송·외부 공개·결제)은 사용자에게 먼저 묻는다.
- 접속 정보·비밀번호는 `.env` 류에만 두고 커밋·출력하지 않는다.
