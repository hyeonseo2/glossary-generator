# Google Cloud Run 배포 가이드 (en_ko_glossary)

요구사항
- Google Cloud SDK (`gcloud`) 설치
- Cloud Build/Cloud Run 권한 (Cloud Run Admin, Cloud Build Editor)
- 프로젝트 설정 완료

배포 절차

```bash
gcloud config set project <YOUR_PROJECT_ID>
gcloud run deploy en-ko-glossary \
  --source . \
  --region asia-northeast3 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars PORT=8080
```

주의
- 기본 컨테이너가 첫 실행 시 spaCy 모델 다운로드(`en_core_web_sm`)를 수행하므로 초기 시작이 조금 걸릴 수 있습니다.
- 실행 후 브라우저에서 배포된 URL로 접속하면 로컬 웹 UI와 동일하게 사용 가능:
  - `/` : 전체 문서 목록 + 일괄 추출 실행
  - `/run-all` : POST로 전체 추출 수행
  - `/api/pairs` : 매칭 문서 목록 API
  - `/output/{파일명}` : 결과 파일 다운로드

샘플 실행 명령(로컬)
```bash
cd /home/g0525yhs/en_ko_glossary
uv run python -m glossary_pipeline.web --host 0.0.0.0 --port 8080
```
