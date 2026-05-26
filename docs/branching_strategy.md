# 브랜치 전략

## 기본 구조

```text
main
  안정 버전 브랜치
  발표나 시연에 바로 사용할 수 있는 상태만 반영한다.

develop
  개발 통합 브랜치
  팀원 작업 결과를 먼저 모으고 테스트하는 브랜치다.

feature/*
  기능별 작업 브랜치
  각 기능은 develop에서 새 브랜치를 만들어 작업한다.
```

## 현재 브랜치

```text
main
develop
feature/release-detection
feature/trajectory-overlay
feature/board-simulator
feature/led-control
feature/docs-update
```

## 작업 시작 방법

작업 전에는 항상 `develop`을 최신으로 맞춘다.

```bash
git switch develop
git pull origin develop
```

기능 브랜치로 이동한다.

```bash
git switch feature/trajectory-overlay
```

만약 새 기능 브랜치를 만들고 싶으면 다음처럼 만든다.

```bash
git switch develop
git pull origin develop
git switch -c feature/새기능이름
git push -u origin feature/새기능이름
```

## 작업 완료 후 push

```bash
git status
git add .
git commit -m "작업 내용"
git push
```

## Pull Request 규칙

기능 브랜치 작업이 끝나면 GitHub에서 Pull Request를 만든다.

```text
base: develop
compare: feature/기능이름
```

즉, 기능 브랜치는 바로 `main`으로 보내지 않고 `develop`으로 먼저 보낸다.

## main 반영 규칙

`develop`에서 전체 기능이 안정적으로 동작하면 `main`으로 Pull Request를 만든다.

```text
base: main
compare: develop
```

`main`은 발표 가능한 안정 버전으로 유지한다.

## 주의사항

- `main`에 직접 push하지 않는다.
- `develop`에도 가능하면 직접 push하지 않고 Pull Request로 합친다.
- `.venv/`, `videos/*.mp4`, `output/*`, `book/`은 GitHub에 올리지 않는다.
- 작업 시작 전에 반드시 `git pull origin develop`으로 최신 상태를 받는다.
- 충돌이 나면 혼자 해결하기보다 팀원에게 어떤 파일에서 충돌이 났는지 공유한다.
