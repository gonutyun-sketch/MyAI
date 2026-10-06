# MyFirstAI

PyTorch로 MNIST 손글씨 숫자를 학습하고, 직접 그린 숫자를 예측해 보는 작은 CNN 프로젝트입니다.

## 포함된 기능

- `ai.py`: 완전연결 신경망으로 MNIST를 학습하고 오답 예시를 표시합니다.
- `cnn_ai.py`: 합성곱 신경망(CNN)으로 MNIST를 학습하고 테스트 정확도를 출력합니다.
- `draw_ai.py`: Tkinter 화면에 숫자를 그린 뒤 예측 결과와 0~9 확률을 보여 줍니다.
- `CNN 내부 보기` 버튼으로 첫 번째 합성곱 층의 feature map 16개를 확인할 수 있습니다.
- `mnist_ai.pth`, `mnist_cnn.pth`: 학습된 모델 가중치입니다.

## 실행 환경

- Python 3.14+
- PyTorch 2.14.1
- Torchvision 0.29.1
- Pillow 12.3.0
- Matplotlib 3.11.2

## 설치

프로젝트 폴더에서 가상환경을 만든 뒤 의존성을 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows PowerShell에서 스크립트 실행 정책 때문에 활성화가 막히면 가상환경을 활성화하지 않고 아래처럼 실행해도 됩니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 사용법

CNN을 새로 학습하려면 다음을 실행합니다. 처음 실행할 때 MNIST 데이터가 `data/` 폴더에 다운로드됩니다.

```powershell
python cnn_ai.py
```

학습이 끝나면 `mnist_cnn.pth`가 갱신됩니다. 저장된 CNN 가중치로 손글씨 입력 화면을 실행하려면 프로젝트 루트에서 다음을 실행합니다.

```powershell
python draw_ai.py
```

화면에 숫자를 그린 다음 `AI에게 물어보기`를 누르면 예측 숫자와 확률이 표시됩니다. `CNN 내부 보기`를 누르면 첫 번째 합성곱 층이 입력에서 찾은 특징을 4×4 화면으로 보여 줍니다. `지우기`로 입력을 초기화할 수 있습니다.

완전연결 모델을 학습하고 오답 예시를 확인하려면 다음을 실행합니다.

```powershell
python ai.py
```

## 저장소에 포함하지 않는 파일

가상환경(`.venv/`), 내려받은 MNIST 데이터(`data/`), Python 캐시와 로그는 `.gitignore`로 제외합니다. 모델 가중치 파일은 재현 가능한 실행을 위해 저장소에 포함합니다.

## 참고

`draw_ai.py`는 `mnist_cnn.pth`를 현재 실행 폴더에서 읽습니다. 반드시 이 저장소의 루트 폴더에서 실행하세요.
