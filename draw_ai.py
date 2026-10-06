import tkinter as tk

import torch
from torch import nn
from torchvision import transforms

from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt


# =========================
# 1. 학습했던 CNN 구조 그대로 만들기
# =========================

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(32 * 7 * 7, 128),
    nn.ReLU(),

    nn.Linear(128, 10)
)


# =========================
# 2. 저장된 가중치 불러오기
# =========================

model.load_state_dict(
    torch.load("mnist_cnn.pth", map_location="cpu")
)

model.eval()


# =========================
# 3. 기본 설정
# =========================

SIZE = 280

root = tk.Tk()
root.title("My MNIST AI")


# =========================
# 4. 숫자 그리는 캔버스
# =========================

canvas = tk.Canvas(
    root,
    width=SIZE,
    height=SIZE,
    bg="black"
)
canvas.pack(pady=10)

# 실제 AI 입력용 이미지도 따로 저장
image = Image.new("L", (SIZE, SIZE), color=0)
draw = ImageDraw.Draw(image)

last_x = None
last_y = None


def start_draw(event):
    global last_x, last_y
    last_x = event.x
    last_y = event.y


def draw_digit(event):
    global last_x, last_y

    if last_x is None or last_y is None:
        return

    # 화면에 그리기
    canvas.create_line(
        last_x, last_y,
        event.x, event.y,
        fill="white",
        width=20,
        capstyle=tk.ROUND,
        smooth=True
    )

    # AI용 이미지에도 똑같이 그리기
    draw.line(
        (last_x, last_y, event.x, event.y),
        fill=255,
        width=20
    )

    last_x = event.x
    last_y = event.y


def stop_draw(event):
    global last_x, last_y
    last_x = None
    last_y = None


canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw_digit)
canvas.bind("<ButtonRelease-1>", stop_draw)


# =========================
# 5. 확률 막대그래프 캔버스
# =========================

graph = tk.Canvas(
    root,
    width=380,
    height=300,
    bg="white"
)
graph.pack(pady=10)


def draw_probabilities(probabilities):
    graph.delete("all")

    BAR_START = 40
    BAR_MAX_WIDTH = 250

    for number in range(10):
        probability = probabilities[number]
        y = 10 + number * 28

        # 숫자 표시
        graph.create_text(
            20,
            y + 10,
            text=str(number),
            font=("Arial", 12)
        )

        # 막대 길이
        bar_width = probability * BAR_MAX_WIDTH

        graph.create_rectangle(
            BAR_START,
            y,
            BAR_START + bar_width,
            y + 20,
            fill="skyblue"
        )

        # 퍼센트 표시
        graph.create_text(
            330,
            y + 10,
            text=f"{probability * 100:.1f}%"
        )


# =========================
# 6. 이미지 전처리
# =========================

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


def preprocess_current_image():
    """
    현재 그린 숫자를 MNIST처럼 전처리해서
    [1, 1, 28, 28] 텐서로 반환
    """

    img = image.copy()

    bbox = img.getbbox()

    if bbox is None:
        return None

    # 숫자 영역만 자르기
    img = img.crop(bbox)

    width, height = img.size
    size = max(width, height)

    # 정사각형 검은 배경 만들기
    square = Image.new("L", (size, size), color=0)

    x = (size - width) // 2
    y = (size - height) // 2

    square.paste(img, (x, y))

    # MNIST처럼 여백 추가
    square = ImageOps.expand(
        square,
        border=int(size * 0.2),
        fill=0
    )

    # 28x28로 변환
    tensor = transform(square)

    # 배치 차원 추가 -> [1, 1, 28, 28]
    tensor = tensor.unsqueeze(0)

    return tensor


# =========================
# 7. AI 예측
# =========================

def predict():
    tensor = preprocess_current_image()

    if tensor is None:
        result_label.config(text="숫자를 그려주세요.")
        return

    with torch.no_grad():
        output = model(tensor)

        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()

    probs = probabilities[0].tolist()
    confidence = probs[prediction] * 100

    result_label.config(
        text=f"AI 예측: {prediction}\n확신도: {confidence:.2f}%"
    )

    draw_probabilities(probs)


# =========================
# 8. CNN 내부 보기
# =========================

def show_cnn_features():
    tensor = preprocess_current_image()

    if tensor is None:
        result_label.config(text="먼저 숫자를 그려주세요.")
        return

    first_conv = model[0]

    with torch.no_grad():
        feature_maps = first_conv(tensor)

    # [1, 16, 28, 28] -> [16, 28, 28]
    feature_maps = feature_maps.squeeze(0)

    plt.figure(figsize=(8, 8))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature_maps[i].numpy(), cmap="gray")
        plt.title(f"Filter {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# 9. 초기화
# =========================

def clear():
    global image, draw

    canvas.delete("all")
    graph.delete("all")

    image = Image.new("L", (SIZE, SIZE), color=0)
    draw = ImageDraw.Draw(image)

    result_label.config(text="숫자를 그려보세요")


# =========================
# 10. 버튼 / 라벨
# =========================

predict_button = tk.Button(
    root,
    text="AI에게 물어보기",
    command=predict
)
predict_button.pack(pady=5)

feature_button = tk.Button(
    root,
    text="CNN 내부 보기",
    command=show_cnn_features
)
feature_button.pack(pady=5)

clear_button = tk.Button(
    root,
    text="지우기",
    command=clear
)
clear_button.pack(pady=5)

result_label = tk.Label(
    root,
    text="숫자를 그려보세요",
    font=("Arial", 20)
)
result_label.pack(pady=10)


root.mainloop()