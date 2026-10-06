import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =========================
# 1. 데이터 준비
# =========================

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)


# =========================
# 2. AI의 뇌 만들기
# =========================

model = nn.Sequential(
    nn.Flatten(),

    nn.Linear(784, 128),
    nn.ReLU(),

    nn.Linear(128, 10)
)


# =========================
# 3. 틀린 정도를 계산하는 방법
# =========================

loss_function = nn.CrossEntropyLoss()


# =========================
# 4. 가중치를 수정하는 방법
# =========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# 5. 학습
# =========================

epochs = 5

for epoch in range(epochs):

    total_loss = 0

    for images, labels in train_loader:

        # 1. AI가 숫자를 예측
        predictions = model(images)

        # 2. 실제 정답과 비교
        loss = loss_function(predictions, labels)

        # 3. 이전에 계산한 기울기 삭제
        optimizer.zero_grad()

        # 4. 어디를 얼마나 수정해야 하는지 계산
        loss.backward()

        # 5. 실제 가중치 수정
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {average_loss:.4f}"
    )
print("학습 완료!")

torch.save(model.state_dict(), "mnist_ai.pth")

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

# =========================
# 6. 성능 테스트
# =========================

correct = 0
total = 0

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        predictions = model(images)

        predicted_numbers = predictions.argmax(dim=1)

        correct += (predicted_numbers == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100

print(f"테스트 정확도: {accuracy:.2f}%")

import matplotlib.pyplot as plt

# 틀린 문제 찾기
model.eval()

wrong_images = []
wrong_answers = []
wrong_predictions = []

with torch.no_grad():
    for images, labels in test_loader:

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        for i in range(len(labels)):
            if predictions[i] != labels[i]:
                wrong_images.append(images[i])
                wrong_answers.append(labels[i].item())
                wrong_predictions.append(predictions[i].item())

            if len(wrong_images) >= 9:
                break

        if len(wrong_images) >= 9:
            break


# 오답 9개 출력
plt.figure(figsize=(8, 8))

for i in range(9):

    plt.subplot(3, 3, i + 1)

    plt.imshow(
        wrong_images[i].squeeze(),
        cmap="gray"
    )

    plt.title(
        f"Answer: {wrong_answers[i]} / AI: {wrong_predictions[i]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()