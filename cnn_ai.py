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

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)


# =========================
# 2. CNN 만들기
# =========================

model = nn.Sequential(

    # 28x28 이미지에서 특징 16종류 찾기
    nn.Conv2d(
        in_channels=1,
        out_channels=16,
        kernel_size=3,
        padding=1
    ),

    nn.ReLU(),

    # 이미지 크기 절반
    nn.MaxPool2d(2),

    # 더 복잡한 특징 32종류 찾기
    nn.Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        padding=1
    ),

    nn.ReLU(),

    nn.MaxPool2d(2),

    # 32 x 7 x 7 → 1568개
    nn.Flatten(),

    nn.Linear(32 * 7 * 7, 128),

    nn.ReLU(),

    nn.Linear(128, 10)
)


# =========================
# 3. Loss / Optimizer
# =========================

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# 4. 학습
# =========================

epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        predictions = model(images)

        loss = loss_function(
            predictions,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs}"
        f" | Loss: {average_loss:.4f}"
    )


print("학습 완료!")


# =========================
# 5. 테스트
# =========================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        predictions = model(images)

        predicted_numbers = predictions.argmax(dim=1)

        correct += (
            predicted_numbers == labels
        ).sum().item()

        total += labels.size(0)


accuracy = correct / total * 100

print(f"테스트 정확도: {accuracy:.2f}%")


# =========================
# 6. 모델 저장
# =========================

torch.save(
    model.state_dict(),
    "mnist_cnn.pth"
)

print("모델 저장 완료: mnist_cnn.pth")