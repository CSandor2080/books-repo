import torch
from ray.train.torch import TorchTrainer
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim

from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig

def train_func(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST('/data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(
        num_workers=2,
        use_gpu=False
    ),
    run_config=RunConfig(
        checkpoint_config=CheckpointConfig(
            checkpoint_frequency=1
        )
    )
)

# Use tune.run to start the training job with the specified config
tune.run(
    trainer,
    config={
        "batch_size": 64,
        "lr": 0.001,
        "epochs": 5
    }
)
