import config
import torch
import torch.optim as optim
from model import YoloV3
from loss import YoloLoss
from tqdm import tqdm

def train_fn(model, train, test, loss_fn, optimizer, scaler, scaled_anchors):
    loop = tqdm(train, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE)
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                    loss_fn(out[0], y0, scaled_anchors[0])
                    + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2])
            )

        optimizer.zero_grad()
        losses += [loss]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)


def main():
    model = YoloV3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train, test = config.utils.load_dataset('/8examples.csv', '/test.csv', num_classes=config.NUM_CLASSES)

    scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(model, train, test, loss_fn, optimizer, scaler, scaled_anchors)


if __name__ == '__main__':
    main()
