"""Train a small network to predict x, y, yaw from the front camera.

Training uses only the first 896 s of the night drive. The locked laps are
scored once after training. Daylight has no positions, so it is not in this
score. This does not replace output/submission_retrieval.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ejo_vision.evaluate import interpolate_pose  # noqa: E402
from ejo_vision.retrieve import DEV_END_S, VAL_END_S  # noqa: E402
from retrieve_localize import GT_CSV, TRAIN_MCAP, _gt_arrays, window_metrics  # noqa: E402

WIDTH, HEIGHT = 96, 48
EPOCHS = 12
BATCH = 64
LR = 1e-3
FL = "/constructor0/sensor/camera_fl/compressed_image"
WEIGHTS = ROOT / "output" / "pose_net.pt"


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 4)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image).flatten(1))


def load_frames():
    tg, gx, gy, gyaw = _gt_arrays(GT_CSV)
    images, stamps, xy, yaw = [], [], [], []
    with open(TRAIN_MCAP, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _schema, _channel, _message, msg in reader.iter_decoded_messages(topics=[FL]):
            ts = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            pose = interpolate_pose(ts, tg, gx, gy, gyaw)
            if pose is None:
                continue
            gray = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_GRAYSCALE)
            small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            images.append(small)
            stamps.append(ts)
            xy.append((pose[0], pose[1]))
            yaw.append(pose[2])
    return (
        np.stack(images),
        np.asarray(stamps, dtype=np.int64),
        np.asarray(xy, dtype=np.float64),
        np.asarray(yaw, dtype=np.float64),
        tg,
        gx,
        gy,
        gyaw,
    )


def main() -> None:
    images, stamps, xy, yaw, tg, gx, gy, gyaw = load_frames()
    t0 = int(tg[0])
    dev = stamps <= t0 + int(DEV_END_S * 1e9)
    print(f"frames {len(images)}  train {int(dev.sum())}  later {int((~dev).sum())}")

    mux, muy = xy[dev, 0].mean(), xy[dev, 1].mean()
    sx = float(xy[dev, 0].std()) or 1.0
    sy = float(xy[dev, 1].std()) or 1.0
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PoseNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    train_img = torch.from_numpy(images[dev]).float().unsqueeze(1) / 255.0
    train_y = torch.stack(
        [
            torch.from_numpy((xy[dev, 0] - mux) / sx),
            torch.from_numpy((xy[dev, 1] - muy) / sy),
            torch.from_numpy(np.sin(yaw[dev])),
            torch.from_numpy(np.cos(yaw[dev])),
        ],
        dim=1,
    ).float()

    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(train_img))
        total = 0.0
        for start in range(0, len(order), BATCH):
            idx = order[start : start + BATCH]
            pred = model(train_img[idx].to(device))
            loss = loss_fn(pred, train_y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        print(f"epoch {epoch + 1:02d}  train loss {total / len(train_img):.4f}")

    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(images), BATCH):
            batch = torch.from_numpy(images[start : start + BATCH]).float().unsqueeze(1) / 255.0
            pred = model(batch.to(device)).cpu().numpy()
            px = pred[:, 0] * sx + mux
            py = pred[:, 1] * sy + muy
            pyaw = np.atan2(pred[:, 2], pred[:, 3])
            for ts, x, y, yaw_hat in zip(stamps[start : start + BATCH], px, py, pyaw):
                rows.append((int(ts), float(x), float(y), float(yaw_hat)))

    WEIGHTS.parent.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "mux": mux, "muy": muy, "sx": sx, "sy": sy}, WEIGHTS)
    print(f"wrote {WEIGHTS.name}")
    print("\ntrained network, night drive")
    window_metrics("development", rows, tg, gx, gy, gyaw, t0, 0.0, DEV_END_S)
    window_metrics("validation", rows, tg, gx, gy, gyaw, t0, DEV_END_S, VAL_END_S)
    window_metrics("tail", rows, tg, gx, gy, gyaw, t0, VAL_END_S, 1e7)


if __name__ == "__main__":
    main()
