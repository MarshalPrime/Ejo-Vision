"""Front camera and speed samples from a Constructor Labs bag.

Yields messages in log order. The caller decides which frames enter the
picture book. This module does not read position.
"""

from __future__ import annotations

from pathlib import Path

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

FRONT_CAMERA = "/constructor0/sensor/camera_fl/compressed_image"
SPEED = "/constructor0/state_estimation"
CAMERAS = ("cl", "fl", "fr", "cr", "rl", "r", "rr")


def camera_topic(name: str) -> str:
    return f"/constructor0/sensor/camera_{name}/compressed_image"


def iter_speed_and_front(mcap_path: str | Path):
    """Yield ('speed', t, vx, vy, wz) or ('frame', t, jpeg) in write order.

    t is the message header stamp in nanoseconds. vx is forward, vy is left,
    wz is yaw rate. jpeg is the compressed front-camera bytes.
    """
    with open(mcap_path, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _schema, channel, _message, msg in reader.iter_decoded_messages(
            topics=[SPEED, FRONT_CAMERA]
        ):
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if channel.topic == SPEED:
                yield ("speed", stamp, float(msg.vx_mps), float(msg.vy_mps), float(msg.wz_radps))
            else:
                yield ("frame", stamp, bytes(msg.data))


def iter_all_cameras(mcap_path: str | Path, every: int = 5):
    """Yield speed samples and camera frames in write order.

    Every front frame is included. Other cameras are included on every `every`th
    frame. `jpeg` is None when this frame is not a picture-book query.
    Frames before the first speed sample are skipped, matching the front localizer.
    """
    name_of = {camera_topic(name): name for name in CAMERAS}
    counts = {name: 0 for name in CAMERAS}
    started = False
    with open(mcap_path, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _schema, channel, _message, msg in reader.iter_decoded_messages(
            topics=[SPEED, *name_of]
        ):
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if channel.topic == SPEED:
                started = True
                yield ("speed", stamp, float(msg.vx_mps), float(msg.vy_mps), float(msg.wz_radps))
                continue
            if not started:
                continue
            name = name_of[channel.topic]
            counts[name] += 1
            is_query = counts[name] % every == 0
            if not is_query and name != "fl":
                continue
            jpeg = bytes(msg.data) if is_query else None
            yield ("frame", name, stamp, jpeg)
