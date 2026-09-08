#!/usr/bin/env python3
"""Build the test model: a plausible Micro-class Ants policy.

Not a good player -- the weights are random. What matters is that it is a *real* ONNX graph of the
shape a competitor would actually submit, so `/inspect` reports real operators and real parameter
counts, `/validate` measures real FLOPs at the shapes the adapter produced, and `/play` runs a real
batched inference. A hand-written two-node graph would let every one of those be wrong.

The architecture is the one the platform design §5 sanity-checks: a small fully-convolutional trunk over
the board, gathered at the ants' positions to a per-ant policy over five moves. It takes the three
inputs the reference adapter produces (`board`, `ant_r`, `ant_c`) and answers the one it consumes
(`policy`, [N, 5]).

    python3 tests/fixtures/make-model.py          # needs torch and onnx
"""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))


class Ants(nn.Module):
    def __init__(self, planes=6, ch=24, moves=5):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(planes, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, moves, 1),
        )

    def forward(self, board, ant_r, ant_c):
        x = self.trunk(board.float())                    # [1, moves, H, W]
        x = x.squeeze(0).flatten(1)                      # [moves, H*W]
        idx = ant_r.long() * board.shape[-1] + ant_c.long()
        return x.index_select(1, idx).transpose(0, 1)    # [N, moves]


class AntsDense(nn.Module):
    """The same trunk, authored to **batch**.

    The difference is the whole of what batching depends on, and it is the competitor's choice, not
    the loader's: this one takes only the board, with a dynamic leading dimension, and answers a
    dense policy map for the whole board. Every seat in a wave then feeds a tensor of the same
    shape -- one preset per wave means one map size -- so the loader can stack them and run one
    inference for the wave.

    The per-unit gather moves out of the graph and into the adapter's `out` program, which is why
    `out` is handed the observation alongside the outputs (docs/design.md §3.2): the ant positions are in
    the observation, and without them the dense answer cannot be turned into moves.

    `Ants` above cannot batch, and nothing is wrong with it: its `ant_r`/`ant_c` are ragged across
    seats, so no two rows agree in shape. It is the honest example of a model that costs one
    inference per seat.
    """

    def __init__(self, planes=6, ch=24, moves=5):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(planes, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, moves, 1),
        )

    def forward(self, board):
        return self.trunk(board.float())             # [B, moves, H, W]


def main():
    torch.manual_seed(7)  # the fixture must be byte-identical on every machine
    m = Ants().eval()
    board = torch.zeros(1, 6, 128, 128, dtype=torch.int8)
    ant_r = torch.zeros(90, dtype=torch.int32)
    ant_c = torch.zeros(90, dtype=torch.int32)
    out = os.path.join(HERE, "ants-micro.onnx")
    torch.onnx.export(
        m, (board, ant_r, ant_c), out,
        input_names=["board", "ant_r", "ant_c"], output_names=["policy"],
        dynamic_axes={"board": {2: "H", 3: "W"}, "ant_r": {0: "N"},
                      "ant_c": {0: "N"}, "policy": {0: "N"}},
        opset_version=17, dynamo=False)
    print(f"wrote {out} ({os.path.getsize(out)} bytes, "
          f"{sum(p.numel() for p in m.parameters())} parameters)", file=sys.stderr)

    torch.manual_seed(7)
    d = AntsDense().eval()
    out = os.path.join(HERE, "ants-dense.onnx")
    torch.onnx.export(
        d, (torch.zeros(1, 6, 128, 128, dtype=torch.int8),), out,
        input_names=["board"], output_names=["policy"],
        dynamic_axes={"board": {0: "B", 2: "H", 3: "W"}, "policy": {0: "B", 2: "H", 3: "W"}},
        opset_version=17, dynamo=False)
    print(f"wrote {out} ({os.path.getsize(out)} bytes, "
          f"{sum(p.numel() for p in d.parameters())} parameters)", file=sys.stderr)


if __name__ == "__main__":
    main()
