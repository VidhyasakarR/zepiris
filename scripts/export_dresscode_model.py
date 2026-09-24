"""Export the SigLIP2 image encoder used by the dress-code classifier to ONNX.

The ML service runs the uniform classifier on onnxruntime only (no
transformers/torch at inference). This script produces the encoder file once:

    pip install "transformers>=4.50"          # export-time only
    python scripts/export_dresscode_model.py  # -> models/siglip2_base_vision.onnx

Model: google/siglip2-base-patch16-224 (Apache-2.0). Only the vision tower is
exported; its pooled output (768-d) is what the classifier head in
zepiris/ml_inference/assets/dresscode_head.json was trained on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "google/siglip2-base-patch16-224"


class _Vision(torch.nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision

    def forward(self, pixel_values):
        return self.vision(pixel_values=pixel_values).pooler_output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/siglip2_base_vision.onnx")
    args = ap.parse_args()

    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL_ID).eval()
    wrapper = _Vision(model.vision_model).eval()
    dummy = torch.randn(2, 3, 224, 224)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (dummy,), str(out), input_names=["pixel_values"], output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        ref = wrapper(x).numpy()
    got = sess.run(None, {"pixel_values": x.numpy()})[0]
    cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB); torch vs onnx cosine min {cos.min():.6f}")


if __name__ == "__main__":
    main()
