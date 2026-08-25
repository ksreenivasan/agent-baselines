import base64
import io
import json

from inspect_ai.model import ChatMessageUser, ContentImage
from PIL import Image

from agent_baselines.evals.hle_tools_v0.dataset import load_hle_dataset


def test_fixture_is_one_multimodal_sample():
    dataset = load_hle_dataset(fixture=True)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.id == "fixture-multimodal-001"
    assert sample.target == "1"
    assert sample.metadata["answer_type"] == "exact_match"
    assert isinstance(sample.input[0], ChatMessageUser)
    image = next(item for item in sample.input[0].content if isinstance(item, ContentImage))
    with Image.open(io.BytesIO(base64.b64decode(image.image.split(",", 1)[1]))) as decoded:
        assert decoded.size == (16, 16)
        assert decoded.convert("RGB").getpixel((0, 0)) == (0, 0, 255)


def test_manifest_filters_and_preserves_order(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"ids": ["fixture-multimodal-001"]}))
    dataset = load_hle_dataset(fixture=True, manifest_path=manifest)
    assert [sample.id for sample in dataset] == ["fixture-multimodal-001"]
