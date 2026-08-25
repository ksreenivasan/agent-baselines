from inspect_ai.model import ChatMessageUser, ContentImage

from agent_baselines.evals.hle_tools_v0.dataset import load_hle_dataset


def test_fixture_is_one_multimodal_sample():
    dataset = load_hle_dataset(fixture=True)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.id == "fixture-multimodal-001"
    assert sample.target == "1"
    assert sample.metadata["answer_type"] == "exact_match"
    assert isinstance(sample.input[0], ChatMessageUser)
    assert any(isinstance(item, ContentImage) for item in sample.input[0].content)
