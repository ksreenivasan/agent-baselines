from agent_baselines.evals.hle_tools_v0.task import hle_tools_v0


def test_task_uses_turn_and_time_bounds_without_cumulative_token_limit():
    task = hle_tools_v0(fixture=True)
    assert task.token_limit is None
    assert task.message_limit == 80
    assert task.time_limit == 1800
