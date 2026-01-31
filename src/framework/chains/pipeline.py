"""Composable chains: combine chains in a DAG/pipeline config."""

from typing import Any, Optional

from .base import Chain


class PipelineStep:
    """Single step: chain type id and config. Output is passed to next step by key."""

    def __init__(self, step_id: str, chain: Chain, output_key: str = "output"):
        self.step_id = step_id
        self.chain = chain
        self.output_key = output_key


class Pipeline(Chain):
    """Run a sequence of chains; each step receives merged initial inputs + previous outputs."""

    def __init__(self, steps: list[PipelineStep], final_output_key: Optional[str] = None):
        self._steps = steps
        self._final_output_key = final_output_key or (steps[-1].output_key if steps else "output")

    def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> Any:
        state: dict[str, Any] = dict(inputs)
        for step in self._steps:
            out = step.chain.invoke(state, **kwargs)
            state[step.output_key] = out
        return state.get(self._final_output_key, state)


def pipeline_from_config(
    config: list[dict[str, Any]],
    build_chain: Any,
) -> Pipeline:
    """
    Build a Pipeline from a list of step configs.
    config: [{"id": "s1", "output_key": "summary", ...}, ...]  # rest passed to build_chain.
    build_chain: callable(step_config: dict) -> Chain. Builds a chain from one step config.
    """
    steps = []
    for i, c in enumerate(config):
        step_id = c.get("id", f"step_{i}")
        output_key = c.get("output_key", "output")
        chain = build_chain(c)
        steps.append(PipelineStep(step_id=step_id, chain=chain, output_key=output_key))
    final_key = config[-1].get("output_key", "output") if config else "output"
    return Pipeline(steps=steps, final_output_key=final_key)
