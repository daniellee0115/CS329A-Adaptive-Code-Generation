from .simple_samplers import GreedyMethod, SampleMultiple
import re

__all__ = ["GreedyMethod", "SampleMultiple"]


def get_sampler(
    method: str,
    model: str,
    system_prompt: str = None,
    n_samples: int = 3,
    temperature: float = 0.0,
):
    if method == "sample_multiple":
        assert temperature > 0.0, "Sample multiple requires temperature > 0"
        return SampleMultiple(model, system_prompt, n_samples, temperature)
    elif method == "greedy":
        assert (
            temperature == 0.0
        ), "Greedy decoding does not use temperature, it will be 0 by default"
        return GreedyMethod(model, system_prompt)
    else:
        raise ValueError(f"Unknown method: {method}")

def extract_code(response: str) -> str:
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        extracted_text = match.group(1)
        return extracted_text
    else:
        return ""
