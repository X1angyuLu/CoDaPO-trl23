

def convert_to_serializable(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif hasattr(obj, "__dict__"):
        return convert_to_serializable(vars(obj))
    else:
        return str(obj)

def make_conversation(example, prompt_column: str, system_prompt=None):
    prompt = []

    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})

    if prompt_column not in example:
        raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

    prompt.append({"role": "user", "content": example[prompt_column]})
    return {"prompt": prompt}