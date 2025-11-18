from ev.agent.runner import ModelConfig, AvailableModels


def resolve_model_config(model_str: str) -> ModelConfig:
    # expected formats:
    #   "openai[gpt5_mini]"
    #   "openai[gpt-5]"
    #   "groq[kimi_k2_instruct]"
    #   "groq[moonshotai/kimi-k2-instruct]"
    #   "groq[openai/gpt-oss-120b]"
    try:
        provider_part, rest = model_str.split("[", 1)
        ident = rest.rstrip("]")
    except ValueError:
        raise ValueError(
            f"Invalid model format '{model_str}'. Expected provider[identifier], "
            f"for example 'openai[gpt-5]' or 'groq[kimi_k2_instruct]'."
        )

    provider_key = provider_part.strip()
    ident = ident.strip()

    provider_cls = getattr(AvailableModels, provider_key, None)
    if provider_cls is None:
        raise ValueError(
            f"Unknown provider '{provider_key}'. "
            f"Available providers on AvailableModels are: "
            f"{', '.join(n for n in dir(AvailableModels) if not n.startswith('_'))}"
        )

    # 1) try direct attribute name match (e.g. groq.kimi_k2_instruct)
    if hasattr(provider_cls, ident):
        cfg = getattr(provider_cls, ident)
        if isinstance(cfg, ModelConfig):
            return cfg

    # 2) fallback: search by underlying .name field
    for attr_name in dir(provider_cls):
        cfg = getattr(provider_cls, attr_name)
        if isinstance(cfg, ModelConfig) and cfg.name == ident:
            return cfg

    raise ValueError(
        f"Could not resolve model identifier '{ident}' for provider '{provider_key}'. "
        f"Use an attribute name (e.g. groq[kimi_k2_instruct]) or the exact model "
        f"name (e.g. groq[moonshotai/kimi-k2-instruct])."
    )
