def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    margin = (
        z
        * ((proportion * (1 - proportion) + z2 / (4 * total)) / total) ** 0.5
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
