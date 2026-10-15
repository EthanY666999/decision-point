STATISTICAL = frozenset({'gradient_above_20x','assistant_length_above_3x','token_limit_above_5_percent','step_seconds_above_4x'})
def partition(reasons):
    return ([r for r in reasons if r not in STATISTICAL], [r for r in reasons if r in STATISTICAL])
