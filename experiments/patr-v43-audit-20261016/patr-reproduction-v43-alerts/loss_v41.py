"""Paper-authorized trajectory mean PPO with clipped low-variance KL."""
def low_var_kl(logp, ref):
    delta = (ref - logp).clamp(-20, 20)
    return (delta.exp() - delta - 1).clamp(-10, 10)


def trajectory_mean(values, lengths, minibatch_size):
    # Each micro contributes its trajectory means to the SAME actual minibatch.
    # Never normalize each micro independently; the short tail uses its true size.
    if not lengths or any(n <= 0 for n in lengths) or sum(lengths) != values.numel():
        raise ValueError('Assistant lengths must partition all selected tokens')
    if minibatch_size < len(lengths):
        raise ValueError('Invalid optimizer minibatch size')
    return sum(chunk.mean() for chunk in values.split(lengths)) / minibatch_size


def token_mean(values, minibatch_tokens):
    """v4.2: verl/DAPO token-mean. sum over this micro / ALL assistant tokens of the
    actual optimizer minibatch, so micros accumulate to one exact minibatch mean.
    Every token gets the same reduction weight; weights sum to one.
    This does not bound neural-network parameter gradient norms."""
    if minibatch_tokens <= 0 or values.numel() > minibatch_tokens:
        raise ValueError('Invalid minibatch token count')
    return values.sum() / minibatch_tokens
