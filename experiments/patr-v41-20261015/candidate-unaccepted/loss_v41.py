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
