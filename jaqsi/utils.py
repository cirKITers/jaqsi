import jax


def safe_random_split(random_key: jax.random.PRNGKey, *args, **kwargs):
    if random_key is None:
        return None, None
    else:
        return jax.random.split(random_key, *args, **kwargs)
