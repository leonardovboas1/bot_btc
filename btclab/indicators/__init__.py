"""Pine-faithful indicator library (pure functions: numpy arrays in, arrays out).

Submodules: base (ta.* semantics, HTF helper), gaussian_channel, cipher_b, stoch_rsi,
momentum, trend, registry. Functions are deliberately not re-exported here because
several share their module's name (e.g. ``gaussian_channel.gaussian_channel``).
"""
from . import base, cipher_b, gaussian_channel, momentum, registry, stoch_rsi, trend  # noqa: F401
