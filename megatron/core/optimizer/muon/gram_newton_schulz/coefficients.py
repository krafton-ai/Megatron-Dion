"""
Newton-Schulz coefficient presets.

Each preset is a list of [a, b, c] coefficients for each iteration.
"""

# https://x.com/YouJiacheng/status/1905861218138804534
YOU_COEFFICIENTS = [
    [4.0848, -6.8946, 2.9270],
    [3.9505, -6.3029, 2.6377],
    [3.7418, -5.5913, 2.3037],
    [2.8769, -3.1427, 1.2046],
    [2.8366, -3.0525, 1.2012]
]

# https://arxiv.org/pdf/2505.16932
_unmodified_polar_express_coefficients = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]
safety_factor = 1.05
POLAR_EXPRESS_COEFFICIENTS = [
    (a / safety_factor, b / safety_factor**3, c / safety_factor**5)
    for (a, b, c) in _unmodified_polar_express_coefficients
]
