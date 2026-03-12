BASE_SQUAT_MODIFIERS = {
    "back": 1,
    "front": 0.85,
    "overhead": 0.65,
}

VARIATION_MODIFIERS = {
    "normal": 1.00,
    "pause": 0.92,
    "tempo": 0.90,
    "in-the-hole": 0.94,
    "1 1/4": 0.88,
}

def epley_multiplier(r: int) -> float:
    if r <= 0:
        raise ValueError("Repetitions must be greater than 0")
    elif r == 1:
        return 1
    else:
        return 1 + r / 30

