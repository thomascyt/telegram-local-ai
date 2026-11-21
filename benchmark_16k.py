"""Benchmark script: efficient trial-division primality test."""


def is_prime(n: int) -> bool:
    """Return True if n is prime, using 6k +/- 1 trial division."""
    if n < 2:
        return False
    if n < 4:
        return True  # 2 and 3 are prime
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


if __name__ == "__main__":
    # 1. Smallest prime
    assert is_prime(2) is True
    # 2. Larger prime
    assert is_prime(97) is True
    # 3. Small composite
    assert is_prime(10) is False
    # 4. Larger composite (product of two primes)
    assert is_prime(97 * 89) is False
    # 5. Edge cases: 0, 1, and a negative number
    assert not is_prime(0) and not is_prime(1) and not is_prime(-7)
    print("All 5 assertions passed.")
