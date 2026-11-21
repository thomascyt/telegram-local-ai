import pytest
from calculator import add, subtract, multiply, divide


# ── Normal (int) operations ──────────────────────────────────────

def test_add_positive():
    assert add(2, 3) == 5

def test_subtract_positive():
    assert subtract(10, 4) == 6

def test_multiply_positive():
    assert multiply(3, 7) == 21

def test_divide_positive():
    assert divide(15, 3) == 5


# ── Negative numbers ─────────────────────────────────────────────

def test_add_negative():
    assert add(-2, -3) == -5

def test_subtract_mixed_signs():
    assert subtract(-2, 3) == -5

def test_multiply_negative():
    assert multiply(-4, 3) == -12

def test_divide_by_negative():
    assert divide(10, -2) == -5


# ── Floating-point numbers ───────────────────────────────────────

def test_add_floats():
    assert add(1.5, 2.7) == pytest.approx(4.2)

def test_subtract_floats():
    assert subtract(5.0, 2.1) == pytest.approx(2.9)

def test_multiply_floats():
    assert multiply(1.1, 2.2) == pytest.approx(2.42)

def test_divide_floats():
    assert divide(7.5, 2.5) == pytest.approx(3.0)


# ── Zero / division by zero ──────────────────────────────────────

def test_add_zero():
    assert add(5, 0) == 5

def test_subtract_zero():
    assert subtract(5, 0) == 5

def test_multiply_by_zero():
    assert multiply(9, 0) == 0

def test_divide_by_zero_raises_value_error():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(1, 0)
