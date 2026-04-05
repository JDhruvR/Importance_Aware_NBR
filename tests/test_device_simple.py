# Simple test that doesn't actually run the function but checks syntax
import ast
import sys


def test_device_syntax():
    """Test that device.py has correct syntax without executing it."""
    with open("nbr/utils/device.py", "r") as f:
        content = f.read()

    try:
        tree = ast.parse(content)
        print("Device utility syntax is valid")
        return True
    except SyntaxError as e:
        print(f"Syntax error in device.py: {e}")
        return False


if __name__ == "__main__":
    success = test_device_syntax()
    sys.exit(0 if success else 1)
