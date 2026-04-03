"""Demo buggy script: a simple function with a division-by-zero bug."""


def compute_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    # Bug: no guard for empty list → ZeroDivisionError
    return total / count


if __name__ == "__main__":
    data = [10, 20, 30]
    print(f"Average of {data}: {compute_average(data)}")

    empty = []
    print(f"Average of {empty}: {compute_average(empty)}")
