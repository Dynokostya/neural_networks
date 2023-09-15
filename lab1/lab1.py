import numpy as np


def task1(x: int):
    # 1.1
    a: int = 1
    b: float = 2
    c: str = 'hello'
    d: list = [1, '2', 3.4]
    e: dict = {'a': 1, 'b': 2}
    f: set = {1, 2, 3}
    g: tuple = (1, 2, 3)

    out: list = [a, b, c, d, e, f, g]
    print('---1.1---')
    print(*out, sep=', ')

    # 1.2
    print('\n---1.2---')
    print('Even') if x % 2 == 0 else print('Odd')

    # 1.3
    print('\n---1.3---')
    for i in out[:-1]:
        print(i, end=', ')
    else:
        print(out[-1])

    i = 0
    while i < len(out[:-1]):
        print(out[i], end=', ')
        i += 1
    else:
        print(out[-1])

    # 1.4
    def factorial(num: int):
        if num == 1:
            return 1
        return num * factorial(num - 1)

    print('\n---1.4---')
    print(factorial(5))


def task2():
    # 2.1
    arr1 = np.array([1, 2])
    arr2 = np.array([[1, 2], [3, 4]])
    print('\n---2.1---')
    print(arr1.shape)
    print(arr2.shape)

    # 2.2
    print('\n---2.2---')
    print(arr1 * 2)
    print(arr2 + 1)

    # 2.3
    print('\n---2.3---')
    arr3 = np.arange(0, 1, 0.1)
    print(arr3)

    # 2.4
    arr4 = np.array([3, 4])
    print('\n---2.4---')
    print(arr1 * arr4)


task1(1)
task2()
