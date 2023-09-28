import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

wine = load_wine()
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)


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


def task3():
    # 3.1-3.3
    x = np.arange(-10, 10, 0.1)
    y1 = x ** 2
    y2 = x ** 3
    plt.plot(x, y1, label='y = x^2')
    plt.plot(x, y2, label='y = x^3')
    plt.title('Functions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 3.4
    plt.clf()
    np.random.seed(0)
    data = np.random.normal(0, 1, 1000)
    plt.hist(data, bins=20)
    plt.title('Histogram')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.show()


def task4():
    # 4.1
    wine_df['target'] = wine.target

    # 4.2
    print('\n---4.2---')
    print(wine_df[:5])

    # 4.3
    print('\n---4.3---')
    print(wine_df.describe())

    # 4.4
    class_counts = wine_df['target'].value_counts()
    print('\n---4.4---')
    print(class_counts)


def task5():
    # 5.1
    mean_values = np.mean(wine_df, axis=0)
    print('\n---5.1---')
    print(mean_values)

    # 5.2, 5.4
    for col in wine_df.columns:
        plt.hist(wine_df[col], bins=20)
        plt.title(f'{col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    # 5.3
    wine_df.boxplot(figsize=(12, 6), vert=False)
    plt.title('Box plots')
    plt.xlabel('Values')
    plt.show()

    # 5.5
    correlation_matrix = wine_df.corr()
    print('\n---5.5---')
    print(correlation_matrix)


task1(1)
task2()
task3()
task4()
task5()
