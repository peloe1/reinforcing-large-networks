import numpy as np

if __name__ == "__main__":
    a: list[np.ndarray] = [np.asarray([1,2,3,4]), np.asarray([4,5,6]), np.asarray([7,8,9,5,6])]

    print(sorted(a, key=lambda x: x.shape[0]))
