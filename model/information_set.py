import contextlib
import os
import sys

# Avoid printing the annoying error message
with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
    import polytope as pc  # or wherever the message gets triggered

import numpy as np


def compute_extreme_points(vector):
    equal_to_one = [1]*len(vector) #sum w == 1
    A_matrix = [equal_to_one]
    b_vector = [1]

    #Loop for preferences, if fj > fi => 1/fi - 1/fj < 0

    for i in range(len(vector)):
        fi = vector[i]
        for j in range(i+1, len(vector)):
            fj = vector[j]
            zeros = [0]*len(vector)
            if fj > fi:
                zeros[i] = 1/fi
                zeros[j] = -1/fj
            elif fj < fi:
                zeros[i] = -1/fi
                zeros[j] = 1/fj

            A_matrix.append(zeros)
            b_vector.append(0)

    for i in range(len(vector)):
        zero_vector = [0]*len(vector) 
        one_vector = zero_vector.copy()
        minus_one_vector = zero_vector.copy()
        one_vector[i] = 1 #w < 1
        minus_one_vector[i] = -1 #w > 0
        b_vector.append(1)
        b_vector.append(0)
        A_matrix.append(one_vector)
        A_matrix.append(minus_one_vector)

    A = np.asarray(A_matrix)

    b = np.asarray(b_vector)
    p = pc.Polytope(A, b)
    pp = pc.extreme(p)
    if pp is not None:
        pp = [x for x in pp if sum(x) >0.5] #remove 0
    else:
        print("No extreme points found")
        return []
    
    return np.array(pp)


if __name__ == "__main__":
    # No preference
    # vector = [1,1] 

    # Second metric twice as important
    vector = [1,2]

    pp = compute_extreme_points(vector)

    print(pp)

    # This has been checked and this works 21.05.2024
