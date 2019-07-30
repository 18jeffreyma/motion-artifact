import tensorflow as tf

# adapted from https://github.com/grishasergei/conviz/
def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(np.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)

def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]

def pca(X, num_observations=64, n_dimensions = 50):
    singular_values, u, _ = tf.svd(X)
    sigma = tf.diag(singular_values)
    print(sigma)
    
    sigma = tf.slice(sigma, [0, 0], [num_observations, n_dimensions])
    
    pca = tf.matmul(u, sigma)
    pca = tf.transpose(pca)
    return pca