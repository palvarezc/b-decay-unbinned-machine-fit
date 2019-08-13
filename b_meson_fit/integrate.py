import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()


def trapezoid(integral, x_start, x_stop, dx):
    """
    Integrate function using trapezoid rule as Tensorflow's odeint() is much slower

    Args:
        integral (function): Function that takes 'x' as argument and returns y
        x_start (tensor): Rank-0 tensor of x value to start at
        x_stop( tensor): Rank-0 tensor of x value to stop at
        dx (tensor): Rank-0 tensor of # of trapezoids

    Return:
        tensor: Rank-0 tensor of integrated value. Has the the same dtype as integral() returns
    """
    # Generate points to integrate
    steps = tf.cast((x_stop - x_start) / dx, dtype=tf.int32)
    x = tf.linspace(x_start, x_stop, steps + 1)

    # Get tensor of y values for our x points
    y = integral(x)

    # Make tensors of the start and end y values for our trapezoids
    # So if y=[10.0, 25.0, 45.0, 20.0]
    y_start = y[:-1]  # ... [10.0, 25.0, 45.0]
    y_stop = y[1:]     # ... [25.0, 45.0, 20.0]

    # Make a tensor of our trapezoid areas
    y_combined = tf.math.add(y_start, y_stop)
    trapezoids = (y_combined / tf.constant(2.0, dtype=y_combined.dtype)) * tf.cast(dx, dtype=y_combined.dtype)

    # Add the trapezoids together
    return tf.reduce_sum(trapezoids)
