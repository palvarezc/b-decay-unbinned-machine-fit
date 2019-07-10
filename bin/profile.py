#!/usr/bin/env python
"""
The script generates profile information that can be viewed in Tensorboard. I (Liam) didn't find it
that useful but I'm committing this anyway in case someone else does.

The script must be run as root due to Nvidia's "recent permission restrictions":
https://devtalk.nvidia.com/default/topic/1047744/jetson-agx-xavier/jetson-xavier-official-tensorflow-package-can-t-initialize-cupti/post/5319306/#5319306

You can run it from the project folder with:
$ env PYTHONPATH="${PYTHON_PATH}:`pwd`" sudo -E --preserve-env=PYTHONPATH python bin/profile.py

If you get errors when running it about missing CUPTI libraries then you will need to locate your CUPTI lib
directory and add it to ldconfig. For Gentoo Linux this can be done with:
# echo '/opt/cuda/extras/CUPTI/lib64' > /etc/ld.so.conf.d/cupti.conf
# ldconfig
That directory is almost certainly different in other Linux distributions.

Tensorboard can be launched from the project folder with:
$ tensorboard --logdir={}/ --host=127.0.0.1 --port=6006

Note that the Profile tab in Tensorboard only works in Chrome. In Firefox you will get a blank page.
"""
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

with bmf.Script(log=True) as script:
    if not script.user_is_root():
        script.stderr('This script needs root permissions. You can run it from the project folder with:')
        script.stderr(
            'env PYTHONPATH="${PYTHON_PATH}:`pwd`" sudo -E --preserve-env=PYTHONPATH python bin/profile.py')
        exit(1)

    signal_coeffs = bmf.coeffs.signal()
    signal_events = bmf.signal.generate(signal_coeffs)

    optimizer = bmf.Optimizer(
        script,
        bmf.coeffs.fit(),
        signal_events,
        'Adam',
        signal_coeffs=signal_coeffs,
        learning_rate=0.10
    )

    for i in range(100):
        tf.summary.trace_on(graph=True, profiler=True)
        optimizer.minimize()
        tf.summary.trace_export(name='trace_%d' % optimizer.step, step=optimizer.step, profiler_outdir=script.log.dir)
        tf.summary.flush()
