## Introduction

This project is for doing an unbinned `B0 → K* μ μ` angular observable fit based on this
[paper](https://arxiv.org/abs/1504.00574) but using Tensorflow. The fitting is done to a generated toy signal.
See [coeffs.py](./b_meson_fit/coeffs.py) for the values used.

## Requirements

Ensure Tensorflow `1.14` is installed either via from source,
or the `tensorflow-cpu` or `tensorflow-gpu` pip packages. E.g.
```
pip install --user --upgrade 'tensorflow-gpu==1.14.*'
```

Install dependencies:
```
pip install --user --upgrade --upgrade-strategy only-if-needed -r requirements.txt
```

## Development

This software has been developed with the [PyCharm](https://www.jetbrains.com/pycharm/) IDE. The unit tests and 
scripts should just run within it with no configuration necessary. If you're using something else, I assume
you know how to set your `PYTHONPATH` correctly.

Linux has been used for this development. Whilst none of this software has been tested on Windows, there aren't
any known reasons why it wouldn't run.

## Fitting

The script [fit.py](./bin/fit.py) can be used to run a fitting ensemble. Fitted coefficients will be outputted to a CSV file.
If the script is quit, if it continue appending to the same file the next time it is started.

The fit will start with random coefficients between `-100%` and `+100%` of the signal value. 
Some coefficients may take longer than others, in which case the earlier converged ones can start to suffer from
exploding gradients. One technique to address this
has been to implement a timeline of previous gradients. If a gradient stddev is less than `5e-7` over `100` 
iterations, then that coefficient is considered converged and no further optimizing will be applied to it. However, 
this method adds a not insignificant performance hit as it ends up processing on the CPU rather than GPU.

There can still be occasional times where the timeline and stopping certain coefficients is not enough and gradients
still won't settle. To address this, the `fit.py` script will restart the iteration with different random
coefficients if the fit hasn't converged for `20,000` iterations.

Further work needs to be done to see if a combination of tuning optimizer hyper-parameters (e.g. `epsilon`),
gradient clipping, and the iteration restart count can be enough to make fitting work consistently fast and well
without the performance slow down of the timeline.

## Using Tensorboard

[Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) can be used to tune the optimizer. Values
will be logged for Tensorboard from either [compare_optimizers.py](./bin/compare_optimizers.py), or from 
[fit.py](./bin/fit.py) if `log` is set to `True`. Note that logging statistics has a large performance hit
so should not be used for production runs. Once scripts that have logging enabled start, they will output
the command to start Tensorboard. Additionally they will output a `Filter regex` that can be used in the left hand pane
of the `Scalars` page to filter that particular run.

## Profiling

Profiling can be achieved through the [profile.py](./bin/profile.py) script and viewed in Tensorboard. More about how
to use the `Profile` tab can be found
 [here](https://www.tensorflow.org/tensorboard/r2/tensorboard_profiling_keras#trace_viewer).

The script must be run as root due to Nvidia's 
["recent permission restrictions"](https://devtalk.nvidia.com/default/topic/1047744/jetson-agx-xavier/jetson-xavier-official-tensorflow-package-can-t-initialize-cupti/post/5319306/#5319306).
You can run it from the project folder with:
```
$ env PYTHONPATH="${PYTHON_PATH}:`pwd`" sudo -E --preserve-env=PYTHONPATH python bin/profile.py
```
Once the script starts it will output the command to start Tensorboard (which shouldn't be run as root).

If you get errors when running it about missing CUPTI libraries then you will need to locate your CUPTI lib
directory and add it to ldconfig. For Gentoo Linux this can be done with:

```
# echo '/opt/cuda/extras/CUPTI/lib64' > /etc/ld.so.conf.d/cupti.conf
# ldconfig
```

That directory is almost certainly different in other Linux distributions.

Note that the Profile tab in Tensorboard only works in Chrome. In Firefox you will get a blank page.

## Roadmap

* Fix test_decay_rate_frac_s() unit test
* Add real signal values for a_00_l and a_00_r.
* Investigate whether just aborting non-converging runs is quicker than using the gradient timeline. For this
to work there will need to be another method of determining when fit has converged.
* Tune the optimizer better to improve fitting performance and quality.
* Do large ensemble runs and plot results.
* Compare physics models. Use different signal coefficients and compare P values.
* Change scripts to pass options as command line options instead of having variables at the top of files.
* Add background. Will need B-meson mass term in PDF, a background event generator composed of polynomials,
and fitting based on nuisance parameters for those polynomials.