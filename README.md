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

To run scripts at the CLI your `PYTHONPATH` will need setting correctly. One way of doing that is adding
`export PYTHONPATH="/path/to/repo"` to your `~/.bashrc`.

## Development

This software has been developed on Linux with Python 3.6 and the [PyCharm](https://www.jetbrains.com/pycharm/) IDE.
The unit tests and scripts should just run within PyCharm with no configuration necessary. Whilst none of this software
has been tested on Windows, there aren't any known reasons why it wouldn't run.

## Fitting

The script [fit.py](./bin/fit.py) can be used to run a fitting ensemble. The fits will start with random coefficients
between `-100%` and `+100%` of the signal coefficient values. Run `./bin/fit.py --help` to see all options and defaults.

You can run it for multiple iterations with either the `-i` or `--iterations` arguments. E.g.:

```
$ ./bin/fit.py -i 1000
```

Results can be logged to a CSV file with the `-c` or `--csv` arguments. If the script is quit, it will continue
appending to the same file when it is restarted. E.g.:

```
$ ./bin/fit.py -i 1000 -c myfile.csv
```

The model to use for the signal coefficients can be specified with the `-S` or `--signal-model` options. The
 default is `SM`. E.g.:

```
$ ./bin/fit.py -i 1000 -c SM_run.csv -S SM
$ ./bin/fit.py -i 1000 -c NP_run.csv -S NP
```

The initialization scheme for the fit coefficients can be chosen with the `-f` or `--fit-init` options. 
You can use a specific algorithm. E.g.:

```
$ ./bin/fit.py -f TWICE_LARGEST_SIGNAL_SAME_SIGN
```

The algorithms available are:

 * `TWICE_LARGEST_SIGNAL_SAME_SIGN`: Initialize coefficients from `0` to `2x` the largest value for each coefficient
  in all signal models.
 * `TWICE_CURRENT_SIGNAL_ANY_SIGN`: Initialize coefficients from `-2x` to `+2x` the value in the signal model used.
 * `CURRENT_SIGNAL`: Initialize coefficients to the same values as the signal model used.
 
Alternatively the fit coefficients can all be initialized to a specific value.
To do that specify a floating point number. E.g.:

```
$ ./bin/fit.py -f 123.456
```

The target device (e.g. `GPU:0`, `GPU:1`, `CPU:0` etc.) can be specified with `-d` or `--device`. This can be useful
if you want to start multiple scripts in parallel running on different devices. The value defaults to `GPU:0`. E.g.:

```
$ ./bin/fit.py -i 1000 -c myfile_gpu3.csv -d GPU:3
```

The optimizer (`-o`/`--opt-name`), learning rate (`-r`/`--learning-rate`) and additional optimizer parameters
(`-p`/`--opt-param`) can be supplied. E.g.

```
$ ./bin/fit.py -o Adam -r 0.01 -p beta1 0.95 -p epsilon 1e-3
```

Gradient clipping by global norm is disabled by default, but can be enabled with the `-P` or `--grad-clip` arguments.
E.g.

```
$ ./bin/fit.py -P 2.5
```

The fits will be considered converged when the maximum gradient for any coefficient is less than `5e-7`. You can change
this value with the `-u` or `--grad-max-cutoff` arguments:

```
$ ./bin/fit.py -u 1e-8
```

## Using Tensorboard

[Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) can be used to tune the optimizer. Values
will be logged for Tensorboard from either [compare_optimizers.py](./bin/compare_optimizers.py), or when running the 
[fit.py](./bin/fit.py) script with the `-l` or `--log` arguments. Note that logging statistics has a large performance
hit so should not be used for production runs. Once scripts that have logging enabled start, they will output
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
$ sudo -E --preserve-env=PYTHONPATH ./bin/profile.py
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

Plots:

* Plot CP averaged/antisymmetric observables?
* Implement test statistics (arXiv:1504.00574 chapter 5) (Use largest range)
* Implement P-value comparison? (Use largest range)
* Implement likelihood distributions?
* Plot C9 vs C10

Cleanup:

* Lazy importing for BW?
* Make CSV reading into module/write tests.
* Cleanup plotter/write docs
* Split signal.py into other files (e.g. observables, decay_rate). Sort coeffs vs amplitudes params.
* Check/complete all docstrings
* Split signal plotting into separate file
* Get scripts to output plots for scientific publication. Make plotting work in PyCharm and at CLI.
* Rename/document all scripts
* Add warnings to docs about adding/changing signal coefficients & the effect on "largest" initialization
* Test all scripts/unit tests
* Ensure default optimizer params are sensible
* Document commands for how data and figures were generated for report
* Put paper in repo

Further:

* Add real signal values for a_00_l and a_00_r.
* Tune the optimizer better to improve fitting performance and quality.
* Add background. Will need B-meson mass term in PDF, a background event generator composed of polynomials,
and fitting based on nuisance parameters for those polynomials.