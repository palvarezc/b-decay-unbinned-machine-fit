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

## Q test statistic

The script [q_test_statistic.py](./bin/q_test_statistic.py) can be used to generate Q test statistics. The script
allows all options that [fit.py](./bin/fit.py) does except the `-c` or `--csv` options.

Additionally the `-n`/`--null-model`, `-t`/`--test-model` and `-x`/`--txt` options are required.

The `--null-model` and `--test-model` specifies which signal coefficient model the P-wave fit coefficients
should be fixed to for the null and test hypotheses respectively.

The `--txt` option specifies an output txt file to write to.

An example usage is:

```
$ ./bin/q_test_statistic.py -x Q_NP.txt -i 1000 -S NP -t NP -n SM
$ ./bin/q_test_statistic.py -x Q_SM.txt -i 1000 -S SM -t NP -n SM
```

## Using Tensorboard

[Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) can be used to tune the optimizer. Values
will be logged for Tensorboard from either the [compare_optimizers.py](./bin/compare_optimizers.py),
[fit.py](./bin/fit.py), or [q_test_statistic.py](./bin/q_test_statistic.py) scripts with the `-l` or `--log` arguments.
Note that logging statistics has a large performance hit so should not be used for production runs.
Once scripts that have logging enabled start, they will output the command to start Tensorboard.
Additionally they will output a `Filter regex` that can be used in the left hand pane of the `Scalars` page to filter
that particular run.

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

Cleanup:

* Change q test script to write a CSV including all S-wave coeffs and nlls
* Make CSV reading into module/write tests.
* Split signal.py into other files (e.g. observables, decay_rate). Sort coeffs vs amplitudes params.
* Check/complete all docstrings
* Potentially combine plotters into single script
* Test all scripts/unit tests
* Ensure default optimizer params are sensible
* Document other scripts
* Document plotting
* Document commands for how data and figures were generated for report
* Put paper in repo
* Check Further Work

Fixes needed:

* Change signal coefficient values to ones that produce correct shaped observable plots
* Replace a_00_* signal coefficient values with proper values 
* If a fit is partially written to a CSV, the signal coeffients are changed, and then the fit is resumed, the fitting
script will rightly complain about the change and refuse to continue. However if the TWICE_LARGEST_SIGNAL_SAME_SIGN
algorithm is being used and either another signal model is added or another signal model is changed, then when resuming
the fit script won't notice if the coefficient initialisation values are different to what they were when the ensemble
started. One way of fixing this would be to write an extra header row in the CSV to say what the initialisation values
were when first started. The fit script could either continue using those, or refuse to start if they're different.

Further work:

* Tune the optimizer better to improve fitting performance and quality.
* Add background. Will need B-meson mass term in PDF, a background event generator composed of polynomials,
and fitting based on nuisance parameters for those polynomials.