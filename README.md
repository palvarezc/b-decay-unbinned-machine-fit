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

The first time scripts are run it may take a long time to start when using the GPU as CUDA creates its compute cache.
Subsequent runs should all start faster.

Unit tests can be run from the CLI by running the following from the project folder:

```
$ python -m unittest
```

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
$ ./bin/fit.py -o Adam -r 0.01 -p beta_1 0.95 -p epsilon 1e-3
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
allows all options that [fit.py](./bin/fit.py) does.

Additionally the `-n`/`--null-model` and  `-t`/`--test-model` arguments are mandatory. These specify which
signal coefficient model the P-wave fit coefficients should be fixed to for the null and test hypotheses respectively.

An example usage is:

```
$ ./bin/q_test_statistic.py -c Q_NP.csv -i 1000 -S NP -t NP -n SM
$ ./bin/q_test_statistic.py -c Q_SM.csv -i 1000 -S SM -t NP -n SM
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

## Publication Data

Data was generated for publication by running the [generate\_data.sh](./bin/generate_data.sh) bash script.

Values and plots were subsequently mined by the [process\_data.sh](./bin/process_data.sh) bash script. This
script will log to the file `results/process_data.log` so that outputted values of interest can be later found.

## Misc scripts

The other scripts take no arguments and do the following:

* [benchmark.py](./bin/benchmark.py): Tests speed of key fit functions. Frequently used to test for performance 
regressions during development. You should run this before and after modifying PDF terms as Tensorflow's 
[autograph](https://www.tensorflow.org/guide/autograph) can be quite particular
* [coeff\_contours.py](./bin/coeff_contours.py): Produces a surface plot of the negative log likelihood by scanning
over two coefficients whilst keeping the rest at signal values. Used during early development to ensure minima
existed. To change the two coefficients plotted, change the `cx_idx` and `cy_idx` IDs in that file.
* [coeff\_curves.py](./bin/coeff_curves.py): Produces plots of the negative log likelihood for each coefficient by
scanning them whilst keeping the rest at signal values. Used during early development to ensure minima existed.
* [compare\_optimizers.py](./bin/compare_optimizers.py): Takes combinations of optimizer algorithms and parameters
defined in the `combos` variable in that file and logs the fits for viewing in Tensorboard. Useful for picking
macro machine learning options (e.g. which optimizer to use). For more subtle settings an ensemble fit should be
performed instead as [compare\_optimizers.py](./bin/compare_optimizers.py) fixes all initial coefficients to `1.0`
so that comparison is possible - which isn't very realistic.
* [table\_mean\_err\_pull.py](./bin/table_mean_err_pull.py): Output the LaTeX for a table of signal values, means,
standard errors and pull means for a given CSV fit file. Takes a CSV file as a single argument. Used for publication
* [table\_signal\_coeffs.py](./bin/table_signal_coeffs.py): Output the LaTeX for a table of all the coefficient values.
Used for publication.
* [time\_taken.py](./bin/time_taken.sh): Takes a CSV file as an argument and outputs the average time taken per fit.


## Roadmap

Fixes needed:

* Change signal coefficient values to ones that produce correct shaped observable plots (see paper)
* Replace a_00_* signal coefficient values with proper values (see paper)
* If a fit is partially written to a CSV, the signal coefficients are changed, and then the fit is resumed, the fitting
script will rightly complain about the change and refuse to continue. However if the TWICE\_LARGEST\_SIGNAL\_SAME\_SIGN
algorithm is being used and either another signal model is added or another signal model is changed, then when resuming
the fit script won't notice if the coefficient initialisation values are different to what they were when the ensemble
started. One way of fixing this would be to write an extra header row in the CSV to say what the initialisation values
were when first started. The fit script could either continue using those, or refuse to start if they're different.

Further work:

* Tune the optimizer better to improve fitting performance and quality.
* Add background. Will need B-meson mass term in PDF, a background event generator composed of polynomials,
and fitting based on nuisance parameters for those polynomials.

Potential cleanups:

* Move uses of `csv.DictReader` into a CSV reader class
* Split `signal.py` into other files (e.g. `observables.py`, `decay_rate.py`). Without negatively affecting fit
performance, address the fact that some of those  functions take a flat coefficient list but others take amplitudes
(the  3 `decay_rate_angle_integrated*()` functions are a good example of this inconsistency as can be seen
in `plot_frac_s.py`)
* Combine plotters into single script