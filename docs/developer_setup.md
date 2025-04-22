# Developer Setup

This repository uses [pixi](https://pixi.sh/latest/) to manage the environment through the pyproject.toml file.

Note that installing a pixi environment will automatically install the dem-handler project in editable mode.

Using pixi is beneficial as it allows us to store named requirements in the pyproject.toml file, and provides a pixi.lock file that captures specific versions whenever the pixi environment is solved/installed.

## Install pixi

Follow the [pixi installation guide](https://pixi.sh/latest/#installation).

## Install pixi environments
Environments are associated with the project.

* The `default` environment contains packages required for the code base (e.g. gdal, rasterio).
* The `dev` environment contains everything from the `default` environment, PLUS packages required for tests (e.g. pytest).

`cd` to the repository folder and install the environments:

To install both environments, run
```bash
pixi install --all
```

## Install pre-commit tasks
For development, we use `pre-commit` to run linting with black, as well as some extra tasks to ensure the pixi.lock file and environment.yml files are up-to-date.

As a one-time action, run 
```
pixi run -e dev pre-commit install
```

## Run predefined tasks
Pixi supports tasks (similar to using a Makefile) which can help automate common actions. In the repo, we have the following tasks, associated with the `dev` environment:
* `download_test_data`, which will download test data from AWS
* `test`, which depends on `download_test_data` and will then run `pytest`

To run tests, use
```bash
pixi run test
```

## Run a single command using pixi
For the default environment, use
```bash
pixi run <command>
```

For the `dev` environment, use
```bash
pixi run -e dev <command>
```

### Activate the environment
For longer sessions, you can activate the environment by running
```bash
pixi shell
```
or 
```bash
pixi shell -e dev
```
To exit the shell, run 
```bash
exit
```

## Add dependencies
If wanting to install from conda-forge, use
```bash
pixi add <conda-forge-package>
```

If wanting to install from pypi, use
```
pixi add --pypi <pypi-package>
```

When a new package, consider whether it is required to run the code, or to do development/run tests. 

If needed for development, add it to the dev environment:
```bash
pixi add --feature dev --pypi <pypi-package>
```