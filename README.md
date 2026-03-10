# mmwave-analysislib

This is the analysis library for mm-wave lab.

## Installation

Clone this repository with `git clone`.
Then, on a command line and a Python environment of your choice,
navigate to this directory and run
```
pip install -e . --config-settings editable_mode=strict
```
This creates an editable installation of the package.
The configuration for a [strict install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html#strict-editable-installs) is optional,
but helpful for code editor autocomplete features.

## Example import

```python
from analysislib.common.tweezer_correlator import TweezerCorrelator
```