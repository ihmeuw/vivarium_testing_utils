======================
Vivarium Testing Utils
======================

**NOTE: This repository has been archived.**

The ``vivarium_testing_utils`` package has been renamed and migrated into the
`vivarium-suite monorepo <https://github.com/ihmeuw/vivarium-suite>`_.

What changed
------------

- **Import path:** ``vivarium_testing_utils`` -> ``vivarium.testing_utils``
- **Source:** ``ihmeuw/vivarium_testing_utils`` (archived) ->
  ``ihmeuw/vivarium-suite`` (under ``libs/testing-utils/``)

To migrate fully to the new package
-----------------------------------

**Install:**

.. code-block:: bash

    pip install vivarium-testing-utils  # no change here

**Import:**

.. code-block:: python

    import vivarium.testing_utils  # was: import vivarium_testing_utils

Original package overview
=========================

This is a repository that will store utility features to help test
Vivarium software.

**Vivarium Testing Utils requires Python 3.8-3.11 to run**

You can install ``vivarium_testing_utils`` from PyPI with pip:

  ``> pip install vivarium_testing_utils``

or build it from source with

  ``> git clone https://github.com/ihmeuw/vivarium_testing_utils.git``

  ``> cd vivarium_testing_utils``

  ``> conda create -n ENVIRONMENT_NAME python=3.11``

  ``> pip install -e .[dev]``
