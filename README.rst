======================
Vivarium Testing Utils
======================

**NOTE: This repository is archived and will receive no further updates.**

The ``vivarium_testing_utils`` package's development has migrated into the
`vivarium-suite monorepo <https://github.com/ihmeuw/vivarium-suite>`_.

What changed
------------

- **PyPI distribution:** ``vivarium-testing-utils`` (unchanged - same name)
- **Import path:** ``vivarium_testing_utils`` -> ``vivarium.testing_utils``
- **Source:** ``ihmeuw/vivarium_testing_utils`` (archived) ->
  ``ihmeuw/vivarium-suite`` (under ``libs/testing-utils/``)
- **Docs:** https://vivarium-testing-utils.readthedocs.io/

This repository's final release was ``v0.5.5``. The ``vivarium-testing-utils``
distribution name is now published from the monorepo starting at ``v0.6.0``, so
``pip install vivarium-testing-utils`` resolves to the monorepo release (which
imports as ``vivarium.testing_utils``). This repository is frozen and will not
receive updates.

To migrate fully to the new package
-----------------------------------

**Install (unchanged):**

.. code-block:: bash

    pip install vivarium-testing-utils

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
