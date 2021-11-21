#!/usr/bin/env python
"""
ollam - test

@author: phdenzel
"""
from tests.prototype import SequentialTestLoader
from tests.utils_test import UtilsModuleTest


def main():

    loader = SequentialTestLoader()

    loader.proto_load(UtilsModuleTest)

    loader.run_suites(verbosity=1)


if __name__ == "__main__":

    main()

    
