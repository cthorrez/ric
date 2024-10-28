#!/bin/bash
rm -rf build/ *.so *.c && python setup.py build_ext --inplace