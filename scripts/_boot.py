"""Delay a script's start so samply can attach, then run it as __main__."""

import runpy
import sys
import time

time.sleep(2)
sys.argv = sys.argv[1:]
_ = runpy.run_path(sys.argv[0], run_name="__main__")
