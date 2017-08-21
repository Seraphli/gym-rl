#!/usr/bin/env python

import shutil
import argparse


def parse_args():
    """Arguments for command line"""
    parser = argparse.ArgumentParser(description="Clear repository")
    parser.add_argument("--nuke", action="store_true", help="whether you want to delete models")
    args = parser.parse_args()
    return args


args = parse_args()
WARNING = """
Warning: Delete all generated folders!
"""
print(WARNING)
shutil.rmtree('tmp', ignore_errors=True)
shutil.rmtree('log', ignore_errors=True)
shutil.rmtree('tf_log', ignore_errors=True)
if args.nuke:
    shutil.rmtree('model', ignore_errors=True)
