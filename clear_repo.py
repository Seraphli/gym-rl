import shutil

WARNING = """
Warning: Delete all generated folders!
"""
print(WARNING)
shutil.rmtree('tmp', ignore_errors=True)
shutil.rmtree('log', ignore_errors=True)
shutil.rmtree('model', ignore_errors=True)
