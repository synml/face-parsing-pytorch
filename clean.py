import os
import platform
import shutil

shutil.rmtree('cam', ignore_errors=True)
shutil.rmtree('demo', ignore_errors=True)
shutil.rmtree('result', ignore_errors=True)
shutil.rmtree('runs', ignore_errors=True)
shutil.rmtree('weights', ignore_errors=True)
shutil.rmtree('__pycache__', ignore_errors=True)
shutil.rmtree('calculators/__pycache__', ignore_errors=True)
shutil.rmtree('datasets/__pycache__', ignore_errors=True)
shutil.rmtree('models/__pycache__', ignore_errors=True)
shutil.rmtree('utils/__pycache__', ignore_errors=True)
os.makedirs('weights')

if platform.system() == 'Windows':
    os.system('cls')
else:
    os.system('clear')
