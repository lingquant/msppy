## This setup.py refers to Django: https://github.com/django/django ##
from setuptools import setup, find_packages
import sys

REQUIRED_GUROBI = (7, 0)
try:
    import gurobipy
    if gurobipy.gurobi.version() < REQUIRED_GUROBI:
        sys.stderr.write("""
        This version of MSP requires gurobipy 7+. Please check Gurobi website to install it properly and then try again:
            $ python install setup.py
        """)
        sys.exit(1)

except:
    sys.stderr.write("""
    This version of MSP requires gurobipy 7+. Please check Gurobi website to install it properly and then try again:
        $ python install setup.py
    """)
    sys.exit(1)

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 5)

# This check and everything above must remain compatible with Python 2.7.
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of MSP requires Python {}.{}, but you're trying to
install it on Python {}.{}.
This may be because you are using a version of pip that doesn't
understand the python_requires classifier. Make sure you
have pip >= 9.0 and setuptools >= 24.2, then try again:
    $ python -m pip install --upgrade pip setuptools
    $ python install setup.py
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)

setup(
    name = 'msppy',
    author = 'Lingquan Ding',
    author_email = 'lding47@gatech.edu',
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    packages = find_packages(),
    install_requires = ['numpy','scipy','pandas','scikit-learn', 'matplotlib'],
    version = '0.1',
    license = 'new BSD',
    description = 'Solve multistage stochastic programs'
)
