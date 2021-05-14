from setuptools import setup
from setuptools import find_packages
import os

dir_repo = os.path.abspath(os.path.dirname(__file__))
# read the contents of REQUIREMENTS file
with open(os.path.join(dir_repo, "requirements.txt"), "r") as f:
    requirements = f.read().splitlines()
# read the contents of README file
with open(os.path.join(dir_repo, "README.md"), encoding="utf-8") as f:
    readme = f.read()

setup(
    name="syntheticprophet",
    version="0.1",
    description="Library for creating synthetic time series",
    url="https://github.com/neuralprophet/synthetic_prophet",
    author="Rodrigo Rivera-Castro",
    author_email="rodrigo.rivera@yahoo.de",
    license="MIT",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "torch",
        "symengine>=0.4",
        "jitcdde==1.4",
        "jitcxde_common==1.4.1",
    ],
    tests_require=["pytest"],
    setup_requires=["pytest-runner"],
    scripts=["scripts/neuralprophet_dev_setup"],
)
