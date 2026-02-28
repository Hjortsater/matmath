from setuptools import setup, Extension
import shutil
import os

# Force rebuild
if os.path.exists("build"):
    shutil.rmtree("build")

module = Extension(
    "hjortMatrixWrapper",
    sources=["hjortMatrixWrapper.c"],
    libraries=["openblas"],  # <-- link OpenBLAS
    library_dirs=[],         # optional, add path if OpenBLAS is non-standard
    extra_compile_args=["-O3", "-fopenmp", "-mavx2", "-march=native", "-D_GNU_SOURCE"],
    extra_link_args=["-fopenmp"],
)

setup(
    name="hjortMatrixWrapper",
    version="1.0",
    ext_modules=[module],
)