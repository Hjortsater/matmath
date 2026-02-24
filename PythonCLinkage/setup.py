from setuptools import setup, Extension
import os

build_temp_dir = os.path.join(os.path.dirname(__file__), "build_temp")

cmat_module = Extension(
    name="cmat",
    sources=[
        "boiler_plate.c",
        "../src/cmat.c"
    ],
    include_dirs=[os.path.dirname(__file__)],  # <-- lets gcc find cmat.h
    extra_compile_args=["-O3"],
)

setup(
    name="cmat",
    version="0.1",
    ext_modules=[cmat_module],
)