from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="custom_op",
    version="0.0.0",
    description="A custom PyTorch MPS extension.",
    ext_modules=[
        cpp_extension.CppExtension(
            "custom_op",
            ["CustomOP.cpp"],
            extra_compile_args=[
                "-arch",
                "arm64",
                "-mmacosx-version-min=12.0",
            ],
            extra_link_args=[
                "-mmacosx-version-min=12.0",
            ],
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    author="JohanLi233",
    author_email="li_zhonghan@qq.com",
    url="https://github.com/JohanLi233",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Programming Language :: Metal",
        "Operating System :: MacOS :: MacOS X",
    ],
    python_requires=">=3.8",
)
