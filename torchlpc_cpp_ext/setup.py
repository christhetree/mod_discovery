from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="torchlpc_cpp",
    ext_modules=[cpp_extension.CppExtension("torchlpc_cpp", ["torchlpc.cpp"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
