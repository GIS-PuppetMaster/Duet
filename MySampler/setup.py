from setuptools import setup
import torch
import os,glob

from torch.utils import cpp_extension
from torch.utils.cpp_extension import (CUDAExtension, CppExtension, BuildExtension)
# 这三个extension很重要！

def get_extensions():
    extensions = []
    ext_name = 'mysampler'  # 编译后保存的文件前缀名称及其位置
    # prevent ninja from using too many resources
    os.environ.setdefault('MAX_JOBS', '4')
    define_macros = []

    if torch.cuda.is_available():
        print(f'Compiling {ext_name} with CUDA')
        define_macros += [('WITH_CUDA', None)]
        # 宏处理，会在每个.h/.cpp/.cu/.cuh源文件前添加 #define WITH_CUDA！！这个操作很重要
        # 这样在拓展的源文件中就可以通过#ifdef WITH_CUDA来判断是否编译代码
        op_files = glob.glob('./src/MySampler.cpp')
        extension = CUDAExtension # 如果cuda可用，那么extension类型为CUDAExtension
    else:
        print(f'Compiling {ext_name} without CUDA')
        op_files = glob.glob('./src/MySampler.cpp')
        extension = CppExtension

    # include_path = os.path.abspath('./csrc')
    ext_ops = extension( # 返回setuptools.Extension类
        name=ext_name,
        sources=op_files,
        include_dirs=cpp_extension.include_paths(),
        define_macros=define_macros,
        extra_compile_args=['-O3'])
    extensions.append(ext_ops)
    return extensions # 由setuptools.Extension组成的list

setup(
    name='mysampler',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension}, # BuildExtension代替setuptools.command.build_ext
    )