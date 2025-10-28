from setuptools import find_packages, setup


VERSION = "0.1"

extras = {}
setup(
    name="model_moe",
    version=VERSION,
    description="S'MoRE: Structural Mixture of Residual Experts for Parameter Efficient LLM Fine-tuning",
    license_files=["LICENSE"],
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="Hanqing Zeng",
    author_email="zhqhku@gmail.com",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={},
    python_requires=">=3.9.0",
    # install_requires=[
    #     "numpy>=1.17",
    #     "packaging>=20.0",
    #     "psutil",
    #     "pyyaml",
    #     "torch>=1.13.0",
    #     "transformers",
    #     "tqdm",
    #     "accelerate>=0.21.0",
    #     "safetensors",
    #     "huggingface_hub>=0.25.0",
    # ],
    # extras_require=extras,
    # classifiers=[
    #     "Development Status :: 5 - Production/Stable",
    #     "Intended Audience :: Developers",
    #     "Intended Audience :: Education",
    #     "Intended Audience :: Science/Research",
    #     "License :: OSI Approved :: Apache Software License",
    #     "Operating System :: OS Independent",
    #     "Programming Language :: Python :: 3",
    #     "Programming Language :: Python :: 3.9",
    #     "Programming Language :: Python :: 3.10",
    #     "Programming Language :: Python :: 3.11",
    #     "Programming Language :: Python :: 3.12",
    #     "Topic :: Scientific/Engineering :: Artificial Intelligence",
    # ],
)