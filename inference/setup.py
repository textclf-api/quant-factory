"""Customer/local installer for a prebuilt TQ distribution tree.

This setup.py does not compile proprietary code. A customer bundle must already
contain these two compiled files inside tq/:
  runtime_ops.<abi>.so
  tq_kernels.<abi>.so

Use ``pip install .``. GGUF is not supported.
"""
from pathlib import Path
from setuptools import find_packages, setup

HERE = Path(__file__).resolve().parent
PKG = HERE / "tq"

runtime_bins = list(PKG.glob("runtime_ops*.so"))
kernel_bins = list(PKG.glob("tq_kernels*.so"))
if not runtime_bins or not kernel_bins:
    raise RuntimeError(
        "This is the customer installer and requires prebuilt private binaries "
        "inside tq/: runtime_ops*.so and tq_kernels*.so. "
        "Build privately with setup_closed.py, then create the customer bundle."
    )

setup(
    name="tq-quant",
    version="0.4.1",
    packages=find_packages(),
    package_data={"tq": ["runtime_ops*.so", "tq_kernels*.so"]},
    include_package_data=True,
    install_requires=["vllm==0.27.1", "safetensors", "huggingface_hub"],
    python_requires=">=3.12,<3.13",
    entry_points={
        "vllm.general_plugins": ["register_tq = tq:register"],
    },
    zip_safe=False,
)
