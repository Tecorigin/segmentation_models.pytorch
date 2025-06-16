from setuptools import setup, find_packages

setup(
    name='segmentation_models_pytorch',
    version='0.5.0',
    description='segmentation_models_pytorch',
    packages=find_packages(),
    include_package_data=True,
     install_requires=[
        "huggingface_hub==0.31.1",
        "numpy==1.26.0",
        "pillow==11.2.1",
        "safetensors==0.5.3",
        "timm==1.0.15",
        "tqdm==4.67.1",

        "opencv-python",
        "matplotlib",
        "lightning",
        "albumentations",
    ],
    python_requires='>=3.8',
)


# python setup.py bdist_wheel