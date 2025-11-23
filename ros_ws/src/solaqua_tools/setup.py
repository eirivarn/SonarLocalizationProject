from setuptools import setup

package_name = "solaqua_tools"

setup(
    name=package_name,
    version="0.0.1",
    packages=[],
    py_modules=[],
    install_requires=["numpy", "pandas", "opencv-python", "scipy", "pyyaml", "matplotlib", "plotly", "pyarrow"],
    author="SOLAQUA",
    author_email="you@example.com",
    description="ROS1 wrapper for SOLAQUA compute_net_distance",
    license="MIT",
)
