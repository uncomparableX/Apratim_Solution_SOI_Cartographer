from setuptools import setup, find_packages

package_name = "Apratim_Solution"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(),

    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),

        (
            "share/" + package_name,
            ["package.xml"],
        ),

        (
            "share/" + package_name + "/launch",
            ["launch/solution.launch.py"],
        ),

        (
            "share/" + package_name + "/config",
            [
                "config/parameters.yaml",
                "config/navigation.yaml",
                "config/exploration.yaml",
                "config/communication.yaml",
                "config/mapping.yaml",
            ],
        ),
    ],

    install_requires=[
        "setuptools",
        "numpy",
        "scipy",
        "opencv-python",
        "networkx",
        "scikit-learn",
        "transforms3d",
        "PyYAML",
    ],

    zip_safe=True,

    maintainer="Apratim Solution",

    maintainer_email="ee24bt002@iitdh.ac.in",

    description="SOI Cartographer Competition Solution",

    license="Apache-2.0",

    tests_require=["pytest"],

    entry_points={
        "console_scripts": [
            "solution_node = Apratim_Solution.main:main",
        ],
    },
)