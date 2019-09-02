from distutils.core import setup


setup(
    name="gnnutils",
    version="0.1",
    include_package_data=True,
    packages=[
        "gnnutils",
    ],
    package_data = {"gnnutils" : ['data/data.zip']}

)