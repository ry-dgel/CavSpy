from distutils.core import setup
setup(name = 'cavspy',
      description = "Library for handling data from CAvity Sensing in Python.",
      author = "Rigel Zifkin",
      author_email = "rydgel.code@gmail.com",
      url = "https://github.com/rydgel/CavSpy",
      packages = ['cavspy'],
      package_dir = {'cavspy' : '.'},
      package_data = {'cavspy' : ['style.mplstyle', 'minPar.so', 'minPar.dll']},
      version_config={
         "template": "{tag}.dev{sha}",
        "dirty_template": "{tag}.dev{sha}",
      },
      setup_requires=['setuptools-git-versioning'],
     )
