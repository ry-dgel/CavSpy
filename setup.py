from distutils.core import setup
setup(name = 'CavSpy',
      description = "Library for handling data from CAvity Sensing in Python.",
      author = "Rigel Zifkin",
      author_email = "rydgel.code@gmail.com",
      url = "https://github.com/rydgel/CavSpy",
      packages = ['CavSpy'],
      package_dir = {'CavSpy' : '.'},
      package_data = {'CavSpy' : ['style.mplstyle', 'minPar.so', 'minPar.dll']}
     )
