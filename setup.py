from distutils.core import setup
setup(name = 'Rylab',
      description = "Rigel's data handling and analysis tools",
      author = "Rigel Zifkin",
      author_email = "rydgel.code@gmail.com",
      url = "https://github.com/rydgel/rylab",
      packages = ['rylab'],
      package_dir = {'rylab' : '.'},
      package_data = {'rylab' : ['style.mplstyle', 'minPar.so', 'minPar.dll']}
     )
