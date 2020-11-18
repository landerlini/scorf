import setuptools 

with open ("README.md", "r") as fh:
  long_description = fh.read() 


setuptools.setup ( 
  name = 'scorf',
  version = '0.0',
  scripts = [],
  author = 'Lucio Anderlini',
  author_email = 'l.anderlini@gmail.com',
  description = 'Sampling from a Conditioned Random Forest',
  long_description = long_description,
  long_description_content_type = 'text/markdown', 
  url = 'https://github.com/landerlini/scorf',
  packages = setuptools.find_package(),
  classifiers = [
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License', 
    'Operating System :: OS Independent',
  ]
)
