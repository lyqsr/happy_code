find . -name '.git'        -type f -print -exec rm -rf {} \;
find . -name '.gitmodules' -type f -print -exec rm -rf {} \;
find . -name '.github'     -type f -print -exec rm -rf {} \;
find . -name '.gitignore'  -type f -print -exec rm -rf {} \;
########
find . -name '*.pyc' -type f -print -exec rm -rf {} \;
find . -name '*.py~' -type f -print -exec rm -rf {} \;
find . -name '__pycache__'   -print -exec rm -rf {} \;
