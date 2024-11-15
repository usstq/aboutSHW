call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
rmdir /s /q build
del clops\cl*.pyd
pip install -e .
