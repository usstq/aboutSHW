# How to debug the kernel using shim in windows
### init vs2022
`%comspec% /k "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"`
### init cm sdk
`sdk\path\setenv.bat`
### compilation config
```
md build && cd build
cmake ..
```
Then open `gemm.sln` in `build` and build all.
### debug config
- copy `sdk\path\cemu\bin\*` to `build\debug`
- set working dir to `build\debug`
- set `-DCM_GENX=1276` for xe1 or `-DCM_GENX=1295` for xe2 in `CMakeLists.txt`
- set envrionment `CM_RT_PLATFORM=ats` for xe1 or `CM_RT_PLATFORM=elg` for xe2
