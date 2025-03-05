

```bash
pip install --upgrade py-libnuma

python -c "import csrc; csrc.simd_test_basic(); csrc.simd_test_tput('all', 30); csrc.simd_test_printreg()"

python -c "import csrc; csrc.test()"

```
