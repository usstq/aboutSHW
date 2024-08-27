## [test-bw.cpp](./test-bw.cpp)

Measure memory read bandwidth

```bash
g++ -O2 -fopenmp ./test-bw.cpp
numactl -C0-42 -m0 ./a.out
```

## [ir2py](./ir2py.py)

convert OpenVINO IR into python script which generates the IR, this is useful for:
 - inspecting the model topology
 - modifing the topology manually

```bash
# convert IR timm_vit_base.xml into a python script
$ python ir2py.py ./timm_vit/timm_vit_base.xml > timm_vit_base.py
```

## [testLLM.py](./testLLM.py)

test openvino LLM performance.
```bash
# test Llama2-7b using fake prompt of 1024 tokens, generate 128 tokens, batch-size 1 & 32
$ numactl -C56-98 -m1 python testLLM.py ./Llama-2-7b-hf/ -p 1024 -n 128 -b 1 1 1 32 32 32 -d 0
# the prompt was extract from https://en.wikipedia.org/wiki/Oxygen, which would generate very determined output `...atoms of the element bind to form dioxygen`
$ numactl -C56-111 -m1 python testLLM.py ./Llama-2-7b-hf/ -p "Oxygen is a chemical element; it has symbol O and atomic number 8. It is a member of the chalcogen group in the periodic table, a highly reactive nonmetal, and an oxidizing agent that readily forms oxides with most elements as well as with other compounds. Oxygen is the most abundant element in Earth's crust, and after hydrogen and helium, it is the third-most abundant element in the universe. At standard temperature and pressure, two" -n 128 -b 1 1 1 4 4 4 8 8 8 -d 0
```
