
```bash
> nvprof.exe --metrics all .\a.exe
==5928== NVPROF is profiling process 5928, command: .\a.exe
==5928== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==5928== Replaying kernel "kernel(int*, int, int)" (done)
==5928== Profiling application: .\a.exe
==5928== Profiling result:
==5928== Metric result:
Invocations                               Metric Name                                                    Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1070 (0)"
    Kernel: kernel(int*, int, int)
          1                             inst_per_warp                                                 Instructions per warp   12.000000   12.000000   12.000000
          1                         branch_efficiency                                                     Branch Efficiency     100.00%     100.00%     100.00%
          1                 warp_execution_efficiency                                             Warp Execution Efficiency     100.00%     100.00%     100.00%
          1         warp_nonpred_execution_efficiency                              Warp Non-Predicated Execution Efficiency     100.00%     100.00%     100.00%
          1                      inst_replay_overhead                                           Instruction Replay Overhead    0.033333    0.033333    0.033333
          1      shared_load_transactions_per_request                           Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request                          Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
          1       local_load_transactions_per_request                            Local Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1      local_store_transactions_per_request                           Local Memory Store Transactions Per Request    0.000000    0.000000    0.000000
          1              gld_transactions_per_request                                  Global Load Transactions Per Request    0.000000    0.000000    0.000000
          1              gst_transactions_per_request                                 Global Store Transactions Per Request    4.000000    4.000000    4.000000
          1                 shared_store_transactions                                             Shared Store Transactions           0           0           0
          1                  shared_load_transactions                                              Shared Load Transactions           0           0           0
          1                   local_load_transactions                                               Local Load Transactions           0           0           0
          1                  local_store_transactions                                              Local Store Transactions           0           0           0
          1                          gld_transactions                                              Global Load Transactions           2           2           2
          1                          gst_transactions                                             Global Store Transactions       20480       20480       20480
          1                  sysmem_read_transactions                                       System Memory Read Transactions           4           4           4
          1                 sysmem_write_transactions                                      System Memory Write Transactions           5           5           5
          1                      l2_read_transactions                                                  L2 Read Transactions         181         181         181
          1                     l2_write_transactions                                                 L2 Write Transactions       20493       20493       20493
          1                           global_hit_rate                                     Global Hit Rate in unified l1/tex       0.00%       0.00%       0.00%
          1                            local_hit_rate                                                        Local Hit Rate       0.00%       0.00%       0.00%
          1                  gld_requested_throughput                                      Requested Global Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          1                  gst_requested_throughput                                     Requested Global Store Throughput  154.44GB/s  154.44GB/s  154.44GB/s
          1                            gld_throughput                                                Global Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          1                            gst_throughput                                               Global Store Throughput  154.44GB/s  154.44GB/s  154.44GB/s
          1                     local_memory_overhead                                                 Local Memory Overhead       0.00%       0.00%       0.00%
          1                        tex_cache_hit_rate                                                Unified Cache Hit Rate      50.00%      50.00%      50.00%
          1                      l2_tex_read_hit_rate                                           L2 Hit Rate (Texture Reads)       0.00%       0.00%       0.00%
          1                     l2_tex_write_hit_rate                                          L2 Hit Rate (Texture Writes)       0.00%       0.00%       0.00%
          1                      tex_cache_throughput                                              Unified Cache Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          1                    l2_tex_read_throughput                                         L2 Throughput (Texture Reads)  0.00000B/s  0.00000B/s  0.00000B/s
          1                   l2_tex_write_throughput                                        L2 Throughput (Texture Writes)  154.44GB/s  154.44GB/s  154.44GB/s
          1                        l2_read_throughput                                                 L2 Throughput (Reads)  1.3649GB/s  1.3649GB/s  1.3649GB/s
          1                       l2_write_throughput                                                L2 Throughput (Writes)  154.54GB/s  154.54GB/s  154.54GB/s
          1                    sysmem_read_throughput                                         System Memory Read Throughput  30.888MB/s  30.888MB/s  30.888MB/s
          1                   sysmem_write_throughput                                        System Memory Write Throughput  38.610MB/s  38.610MB/s  38.610MB/s
          1                     local_load_throughput                                          Local Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          1                    local_store_throughput                                         Local Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          1                    shared_load_throughput                                         Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          1                   shared_store_throughput                                        Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          1                            gld_efficiency                                         Global Memory Load Efficiency       0.00%       0.00%       0.00%
          1                            gst_efficiency                                        Global Memory Store Efficiency     100.00%     100.00%     100.00%
          1                    tex_cache_transactions                                            Unified Cache Transactions           0           0           0
          1                             flop_count_dp                           Floating Point Operations(Double Precision)           0           0           0
          1                         flop_count_dp_add                       Floating Point Operations(Double Precision Add)           0           0           0
          1                         flop_count_dp_fma                       Floating Point Operations(Double Precision FMA)           0           0           0
          1                         flop_count_dp_mul                       Floating Point Operations(Double Precision Mul)           0           0           0
          1                             flop_count_sp                           Floating Point Operations(Single Precision)           0           0           0
          1                         flop_count_sp_add                       Floating Point Operations(Single Precision Add)           0           0           0
          1                         flop_count_sp_fma                       Floating Point Operations(Single Precision FMA)           0           0           0
          1                         flop_count_sp_mul                        Floating Point Operation(Single Precision Mul)           0           0           0
          1                     flop_count_sp_special                   Floating Point Operations(Single Precision Special)           0           0           0
          1                             inst_executed                                                 Instructions Executed       61440       61440       61440
          1                               inst_issued                                                   Instructions Issued       63488       63488       63488
          1                        sysmem_utilization                                             System Memory Utilization     Low (1)     Low (1)     Low (1)
          1                          stall_inst_fetch                              Issue Stall Reasons (Instructions Fetch)      26.68%      26.68%      26.68%
          1                     stall_exec_dependency                            Issue Stall Reasons (Execution Dependency)      15.35%      15.35%      15.35%
          1                   stall_memory_dependency                                    Issue Stall Reasons (Data Request)       0.00%       0.00%       0.00%
          1                             stall_texture                                         Issue Stall Reasons (Texture)       0.00%       0.00%       0.00%
          1                                stall_sync                                 Issue Stall Reasons (Synchronization)       0.00%       0.00%       0.00%
          1                               stall_other                                           Issue Stall Reasons (Other)      11.49%      11.49%      11.49%
          1          stall_constant_memory_dependency                              Issue Stall Reasons (Immediate constant)      27.59%      27.59%      27.59%
          1                           stall_pipe_busy                                       Issue Stall Reasons (Pipe Busy)       0.51%       0.51%       0.51%
          1                         shared_efficiency                                              Shared Memory Efficiency       0.00%       0.00%       0.00%
          1                                inst_fp_32                                               FP Instructions(Single)           0           0           0
          1                                inst_fp_64                                               FP Instructions(Double)           0           0           0
          1                              inst_integer                                                  Integer Instructions      983040      983040      983040
          1                          inst_bit_convert                                              Bit-Convert Instructions           0           0           0
          1                              inst_control                                             Control-Flow Instructions      163840      163840      163840
          1                        inst_compute_ld_st                                               Load/Store Instructions      163840      163840      163840
          1                                 inst_misc                                                     Misc Instructions      655360      655360      655360
          1           inst_inter_thread_communication                                             Inter-Thread Instructions           0           0           0
          1                               issue_slots                                                           Issue Slots       63488       63488       63488
          1                                 cf_issued                                      Issued Control-Flow Instructions        5120        5120        5120
          1                               cf_executed                                    Executed Control-Flow Instructions        5120        5120        5120
          1                               ldst_issued                                        Issued Load/Store Instructions       30720       30720       30720
          1                             ldst_executed                                      Executed Load/Store Instructions       15360       15360       15360
          1                       atomic_transactions                                                   Atomic Transactions           0           0           0
          1           atomic_transactions_per_request                                       Atomic Transactions Per Request    0.000000    0.000000    0.000000
          1                      l2_atomic_throughput                                       L2 Throughput (Atomic requests)  0.00000B/s  0.00000B/s  0.00000B/s
          1                    l2_atomic_transactions                                     L2 Transactions (Atomic requests)           0           0           0
          1                  l2_tex_read_transactions                                       L2 Transactions (Texture Reads)           0           0           0
          1                     stall_memory_throttle                                 Issue Stall Reasons (Memory Throttle)      16.16%      16.16%      16.16%
          1                        stall_not_selected                                    Issue Stall Reasons (Not Selected)       2.22%       2.22%       2.22%
          1                 l2_tex_write_transactions                                      L2 Transactions (Texture Writes)       20480       20480       20480
          1                             flop_count_hp                             Floating Point Operations(Half Precision)           0           0           0
          1                         flop_count_hp_add                         Floating Point Operations(Half Precision Add)           0           0           0
          1                         flop_count_hp_mul                          Floating Point Operation(Half Precision Mul)           0           0           0
          1                         flop_count_hp_fma                         Floating Point Operations(Half Precision FMA)           0           0           0
          1                                inst_fp_16                                                 HP Instructions(Half)           0           0           0
          1                   sysmem_read_utilization                                        System Memory Read Utilization     Low (1)     Low (1)     Low (1)
          1                  sysmem_write_utilization                                       System Memory Write Utilization     Low (1)     Low (1)     Low (1)
          1               pcie_total_data_transmitted                                           PCIe Total Data Transmitted           0           0           0
          1                  pcie_total_data_received                                              PCIe Total Data Received           0           0           0
          1                inst_executed_global_loads                              Warp level instructions for global loads           0           0           0
          1                 inst_executed_local_loads                               Warp level instructions for local loads           0           0           0
          1                inst_executed_shared_loads                              Warp level instructions for shared loads           0           0           0
          1               inst_executed_surface_loads                             Warp level instructions for surface loads           0           0           0
          1               inst_executed_global_stores                             Warp level instructions for global stores        5120        5120        5120
          1                inst_executed_local_stores                              Warp level instructions for local stores           0           0           0
          1               inst_executed_shared_stores                             Warp level instructions for shared stores           0           0           0
          1              inst_executed_surface_stores                            Warp level instructions for surface stores           0           0           0
          1              inst_executed_global_atomics                  Warp level instructions for global atom and atom cas           0           0           0
          1           inst_executed_global_reductions                         Warp level instructions for global reductions           0           0           0
          1             inst_executed_surface_atomics                 Warp level instructions for surface atom and atom cas           0           0           0
          1          inst_executed_surface_reductions                        Warp level instructions for surface reductions           0           0           0
          1              inst_executed_shared_atomics                  Warp level shared instructions for atom and atom CAS           0           0           0
          1                     inst_executed_tex_ops                                   Warp level instructions for texture           0           0           0
          1                      l2_global_load_bytes       Bytes read from L2 for misses in Unified Cache for global loads           0           0           0
          1                       l2_local_load_bytes        Bytes read from L2 for misses in Unified Cache for local loads           0           0           0
          1                     l2_surface_load_bytes      Bytes read from L2 for misses in Unified Cache for surface loads           0           0           0
          1               l2_local_global_store_bytes   Bytes written to L2 from Unified Cache for local and global stores.      655360      655360      655360
          1                 l2_global_reduction_bytes          Bytes written to L2 from Unified cache for global reductions           0           0           0
          1              l2_global_atomic_store_bytes             Bytes written to L2 from Unified cache for global atomics           0           0           0
          1                    l2_surface_store_bytes            Bytes written to L2 from Unified Cache for surface stores.           0           0           0
          1                l2_surface_reduction_bytes         Bytes written to L2 from Unified Cache for surface reductions           0           0           0
          1             l2_surface_atomic_store_bytes    Bytes transferred between Unified Cache and L2 for surface atomics           0           0           0
          1                      global_load_requests              Total number of global load requests from Multiprocessor           0           0           0
          1                       local_load_requests               Total number of local load requests from Multiprocessor           0           0           0
          1                     surface_load_requests             Total number of surface load requests from Multiprocessor           0           0           0
          1                     global_store_requests             Total number of global store requests from Multiprocessor       20480       20480       20480
          1                      local_store_requests              Total number of local store requests from Multiprocessor           0           0           0
          1                    surface_store_requests            Total number of surface store requests from Multiprocessor           0           0           0
          1                    global_atomic_requests            Total number of global atomic requests from Multiprocessor           0           0           0
          1                 global_reduction_requests         Total number of global reduction requests from Multiprocessor           0           0           0
          1                   surface_atomic_requests           Total number of surface atomic requests from Multiprocessor           0           0           0
          1                surface_reduction_requests        Total number of surface reduction requests from Multiprocessor           0           0           0
          1                         sysmem_read_bytes                                              System Memory Read Bytes         128         128         128
          1                        sysmem_write_bytes                                             System Memory Write Bytes         160         160         160
          1                           l2_tex_hit_rate                                                     L2 Cache Hit Rate       0.00%       0.00%       0.00%
          1                     texture_load_requests             Total number of texture Load requests from Multiprocessor           0           0           0
          1                     unique_warps_launched                                              Number of warps launched        5120        5120        5120
          1                             sm_efficiency                                               Multiprocessor Activity      29.44%      29.44%      29.44%
          1                        achieved_occupancy                                                    Achieved Occupancy    0.679567    0.679567    0.679567
          1                                       ipc                                                          Executed IPC    1.297133    1.297133    1.297133
          1                                issued_ipc                                                            Issued IPC    1.350751    1.350751    1.350751
          1                    issue_slot_utilization                                                Issue Slot Utilization      33.77%      33.77%      33.77%
          1                  eligible_warps_per_cycle                                       Eligible Warps Per Active Cycle    2.297685    2.297685    2.297685
          1                           tex_utilization                                             Unified Cache Utilization    Idle (0)    Idle (0)    Idle (0)
          1                            l2_utilization                                                  L2 Cache Utilization     Low (1)     Low (1)     Low (1)
          1                        shared_utilization                                             Shared Memory Utilization    Idle (0)    Idle (0)    Idle (0)
          1                       ldst_fu_utilization                                  Load/Store Function Unit Utilization     Low (2)     Low (2)     Low (2)
          1                         cf_fu_utilization                                Control-Flow Function Unit Utilization     Low (1)     Low (1)     Low (1)
          1                    special_fu_utilization                                     Special Function Unit Utilization    Idle (0)    Idle (0)    Idle (0)
          1                        tex_fu_utilization                                     Texture Function Unit Utilization     Low (3)     Low (3)     Low (3)
          1           single_precision_fu_utilization                            Single-Precision Function Unit Utilization     Low (3)     Low (3)     Low (3)
          1           double_precision_fu_utilization                            Double-Precision Function Unit Utilization    Idle (0)    Idle (0)    Idle (0)
          1                        flop_hp_efficiency                                            FLOP Efficiency(Peak Half)       0.00%       0.00%       0.00%
          1                        flop_sp_efficiency                                          FLOP Efficiency(Peak Single)       0.00%       0.00%       0.00%
          1                        flop_dp_efficiency                                          FLOP Efficiency(Peak Double)       0.00%       0.00%       0.00%
          1                    dram_read_transactions                                       Device Memory Read Transactions          30          30          30
          1                   dram_write_transactions                                      Device Memory Write Transactions       15769       15769       15769
          1                      dram_read_throughput                                         Device Memory Read Throughput  231.66MB/s  231.66MB/s  231.66MB/s
          1                     dram_write_throughput                                        Device Memory Write Throughput  118.92GB/s  118.92GB/s  118.92GB/s
          1                          dram_utilization                                             Device Memory Utilization     Mid (5)     Mid (5)     Mid (5)
          1             half_precision_fu_utilization                              Half-Precision Function Unit Utilization    Idle (0)    Idle (0)    Idle (0)
          1                          ecc_transactions                                                      ECC Transactions           0           0           0
          1                            ecc_throughput                                                        ECC Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          1                           dram_read_bytes                                Total bytes read from DRAM to L2 cache         960         960         960
          1                          dram_write_bytes                             Total bytes written from L2 cache to DRAM      504608      504608      504608

> nvprof.exe --query-metrics

Available Metrics:
                            Name   Description
Device 0 (NVIDIA GeForce GTX 1070):
                   inst_per_warp:  Average number of instructions executed by each warp

               branch_efficiency:  Ratio of non-divergent branches to total branches expressed as percentage

       warp_execution_efficiency:  Ratio of the average active threads per warp to the maximum number of threads per warp supported on a multiprocessor

warp_nonpred_execution_efficiency:  Ratio of the average active threads per warp executing non-predicated instructions to the maximum number of threads per warp supported on a multiprocessor

            inst_replay_overhead:  Average number of replays for each instruction executed

shared_load_transactions_per_request:  Average number of shared memory load transactions performed for each shared memory load

shared_store_transactions_per_request:  Average number of shared memory store transactions performed for each shared memory store

local_load_transactions_per_request:  Average number of local memory load transactions performed for each local memory load

local_store_transactions_per_request:  Average number of local memory store transactions performed for each local memory store

    gld_transactions_per_request:  Average number of global memory load transactions performed for each global memory load.

    gst_transactions_per_request:  Average number of global memory store transactions performed for each global memory store

       shared_store_transactions:  Number of shared memory store transactions

        shared_load_transactions:  Number of shared memory load transactions

         local_load_transactions:  Number of local memory load transactions

        local_store_transactions:  Number of local memory store transactions

                gld_transactions:  Number of global memory load transactions

                gst_transactions:  Number of global memory store transactions

        sysmem_read_transactions:  Number of system memory read transactions

       sysmem_write_transactions:  Number of system memory write transactions

            l2_read_transactions:  Memory read transactions seen at L2 cache for all read requests

           l2_write_transactions:  Memory write transactions seen at L2 cache for all write requests

                 global_hit_rate:  Hit rate for global loads in unified l1/tex cache. Metric value maybe wrong if malloc is used in kernel.

                  local_hit_rate:  Hit rate for local loads and stores

        gld_requested_throughput:  Requested global memory load throughput

        gst_requested_throughput:  Requested global memory store throughput

                  gld_throughput:  Global memory load throughput

                  gst_throughput:  Global memory store throughput

           local_memory_overhead:  Ratio of local memory traffic to total memory traffic between the L1 and L2 caches expressed as percentage

              tex_cache_hit_rate:  Unified cache hit rate

            l2_tex_read_hit_rate:  Hit rate at L2 cache for all read requests from texture cache

           l2_tex_write_hit_rate:  Hit Rate at L2 cache for all write requests from texture cache

            tex_cache_throughput:  Unified cache throughput

          l2_tex_read_throughput:  Memory read throughput seen at L2 cache for read requests from the texture cache

         l2_tex_write_throughput:  Memory write throughput seen at L2 cache for write requests from the texture cache

              l2_read_throughput:  Memory read throughput seen at L2 cache for all read requests

             l2_write_throughput:  Memory write throughput seen at L2 cache for all write requests

          sysmem_read_throughput:  System memory read throughput

         sysmem_write_throughput:  System memory write throughput

           local_load_throughput:  Local memory load throughput

          local_store_throughput:  Local memory store throughput

          shared_load_throughput:  Shared memory load throughput

         shared_store_throughput:  Shared memory store throughput

                  gld_efficiency:  Ratio of requested global memory load throughput to required global memory load throughput expressed as percentage.

                  gst_efficiency:  Ratio of requested global memory store throughput to required global memory store throughput expressed as percentage.

          tex_cache_transactions:  Unified cache read transactions

                   flop_count_dp:  Number of double-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count.

               flop_count_dp_add:  Number of double-precision floating-point add operations executed by non-predicated threads.

               flop_count_dp_fma:  Number of double-precision floating-point multiply-accumulate operations executed by non-predicated threads. Each multiply-accumulate operation contributes 1 to the count.

               flop_count_dp_mul:  Number of double-precision floating-point multiply operations executed by non-predicated threads.

                   flop_count_sp:  Number of single-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count. The count does not include special operations.

               flop_count_sp_add:  Number of single-precision floating-point add operations executed by non-predicated threads.

               flop_count_sp_fma:  Number of single-precision floating-point multiply-accumulate operations executed by non-predicated threads. Each multiply-accumulate operation contributes 1 to the count.

               flop_count_sp_mul:  Number of single-precision floating-point multiply operations executed by non-predicated threads.

           flop_count_sp_special:  Number of single-precision floating-point special operations executed by non-predicated threads.

                   inst_executed:  The number of instructions executed

                     inst_issued:  The number of instructions issued

              sysmem_utilization:  The utilization level of the system memory relative to the peak utilization on a scale of 0 to 10

                stall_inst_fetch:  Percentage of stalls occurring because the next assembly instruction has not yet been fetched

           stall_exec_dependency:  Percentage of stalls occurring because an input required by the instruction is not yet available

         stall_memory_dependency:  Percentage of stalls occurring because a memory operation cannot be performed due to the required resources not being available or fully utilized, or because too many requests of a given type are outstanding

                   stall_texture:  Percentage of stalls occurring because the texture sub-system is fully utilized or has too many outstanding requests

                      stall_sync:  Percentage of stalls occurring because the warp is blocked at a __syncthreads() call

                     stall_other:  Percentage of stalls occurring due to miscellaneous reasons

stall_constant_memory_dependency:  Percentage of stalls occurring because of immediate constant cache miss

                 stall_pipe_busy:  Percentage of stalls occurring because a compute operation cannot be performed because the compute pipeline is busy

               shared_efficiency:  Ratio of requested shared memory throughput to required shared memory throughput expressed as percentage

                      inst_fp_32:  Number of single-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)

                      inst_fp_64:  Number of double-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)

                    inst_integer:  Number of integer instructions executed by non-predicated threads

                inst_bit_convert:  Number of bit-conversion instructions executed by non-predicated threads

                    inst_control:  Number of control-flow instructions executed by non-predicated threads (jump, branch, etc.)

              inst_compute_ld_st:  Number of compute load/store instructions executed by non-predicated threads

                       inst_misc:  Number of miscellaneous instructions executed by non-predicated threads

 inst_inter_thread_communication:  Number of inter-thread communication instructions executed by non-predicated threads

                     issue_slots:  The number of issue slots used

                       cf_issued:  Number of issued control-flow instructions

                     cf_executed:  Number of executed control-flow instructions

                     ldst_issued:  Number of issued local, global, shared and texture memory load and store instructions

                   ldst_executed:  Number of executed local, global, shared and texture memory load and store instructions

             atomic_transactions:  Global memory atomic and reduction transactions

 atomic_transactions_per_request:  Average number of global memory atomic and reduction transactions performed for each atomic and reduction instruction

            l2_atomic_throughput:  Memory read throughput seen at L2 cache for atomic and reduction requests

          l2_atomic_transactions:  Memory read transactions seen at L2 cache for atomic and reduction requests

        l2_tex_read_transactions:  Memory read transactions seen at L2 cache for read requests from the texture cache

           stall_memory_throttle:  Percentage of stalls occurring because of memory throttle

              stall_not_selected:  Percentage of stalls occurring because warp was not selected

       l2_tex_write_transactions:  Memory write transactions seen at L2 cache for write requests from the texture cache

                   flop_count_hp:  Number of half-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count.

               flop_count_hp_add:  Number of half-precision floating-point add operations executed by non-predicated threads.

               flop_count_hp_mul:  Number of half-precision floating-point multiply operations executed by non-predicated threads.

               flop_count_hp_fma:  Number of half-precision floating-point multiply-accumulate operations executed by non-predicated threads. Each multiply-accumulate operation 
contributes 1 to the count.

                      inst_fp_16:  Number of half-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)

         sysmem_read_utilization:  The read utilization level of the system memory relative to the peak utilization on a scale of 0 to 10

        sysmem_write_utilization:  The write utilization level of the system memory relative to the peak utilization on a scale of 0 to 10

     pcie_total_data_transmitted:  Total data bytes transmitted through PCIe

        pcie_total_data_received:  Total data bytes received through PCIe

      inst_executed_global_loads:  Warp level instructions for global loads

       inst_executed_local_loads:  Warp level instructions for local loads

      inst_executed_shared_loads:  Warp level instructions for shared loads

     inst_executed_surface_loads:  Warp level instructions for surface loads

     inst_executed_global_stores:  Warp level instructions for global stores

      inst_executed_local_stores:  Warp level instructions for local stores

     inst_executed_shared_stores:  Warp level instructions for shared stores

    inst_executed_surface_stores:  Warp level instructions for surface stores

    inst_executed_global_atomics:  Warp level instructions for global atom and atom cas

 inst_executed_global_reductions:  Warp level instructions for global reductions

   inst_executed_surface_atomics:  Warp level instructions for surface atom and atom cas

inst_executed_surface_reductions:  Warp level instructions for surface reductions

    inst_executed_shared_atomics:  Warp level shared instructions for atom and atom CAS

           inst_executed_tex_ops:  Warp level instructions for texture

            l2_global_load_bytes:  Bytes read from L2 for misses in Unified Cache for global loads

             l2_local_load_bytes:  Bytes read from L2 for misses in Unified Cache for local loads

           l2_surface_load_bytes:  Bytes read from L2 for misses in Unified Cache for surface loads

     l2_local_global_store_bytes:  Bytes written to L2 from Unified Cache for local and global stores. This does not include global atomics.

       l2_global_reduction_bytes:  Bytes written to L2 from Unified cache for global reductions

    l2_global_atomic_store_bytes:  Bytes written to L2 from Unified cache for global atomics (ATOM and ATOM CAS)

          l2_surface_store_bytes:  Bytes written to L2 from Unified Cache for surface stores. This does not include surface atomics.

      l2_surface_reduction_bytes:  Bytes written to L2 from Unified Cache for surface reductions

   l2_surface_atomic_store_bytes:  Bytes transferred between Unified Cache and L2 for surface atomics (ATOM and ATOM CAS)

            global_load_requests:  Total number of global load requests from Multiprocessor

             local_load_requests:  Total number of local load requests from Multiprocessor

           surface_load_requests:  Total number of surface load requests from Multiprocessor

           global_store_requests:  Total number of global store requests from Multiprocessor. This does not include atomic requests.

            local_store_requests:  Total number of local store requests from Multiprocessor

          surface_store_requests:  Total number of surface store requests from Multiprocessor

          global_atomic_requests:  Total number of global atomic(Atom and Atom CAS) requests from Multiprocessor

       global_reduction_requests:  Total number of global reduction requests from Multiprocessor

         surface_atomic_requests:  Total number of surface atomic(Atom and Atom CAS) requests from Multiprocessor

      surface_reduction_requests:  Total number of surface reduction requests from Multiprocessor

               sysmem_read_bytes:  Number of bytes read from system memory

              sysmem_write_bytes:  Number of bytes written to system memory

                 l2_tex_hit_rate:  Hit rate at L2 cache for all requests from texture cache

           texture_load_requests:  Total number of texture Load requests from Multiprocessor

           unique_warps_launched:  Number of warps launched. Value is unaffected by compute preemption.

                   sm_efficiency:  The percentage of time at least one warp is active on a specific multiprocessor

              achieved_occupancy:  Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor

                             ipc:  Instructions executed per cycle

                      issued_ipc:  Instructions issued per cycle

          issue_slot_utilization:  Percentage of issue slots that issued at least one instruction, averaged across all cycles

        eligible_warps_per_cycle:  Average number of warps that are eligible to issue per active cycle

                 tex_utilization:  The utilization level of the unified cache relative to the peak utilization on a scale of 0 to 10

                  l2_utilization:  The utilization level of the L2 cache relative to the peak utilization on a scale of 0 to 10

              shared_utilization:  The utilization level of the shared memory relative to peak utilization on a scale of 0 to 10

             ldst_fu_utilization:  The utilization level of the multiprocessor function units that execute shared load, shared store and constant load instructions on a scale of 0 to 10

               cf_fu_utilization:  The utilization level of the multiprocessor function units that execute control-flow instructions on a scale of 0 to 10

          special_fu_utilization:  The utilization level of the multiprocessor function units that execute sin, cos, ex2, popc, flo, and similar instructions on a scale of 0 to 
10

              tex_fu_utilization:  The utilization level of the multiprocessor function units that execute global, local and texture memory instructions on a scale of 0 to 10   

 single_precision_fu_utilization:  The utilization level of the multiprocessor function units that execute single-precision floating-point instructions and integer instructions 
on a scale of 0 to 10

 double_precision_fu_utilization:  The utilization level of the multiprocessor function units that execute double-precision floating-point instructions on a scale of 0 to 10    

              flop_hp_efficiency:  Ratio of achieved to peak half-precision floating-point operations

              flop_sp_efficiency:  Ratio of achieved to peak single-precision floating-point operations

              flop_dp_efficiency:  Ratio of achieved to peak double-precision floating-point operations

          dram_read_transactions:  Device memory read transactions

         dram_write_transactions:  Device memory write transactions

            dram_read_throughput:  Device memory read throughput

           dram_write_throughput:  Device memory write throughput

                dram_utilization:  The utilization level of the device memory relative to the peak utilization on a scale of 0 to 10

   half_precision_fu_utilization:  The utilization level of the multiprocessor function units that execute 16 bit floating-point instructions on a scale of 0 to 10

                ecc_transactions:  Number of ECC transactions between L2 and DRAM

                  ecc_throughput:  ECC throughput from L2 to DRAM

                 dram_read_bytes:  Total bytes read from DRAM to L2 cache

                dram_write_bytes:  Total bytes written from L2 cache to DRAM
```