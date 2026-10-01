"""Starter analytical model for FlashAttention-2 on VEK280.

MM1 and MM2 run on AIE; Softmax runs on PL.
"""

from dataclasses import dataclass, replace
from typing import Optional
from math import inf
import math
import itertools
from multiprocessing import Pool


@dataclass(frozen=True)
class PlatformSpec:
    aie_num: int = 38*8
    aie_memtile_num: int = 38*2
    aie_shimtile_num: int = 28
    aie_mm_m: int = 64
    aie_mm_k: int = 64
    aie_mm_n: int = 64
    aie_mac_percycle: int = 128
    aie_plio_port: int = 128
    aie_freq_MHz: int = 1000
    aie_out_plio: int = 84

    pl_bram_num: int = 600
    pl_uram_num: int = 264
    pl_dsp_num: int = 1312
    pl_freq_MHz: int = 300

    dram_bw_GBpers: float = 60
    dram_portwidth_bit: int = 512

@dataclass(frozen=True)
class ModelPara:
    seq_len: int = 32768
    head_dim: int = 64
    data_type: int = 2 #BF16

@dataclass(frozen=False)
class SearchPara:
    T_row: int = 0
    T_col: int = 0
    mm1_aie_m: int = 0
    mm1_aie_k: int = 0
    mm1_aie_n: int = 0
    mm2_aie_m: int = 0
    mm2_aie_k: int = 0
    mm2_aie_n: int = 0
    softmax_unroll: int = 0
    packetswitch_factor_outplio: int = 0



def ddr_to_pl_Q_latency_ms(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
) -> float:
    """DDR -> PL: load Q tiles."""

    data_volume = searchPara.T_row * modelPara.head_dim * modelPara.data_type
    time = data_volume / (platformSpec.dram_bw_GBpers * 1e6)
    
    return time

def ddr_to_pl_KV_latency_ms(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
) -> float:
    """DDR -> PL: load K/V tiles."""

    data_volume = (modelPara.head_dim * searchPara.T_col + searchPara.T_col * modelPara.head_dim) * modelPara.data_type
    time = data_volume / (platformSpec.dram_bw_GBpers * 1e6)
    
    return time


def mm1_latency_ms(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
    detail: Optional[dict] = None,
) -> float:
    """MM1 on AIE: S = Q @ K^T."""
    
    mm_arraytile_m = searchPara.mm1_aie_m * platformSpec.aie_mm_m
    mm_arraytile_k = searchPara.mm1_aie_k * platformSpec.aie_mm_k
    mm_arraytile_n = searchPara.mm1_aie_n * platformSpec.aie_mm_n
    n_m = math.ceil(searchPara.T_row / mm_arraytile_m)
    n_k = math.ceil(modelPara.head_dim / mm_arraytile_k)
    n_n = math.ceil(searchPara.T_col / mm_arraytile_n)
    n_iter = n_m * n_k * n_n

    def load() -> float:
        """PL -> AIE: move Q/K tiles into AIE."""
        data_volume = platformSpec.aie_mm_m * platformSpec.aie_mm_k * modelPara.data_type
        plio_bw = platformSpec.aie_plio_port / 8 
        pl_cycle = data_volume / plio_bw
        time = pl_cycle / (platformSpec.pl_freq_MHz * 1e6) * 1e3
        return time

    def compute() -> float:
        """AIE: compute Q @ K^T."""
        """Just roughly estimate every used AIE is running in steady stage."""
        mac = platformSpec.aie_mm_m * platformSpec.aie_mm_k * platformSpec.aie_mm_n
        aie_parallel = platformSpec.aie_mac_percycle
        aie_cycle = mac / aie_parallel
        time = aie_cycle / platformSpec.aie_freq_MHz * 1e-3

        return time

    def store() -> float:
        """AIE -> PL: move S tile out of AIE."""
        data_volume = platformSpec.aie_mm_m * platformSpec.aie_mm_n * modelPara.data_type
        plio_bw = platformSpec.aie_plio_port / 8 
        pl_cycle = data_volume / plio_bw
        time = pl_cycle / (platformSpec.pl_freq_MHz * 1e6) * 1e3
        time = time * searchPara.packetswitch_factor_outplio

        return time

    # Total busy time of each resource; the bottleneck one dominates
    t_load, t_compute, t_store = load(), compute(), store()

    busy_load    = n_iter * t_load
    busy_compute = n_iter * t_compute
    busy_store   = n_m * n_n * t_store   # output written once after K accumulation

    total_time = max(busy_load, busy_compute, busy_store) + (t_load + t_compute + t_store)

    if detail is not None:
        # sub-stage -> (per-unit ms, units per tile)
        detail.update(load=(t_load, n_iter), compute=(t_compute, n_iter), store=(t_store, n_m * n_n))

    return total_time


def softmax_latency_ms(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
    detail: Optional[dict] = None,
) -> float:
    """Softmax on PL: P = softmax(S)."""

    n_row = searchPara.T_row
    n_col = searchPara.T_col
    inStream_portwidth_bit = searchPara.mm1_aie_n * platformSpec.aie_plio_port * 2 # cyclic=1, and t2p can achieve 2 read access.
    outStream_portwidth_bit = searchPara.mm2_aie_k * platformSpec.aie_plio_port * 2 # cyclic=1, and t2p can achieve 2 read access.

    def load() -> float:
        """From PL buffer to softmax line-buffer."""
        n_col_bit = n_col * modelPara.data_type * 8
        line_cycle = math.ceil(n_col_bit / inStream_portwidth_bit)
        time = line_cycle / (platformSpec.pl_freq_MHz * 1e6) * 1e3

        return time

    def compute() -> float:
        """Softmax scanning time."""
        line_cycle = math.log2(n_col) # Full add tree.
        # line_cycle += math.ceil(n_col / searchPara.softmax_unroll) # Sub + exp
        # line_cycle += math.log2(n_col) # Sum.
        # line_cycle += math.ceil(n_col / searchPara.softmax_unroll) # Div.
        time = line_cycle / (platformSpec.pl_freq_MHz * 1e6) * 1e3
    
        return time

    def store() -> float:
        """From softmax line-buffer to PL buffer."""
        n_col_bit = n_col * modelPara.data_type * 8
        line_cycle = math.ceil(n_col_bit / outStream_portwidth_bit)
        time = line_cycle / (platformSpec.pl_freq_MHz * 1e6) * 1e3
    
        return time

    t_load, t_compute, t_store = load(), compute(), store()
    total_time = max(t_load, t_compute, t_store) * n_row + (t_load + t_compute + t_store)

    if detail is not None:
        # sub-stage -> (per-line ms, lines per tile)
        detail.update(load=(t_load, n_row), compute=(t_compute, n_row), store=(t_store, n_row))

    return total_time


def mm2_latency_ms(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
    detail: Optional[dict] = None,
) -> float:
    """MM2 on AIE: O = P @ V."""

    mm_arraytile_m = searchPara.mm2_aie_m * platformSpec.aie_mm_m
    mm_arraytile_k = searchPara.mm2_aie_k * platformSpec.aie_mm_k
    mm_arraytile_n = searchPara.mm2_aie_n * platformSpec.aie_mm_n
    n_m = math.ceil(searchPara.T_row / mm_arraytile_m)
    n_k = math.ceil(searchPara.T_col / mm_arraytile_k)
    n_n = math.ceil(modelPara.head_dim / mm_arraytile_n)
    n_iter = n_m * n_k * n_n

    def load() -> float:
        """PL -> AIE: move Q/K tiles into AIE."""
        data_volume = platformSpec.aie_mm_m * platformSpec.aie_mm_k * modelPara.data_type
        plio_bw = platformSpec.aie_plio_port / 8 
        pl_cycle = data_volume / plio_bw
        time = pl_cycle / (platformSpec.pl_freq_MHz * 1e6) * 1e3
        return time

    def compute() -> float:
        """AIE: compute Q @ K^T."""
        """Just roughly estimate every used AIE is running in steady stage."""
        mac = platformSpec.aie_mm_m * platformSpec.aie_mm_k * platformSpec.aie_mm_n
        aie_parallel = platformSpec.aie_mac_percycle
        aie_cycle = mac / aie_parallel
        time = aie_cycle / platformSpec.aie_freq_MHz * 1e-3

        return time

    def store() -> float:
        """AIE -> PL: move S tile out of AIE."""
        data_volume = platformSpec.aie_mm_m * platformSpec.aie_mm_n * modelPara.data_type
        plio_bw = platformSpec.aie_plio_port / 8 
        pl_cycle = data_volume / plio_bw
        time = pl_cycle / (platformSpec.pl_freq_MHz * 1e6) * 1e3
        time = time * searchPara.packetswitch_factor_outplio

        return time

    # Total busy time of each resource; the bottleneck one dominates
    t_load, t_compute, t_store = load(), compute(), store()

    busy_load    = n_iter * t_load
    busy_compute = n_iter * t_compute
    busy_store   = n_m * n_n * t_store   # output written once after K accumulation

    total_time = max(busy_load, busy_compute, busy_store) + (t_load + t_compute + t_store)

    if detail is not None:
        # sub-stage -> (per-unit ms, units per tile)
        detail.update(load=(t_load, n_iter), compute=(t_compute, n_iter), store=(t_store, n_m * n_n))

    return total_time


def pl_to_ddr_latency_ms(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
) -> float:
    """PL -> DDR: write back O tile."""
    data_volume = searchPara.T_row * modelPara.head_dim * modelPara.data_type
    time = data_volume / (platformSpec.dram_bw_GBpers * 1e6)
    
    return time


def end2end_latency_ms(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
) -> float:
    t_q_load  = ddr_to_pl_Q_latency_ms(platformSpec, modelPara, searchPara)
    t_o_store = pl_to_ddr_latency_ms(platformSpec, modelPara, searchPara)

    # Stages repeated for every (row tile, col tile) pair
    inner_stage_ms = [
        ddr_to_pl_KV_latency_ms(platformSpec, modelPara, searchPara),
        mm1_latency_ms(platformSpec, modelPara, searchPara),
        softmax_latency_ms(platformSpec, modelPara, searchPara),
        mm2_latency_ms(platformSpec, modelPara, searchPara),
    ]

    n_row = math.ceil(modelPara.seq_len / searchPara.T_row)
    n_col = math.ceil(modelPara.seq_len / searchPara.T_col)

    # Inner loop over col tiles: pipeline fill + (n_col - 1) bottleneck intervals
    t_inner = sum(inner_stage_ms) + (n_col - 1) * max(inner_stage_ms)

    # Each row tile: Q load -> inner pipeline -> O store, row tiles run back to back
    time = n_row * (t_q_load + t_inner + t_o_store)

    return time


def throughput_GFLOPS(
    modelPara: ModelPara,
    time_ms: float,
) -> float:
    """Effective throughput, counting only MM1 (Q @ K^T) and MM2 (P @ V) MACs; 1 MAC = 2 FLOPs."""
    mm1_mac = modelPara.seq_len * modelPara.seq_len * modelPara.head_dim
    mm2_mac = modelPara.seq_len * modelPara.seq_len * modelPara.head_dim
    flop = 2 * (mm1_mac + mm2_mac)

    return flop / (time_ms * 1e-3) / 1e9


def print_stage_breakdown(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
) -> None:
    """Per-stage latency of one config, mirroring end2end_latency_ms, to expose the bottleneck."""
    n_row = math.ceil(modelPara.seq_len / searchPara.T_row)
    n_col = math.ceil(modelPara.seq_len / searchPara.T_col)

    mm1_detail, softmax_detail, mm2_detail = {}, {}, {}

    # (name, per-tile ms, times executed, sub-stage detail)
    stages = [
        ("Q load (DDR->PL)",  ddr_to_pl_Q_latency_ms(platformSpec, modelPara, searchPara),                  n_row,         None),
        ("KV load (DDR->PL)", ddr_to_pl_KV_latency_ms(platformSpec, modelPara, searchPara),                 n_row * n_col, None),
        ("MM1 (AIE)",         mm1_latency_ms(platformSpec, modelPara, searchPara, mm1_detail),              n_row * n_col, mm1_detail),
        ("Softmax (PL)",      softmax_latency_ms(platformSpec, modelPara, searchPara, softmax_detail),      n_row * n_col, softmax_detail),
        ("MM2 (AIE)",         mm2_latency_ms(platformSpec, modelPara, searchPara, mm2_detail),              n_row * n_col, mm2_detail),
        ("O store (PL->DDR)", pl_to_ddr_latency_ms(platformSpec, modelPara, searchPara),                    n_row,         None),
    ]
    total = end2end_latency_ms(platformSpec, modelPara, searchPara)
    bottleneck = max(stages[1:5], key=lambda s: s[1])[0]

    print(f"\nn_row = {n_row}, n_col = {n_col}, end2end = {total:.6f} ms, "
          f"throughput = {throughput_GFLOPS(modelPara, total):.2f} GFLOPS")
    print("share = busy / end2end for stages, busy / stage per-tile time for sub-stages")
    print(f"{'stage':<20}{'per unit (ms)':>16}{'count':>10}{'busy (ms)':>14}{'share':>10}")
    for name, t, cnt, detail in stages:
        busy = t * cnt
        mark = "  <-- bottleneck" if name == bottleneck else ""
        print(f"{name:<20}{t:>16.6f}{cnt:>10}{busy:>14.4f}{busy / total:>10.1%}{mark}")

        if detail:
            sub_bottleneck = max(detail, key=lambda k: detail[k][0] * detail[k][1])
            for sub, (t_sub, cnt_sub) in detail.items():
                busy_sub = t_sub * cnt_sub
                mark = "  <-- stage bottleneck" if sub == sub_bottleneck else ""
                print(f"  - {sub:<16}{t_sub:>16.8f}{cnt_sub:>10}{busy_sub:>14.6f}{busy_sub / t:>10.1%}{mark}")


def check_pl_ram(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
    detail: Optional[dict] = None,
) -> bool:
    flag = True

    def cal_bram(n_block, cyclic_factor, n_data) -> int:
        """RAM36K, with T2P (36 bit * 1024 depth)"""
        n_ele = n_data / (platformSpec.aie_plio_port / (modelPara.data_type * 8))
        num = n_block * cyclic_factor * math.ceil(platformSpec.aie_plio_port / 36) * math.ceil((n_ele / (n_block * cyclic_factor)) / 1024)
        return num

    def cal_uram(n_block, cyclic_factor, n_data) -> int:
        """RAM288K, with T2P (72 bit * 4096 depth)"""
        n_ele = n_data / (platformSpec.aie_plio_port / (modelPara.data_type * 8))
        num = n_block * cyclic_factor * math.ceil(platformSpec.aie_plio_port / 72) * math.ceil((n_ele / (n_block * cyclic_factor)) / 4096)
        return num

    def check_availability(n_Q, n_K, n_V, n_S, n_P, n_O) -> bool:
        """Each buffer is (n_bram, n_uram); try every BRAM/URAM assignment of the six buffers."""
        buffers = (n_Q, n_K, n_V, n_S, n_P, n_O)

        # Copies of each buffer: K/V/S/P ping-pong across col tiles in the inner pipeline;
        # Q (reused over the whole row tile) and O (accumulated over it) are single,
        # since Q load / O store are serialized in end2end_latency_ms.
        n_copy = (1, 2, 2, 2, 2, 1)

        # choice[i] = 0 -> buffer i in BRAM, 1 -> buffer i in URAM
        for choice in itertools.product((0, 1), repeat=len(buffers)):
            used_bram = sum(buf[0] * k for buf, k, c in zip(buffers, n_copy, choice) if c == 0)
            used_uram = sum(buf[1] * k for buf, k, c in zip(buffers, n_copy, choice) if c == 1)
            if used_bram <= platformSpec.pl_bram_num and used_uram <= platformSpec.pl_uram_num:
                if detail is not None:
                    # buffer -> (n_bram, n_uram, copies, placed in URAM) of the first feasible assignment
                    detail.update(zip(("Q", "K", "V", "S", "P", "O"),
                                      ((buf[0], buf[1], k, c == 1) for buf, k, c in zip(buffers, n_copy, choice))))
                return True

        return False

    # Banks per AIE stream for DDR-facing buffers: one 512-bit DDR beat = 4 x 128-bit words, T2P writes 2 per bank
    ddr_cyclic = platformSpec.dram_portwidth_bit // platformSpec.aie_plio_port // 2

    n_Q = cal_bram(searchPara.mm1_aie_m * searchPara.mm1_aie_k, ddr_cyclic, searchPara.T_row * modelPara.head_dim),\
          cal_uram(searchPara.mm1_aie_m * searchPara.mm1_aie_k, ddr_cyclic, searchPara.T_row * modelPara.head_dim)
    n_K = cal_bram(searchPara.mm1_aie_k * searchPara.mm1_aie_n, ddr_cyclic, modelPara.head_dim * searchPara.T_col), \
        cal_uram(searchPara.mm1_aie_k * searchPara.mm1_aie_n, ddr_cyclic, modelPara.head_dim * searchPara.T_col)
    n_V = cal_bram(searchPara.mm2_aie_k * searchPara.mm2_aie_n, ddr_cyclic, modelPara.head_dim * searchPara.T_col), \
        cal_uram(searchPara.mm2_aie_k * searchPara.mm2_aie_n, ddr_cyclic, modelPara.head_dim * searchPara.T_col)
    n_S = cal_bram(searchPara.mm1_aie_m * searchPara.mm1_aie_n, 1, searchPara.T_row * searchPara.T_col), \
        cal_uram(searchPara.mm1_aie_m * searchPara.mm1_aie_n, 1, searchPara.T_row * searchPara.T_col)
    n_P = cal_bram(searchPara.mm2_aie_m * searchPara.mm2_aie_k, 1, searchPara.T_row * searchPara.T_col), \
        cal_uram(searchPara.mm2_aie_m * searchPara.mm2_aie_k, 1, searchPara.T_row * searchPara.T_col)
    n_O = cal_bram(searchPara.mm2_aie_m * searchPara.mm2_aie_n, ddr_cyclic, modelPara.head_dim * searchPara.T_row), \
        cal_uram(searchPara.mm2_aie_m * searchPara.mm2_aie_n, ddr_cyclic, modelPara.head_dim * searchPara.T_row)

    flag = check_availability(n_Q, n_K, n_V, n_S, n_P, n_O)

    return flag


def print_ram_usage(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
) -> None:
    """BRAM/URAM usage per buffer for the assignment check_pl_ram accepts."""
    detail = {}
    if not check_pl_ram(platformSpec, modelPara, searchPara, detail):
        print("\nPL RAM: no feasible BRAM/URAM assignment")
        return

    print("\nPL RAM usage of best config (first feasible BRAM/URAM placement)")
    print(f"{'buffer':<8}{'BRAM/copy':>11}{'URAM/copy':>11}{'copies':>8}{'placed':>8}{'BRAM':>8}{'URAM':>8}")
    total_bram = total_uram = 0
    for name, (n_bram, n_uram, copies, in_uram) in detail.items():
        used_bram = 0 if in_uram else n_bram * copies
        used_uram = n_uram * copies if in_uram else 0
        total_bram += used_bram
        total_uram += used_uram
        print(f"{name:<8}{n_bram:>11}{n_uram:>11}{copies:>8}{'URAM' if in_uram else 'BRAM':>8}{used_bram:>8}{used_uram:>8}")
    print(f"{'total':<46}{total_bram:>8}{total_uram:>8}")
    print(f"{'available':<46}{platformSpec.pl_bram_num:>8}{platformSpec.pl_uram_num:>8}")
    print(f"{'utilization':<46}{total_bram / platformSpec.pl_bram_num:>8.1%}{total_uram / platformSpec.pl_uram_num:>8.1%}")


def check_mm1_plio(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
) -> bool:
    """Only check mm1 out PLIO now."""
    num = math.ceil(searchPara.mm1_aie_m * searchPara.mm1_aie_n / searchPara.packetswitch_factor_outplio)
    flag = num <= platformSpec.aie_out_plio

    return flag

def check_plio(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
) -> bool:
    """Check both mm1 and mm2 out PLIO."""
    num1 = math.ceil(searchPara.mm1_aie_m * searchPara.mm1_aie_n / searchPara.packetswitch_factor_outplio)
    num2 = math.ceil(searchPara.mm2_aie_m * searchPara.mm2_aie_n / searchPara.packetswitch_factor_outplio)

    # MM1 and MM2 are resident on AIE at the same time, so they share the out-PLIO pool
    flag = num1 + num2 <= platformSpec.aie_out_plio

    return flag

def check_mm1_aie_number(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
) -> bool:
    num = searchPara.mm1_aie_m * searchPara.mm1_aie_k * searchPara.mm1_aie_n
    flag = num <= platformSpec.aie_num

    return flag

def check_mm2_aie_number(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
) -> bool:
    num1 = searchPara.mm1_aie_m * searchPara.mm1_aie_k * searchPara.mm1_aie_n
    num2 = searchPara.mm2_aie_m * searchPara.mm2_aie_k * searchPara.mm2_aie_n
    flag = num1 + num2 <= platformSpec.aie_num

    return flag


def check_dsp(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
) -> bool:
    dsp_per_unroll = 4
    n_dsp = searchPara.softmax_unroll * dsp_per_unroll
    flag = n_dsp <= platformSpec.pl_dsp_num

    return flag

N_PROC = 36
AIE_DIMS = (1, 2, 4, 8, 16, 32, 64)  # candidate AIE array size per mm1/mm2 dimension


def search_tile(tile: tuple) -> tuple:
    """Search all inner parameters for one (T_row, T_col); runs in a worker process."""
    platformSpec = PlatformSpec()
    modelPara = ModelPara()
    searchPara = SearchPara()
    searchPara.T_row, searchPara.T_col = tile

    best_time = inf
    best_para = None

    for searchPara.mm1_aie_m in AIE_DIMS:
        for searchPara.mm1_aie_k in AIE_DIMS:
            for searchPara.mm1_aie_n in AIE_DIMS:
                """Check AIE number"""
                if not check_mm1_aie_number(platformSpec, modelPara, searchPara):
                    continue
                for searchPara.mm2_aie_m in AIE_DIMS:
                    for searchPara.mm2_aie_k in AIE_DIMS:
                        for searchPara.mm2_aie_n in AIE_DIMS:
                            """Check AIE number"""
                            if not check_mm2_aie_number(platformSpec, modelPara, searchPara):
                                continue
                            '''Check PL buffer (needs T_row/T_col and all mm*_aie_* for banking).'''
                            if not check_pl_ram(platformSpec, modelPara, searchPara):
                                continue
                            for searchPara.packetswitch_factor_outplio in range(1, 6):
                                '''Check PLIO (check_plio also covers mm1 alone).'''
                                if not check_plio(platformSpec, modelPara, searchPara):
                                    continue
                                for searchPara.softmax_unroll in (2**n for n in range(4, 8)):
                                    '''Check DSP.'''
                                    if not check_dsp(platformSpec, modelPara, searchPara):
                                        continue

                                    cur_time = end2end_latency_ms(platformSpec, modelPara, searchPara)
                                    if cur_time < best_time:
                                        best_time = cur_time
                                        best_para = replace(searchPara)
                                # Latency is non-decreasing in packetswitch (it only scales MM store time),
                                # so the first PLIO-feasible factor is optimal for this config.
                                break

    return best_time, best_para


def main() -> None:

    # Search tile size from 64 to seq_len.
    tile_sizes = [2**n for n in range(6, ModelPara.seq_len.bit_length())]
    tiles = [(T_row, T_col) for T_row in tile_sizes for T_col in tile_sizes]

    best_time = inf
    best_para = None

    # (T_row, T_col) pairs are independent; cost varies a lot per tile, so hand them out one at a time
    with Pool(N_PROC) as pool:
        for i, (cur_time, cur_para) in enumerate(pool.imap_unordered(search_tile, tiles, chunksize=1), 1):
            if cur_time < best_time:
                best_time = cur_time
                best_para = cur_para
            print(f"[{i}/{len(tiles)}] best so far: {best_time:.6f} ms", flush=True)

    print(f"best latency: {best_time:.6f} ms")
    print(f"throughput  : {throughput_GFLOPS(ModelPara(), best_time):.2f} GFLOPS")
    print(f"best para   : {best_para}")

    if best_para is not None:
        print_stage_breakdown(PlatformSpec(), ModelPara(), best_para)
        print_ram_usage(PlatformSpec(), ModelPara(), best_para)


if __name__ == "__main__":
    main()
