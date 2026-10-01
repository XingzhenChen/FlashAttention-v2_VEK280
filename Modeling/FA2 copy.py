"""Starter analytical model for FlashAttention-2 on VEK280.

MM1 and MM2 run on AIE; Softmax runs on PL.
"""

from dataclasses import dataclass, replace
from typing import Optional
from numpy import inf
import math
import itertools


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
    head_dim: int = 256
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

    return total_time


def softmax_latency_ms(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
) -> float:
    """Softmax on PL: P = softmax(S)."""

    n_row = searchPara.T_row
    n_col = searchPara.T_col
    inStream_portwidth_bit = math.ceil(searchPara.T_col / (searchPara.mm1_aie_n * platformSpec.aie_mm_n)) \
            * searchPara.mm1_aie_n * platformSpec.aie_plio_port * 4 # cyclic=2, and t2p can achieve 2 read access.
    outStream_portwidth_bit = math.ceil(searchPara.T_col / (searchPara.mm2_aie_k * platformSpec.aie_mm_k)) \
            * searchPara.mm2_aie_k * platformSpec.aie_plio_port * 4 # cyclic=2, and t2p can achieve 2 read access.

    def load() -> float:
        """From PL buffer to softmax line-buffer."""
        n_col_bit = n_col * modelPara.data_type * 8
        line_cycle = math.ceil(n_col_bit / inStream_portwidth_bit)
        time = line_cycle / (platformSpec.pl_freq_MHz * 1e6) * 1e3

        return time

    def compute() -> float:
        """Softmax scanning time."""
        line_cycle = math.log2(n_col) # Full add tree.
        line_cycle += math.ceil(n_col / searchPara.softmax_unroll) # Sub + exp
        line_cycle += math.log2(n_col) # Sum.
        line_cycle += math.ceil(n_col / searchPara.softmax_unroll) # Div.
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

    return total_time


def mm2_latency_ms(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
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


def check_pl_ram(
    platformSpec: PlatformSpec,
    modelPara: ModelPara,
    searchPara: SearchPara,
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
                return True

        return False

    n_Q = cal_bram(searchPara.mm1_aie_m * searchPara.mm1_aie_k, platformSpec.dram_portwidth_bit / platformSpec.aie_plio_port / 2, searchPara.T_row * modelPara.head_dim),\
          cal_uram(searchPara.mm1_aie_m * searchPara.mm1_aie_k, platformSpec.dram_portwidth_bit / platformSpec.aie_plio_port / 2, searchPara.T_row * modelPara.head_dim)
    n_K = cal_bram(searchPara.mm1_aie_k * searchPara.mm1_aie_n, platformSpec.dram_portwidth_bit / platformSpec.aie_plio_port / 2, modelPara.head_dim * searchPara.T_col), \
        cal_uram(searchPara.mm1_aie_k * searchPara.mm1_aie_n, platformSpec.dram_portwidth_bit / platformSpec.aie_plio_port / 2, modelPara.head_dim * searchPara.T_col)
    n_V = cal_bram(searchPara.mm2_aie_k * searchPara.mm2_aie_n, platformSpec.dram_portwidth_bit / platformSpec.aie_plio_port / 2, modelPara.head_dim * searchPara.T_col), \
        cal_uram(searchPara.mm2_aie_k * searchPara.mm2_aie_n, platformSpec.dram_portwidth_bit / platformSpec.aie_plio_port / 2, modelPara.head_dim * searchPara.T_col)
    n_S = cal_bram(searchPara.mm1_aie_m * searchPara.mm1_aie_n, platformSpec.dram_portwidth_bit / platformSpec.aie_plio_port / 2, searchPara.T_row * searchPara.T_col), \
        cal_uram(searchPara.mm1_aie_m * searchPara.mm1_aie_n, platformSpec.dram_portwidth_bit / platformSpec.aie_plio_port / 2, searchPara.T_row * searchPara.T_col)
    n_P = cal_bram(searchPara.mm2_aie_m * searchPara.mm2_aie_k, platformSpec.dram_portwidth_bit / platformSpec.aie_plio_port / 2, searchPara.T_row * searchPara.T_col), \
        cal_uram(searchPara.mm2_aie_m * searchPara.mm2_aie_k, platformSpec.dram_portwidth_bit / platformSpec.aie_plio_port / 2, searchPara.T_row * searchPara.T_col)
    n_O = cal_bram(searchPara.mm2_aie_m * searchPara.mm2_aie_n, platformSpec.dram_portwidth_bit / platformSpec.aie_plio_port / 2, modelPara.head_dim * searchPara.T_row), \
        cal_uram(searchPara.mm2_aie_m * searchPara.mm2_aie_n, platformSpec.dram_portwidth_bit / platformSpec.aie_plio_port / 2, modelPara.head_dim * searchPara.T_row)

    flag = check_availability(n_Q, n_K, n_V, n_S, n_P, n_O)

    return flag

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

def main() -> None:

    platformSpec = PlatformSpec()
    modelPara = ModelPara()
    searchPara = SearchPara()

    best_time = inf
    best_para = None

    for searchPara.T_row in (2**n for n in range(6, ModelPara.seq_len.bit_length())):
        for searchPara.T_col in (2**n for n in range(6, ModelPara.seq_len.bit_length())):
            # Search tile size from 64 to seq_len.
            for searchPara.mm1_aie_m in range(1, 9):
                for searchPara.mm1_aie_k in range(1, 9):
                    for searchPara.mm1_aie_n in range(1, 9):
                        """Check AIE number"""
                        if not check_mm1_aie_number(platformSpec, modelPara, searchPara):
                            continue
                        for searchPara.mm2_aie_m in range(1, 9):
                            for searchPara.mm2_aie_k in range(1, 9):
                                for searchPara.mm2_aie_n in range(1, 9):
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

    print(f"best latency: {best_time:.6f} ms")
    print(f"best para   : {best_para}")


    


if __name__ == "__main__":
    main()
