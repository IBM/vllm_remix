#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

import math
import copy
from collections import OrderedDict
from typing import Any


# This function verifies the shape of the chip. It performs the following checks
# 1. The number of clusters per chip can be formed into a square
def verify_chip(clusters_per_chip):
    if clusters_per_chip % 4 != 0:
        print("Number of clusters per chip must be a multiple of 4")
        exit()
    else:
        print("Chip is valid")


# This function calculates the pipeline delay of the chip. It considers all of the clusters
# to be arranged an a square, and implments the H-Tree connecting all clusters to a single
# input/output. It then calculates the number of buffers needed based on pipeline_every_tile,
# which represents how many buffers should be inserted every unit of tile width.
def calculate_pipelining(clusters_per_chip, pipeline_every_tile):
    # Number of clusters in the xy direction
    width_height = math.ceil(math.sqrt(clusters_per_chip) * 4)
    # Data first moves to center of chip, which is one half the width_height
    num_tiles_to_destination = [width_height / 2 / pipeline_every_tile]
    # Data then recursively moves in first x, then y direction, to the destination
    width_height /= 4
    divide = False
    while width_height != 1:
        num_tiles_to_destination.append((width_height / pipeline_every_tile))
        if width_height == 2:  # kludge
            break
        if divide:
            width_height /= 2
        divide = not divide
    return int(sum(num_tiles_to_destination)), num_tiles_to_destination


# This function sanity checks for overlaps between a list of intervals
def has_overlap(intervals):
    for i in range(1, len(intervals)):
        prev_end = intervals[i - 1][1]
        curr_start = intervals[i][0]
        if curr_start < prev_end:
            return True
    return False


# This function explores the mapping space and calculates the timings of each mapping
def map_mvm(
    M,
    N,
    input_on_chip=False,
    keep_output_on_chip=False,
    tile_size=512,
    tile_power=60e-3,
    leaf_size=4,
    leaves_per_cluster=4,
    clusters_per_chip=64,
    mvm_latency=200,
    max_power=300,
    tier_switch_latency=200,
    arrays_per_tile=4,
    pcie_gps=64,
    htree_freq_ghz=1,
    pipeline_every_tile=1,
    specific_map=-1,
):
    # pipeline_delay, pipeline_stages = calculate_pipelining(clusters_per_chip, pipeline_every_tile)
    pipeline_delay = 50
    # How many clusters in the xy directions on chip
    cluster_width_height = math.ceil(math.sqrt(clusters_per_chip))
    # Number of tiles in a chip column or row
    # tiles_per_chip_column_row = cluster_width_height * leaves_per_cluster
    # Total leaves on the chip
    num_leaves = leaves_per_cluster * clusters_per_chip

    chip_timings: OrderedDict[str, Any] = OrderedDict(
        {
            "htree_in": [],
            "htree_to_cluster_leaf": {},
            "tier_switch": [],
            "mvm_leaf": {},
            "reduction_level": {},
            "htree_from_cluster": {},
            "htree_out": [],
        }
    )
    # Build the dictionaries holding the timings for the htree and the leaves.
    for leaf in range(leaves_per_cluster):
        mvm_leaf_tile_dict = {}
        htree_leaf_tile_dict = {}

        for tile in range(leaf_size):
            mvm_leaf_tile_dict[tile] = []
            htree_leaf_tile_dict[tile] = []
        chip_timings["mvm_leaf"][f"{leaf}_tile"] = mvm_leaf_tile_dict
        chip_timings["htree_to_cluster_leaf"][f"{leaf}_tile"] = htree_leaf_tile_dict

    # Build the dictionary to contain timings for the reduction of partial sums
    for reduction_level in range(0, int(math.ceil(math.log2(leaves_per_cluster)))):
        chip_timings["reduction_level"][reduction_level] = []

    for cluster in range(cluster_width_height * cluster_width_height):
        cluster_dict = {}
        for cluster_leaf in range(leaves_per_cluster):
            cluster_dict[cluster_leaf] = []
        chip_timings["htree_from_cluster"][f"{cluster}_leaf"] = cluster_dict

    cluster_size = 0.135 * leaf_size * leaves_per_cluster
    chip_size = (cluster_size + cluster_size * 0.20) * clusters_per_chip
    # PCIe latency for 512 bytes in nanoseconds. If the input is already
    # on-chip, set to 0
    pcie_latency_ns = 512 / (pcie_gps * (1024**3) / 1e9) if not input_on_chip else 0
    # H-Tree latency for 512 bytes in nanoseconds
    htree_latency_ns = 1 / htree_freq_ghz * 8
    # Interconnect latency is the max of these two latencies.
    # Ultimately I will always try to balance these, so basically equal
    interconnect_latency = int(math.ceil(max(pcie_latency_ns, htree_latency_ns)))
    # Number of 512 byte input and output blocks for the MVM
    input_blocks = int(math.ceil(M / tile_size))
    output_blocks = int(math.ceil(N / tile_size))
    num_ops = 2 * 1 * M * N

    # Step 1: Data-in and MVM. This section is staged from step 2
    # data-out

    # Data enters input H-Tree, moves to clusters. If the input_on_chip argument
    # is True, the inputs are chained from the previous MVM. Add pipeline_delay
    # cycles to this MVM's input to simulate the delay before the inputs can start
    # arriving.

    # Data is operated on by leaves in NL mode, where N can be 1, 2, 4, and the
    # number of tiles dedicated to an output is N*4. If there is more inputs than
    # tiles, tiles are reused, meaning input will have to pause until the tile
    # is complete. Tile outputs are shipped and accumulated at the DPU in pipelined
    # fashion.

    # The limit on number of leaves that can be used per output is a) the amount
    # such that all outputs get the same number of leaves b) the number such that
    # tiles are not unnecessarily utilized (don't use 8 tiles if there are only
    # 4 inputs), c) the number of leaves in a cluster
    max_leaves_per_output = min(
        int(num_leaves // output_blocks),
        math.ceil(input_blocks / leaf_size),
        leaves_per_cluster,
    )
    # dumb
    if max_leaves_per_output == 3:
        max_leaves_per_output = 4
    # Track the mapping that achieves the lowest latency
    best_timings = {}
    # Iterate over all possible mappings, dictated by the range of number of leaves
    # dedicated to each output, from 1 to max_leaves_per_output.
    # If the user requests a specific mapping, only use that one
    start_map = 1 if specific_map == -1 else specific_map
    max_leaves_per_output = max_leaves_per_output if specific_map == -1 else specific_map
    best_htree_in_latency = -1
    best_mvm_latency = -1
    best_htree_out_latency = -1
    best_total_latency = -1
    end_reduction_time = -1
    best_map = -1
    best_mvm_latency = -1
    best_mvm_plus_reduction_latency = -1
    for leaves_per_output in range(start_map, max_leaves_per_output + 1):
        if leaves_per_output not in [1, 2, 4]:
            continue
        # For printing timing diagram
        timings = copy.deepcopy(chip_timings)

        # Number of tiles dedicated to each output
        tiles_per_output = leaves_per_output * leaf_size
        # The tier we are currently using and a boolean indicating
        # when we incur tier switch overhead
        current_tier = 0
        tier_switch = False
        current_time = 0

        # TODO: Currently assume tier switch latency can't be hidden
        unmasked_tier_switch_latency = tier_switch_latency

        # If the input is on the chip from the previous MVM, we don't have to bring it
        # in over the PCIe. This will most likely be overall a performance improvement,
        # however, from the standpoint of the chip, this means moving data across the
        # H-Tree from leaves/accumulators to tiles. This incurs a worst case latency of
        # traversing the whole reduction path plus moving back to the start of the H-Tree,
        if not input_on_chip:
            input_pipeline_delay = pipeline_delay
        else:
            input_pipeline_delay = pipeline_delay + pipeline_delay

        # Number of outputs each cluster produces
        outputs_per_cluster = min(leaves_per_cluster // leaves_per_output, output_blocks)
        # Count down the remaining inputs
        current_input = 0
        end_mvm_time = 0
        end_htree_in_time = 0
        while current_input < input_blocks:
            # Iterate over all tiles dedicated to each output, and calculate
            # the timings of the Tile, H-Tree, and PCIe
            for tile in range(tiles_per_output):
                leaf = tile // leaf_size
                leaf_tile = tile % leaf_size
                begin_htree_in_at = current_time
                if tier_switch and tile == 0:
                    begin_htree_in_at += unmasked_tier_switch_latency
                begin_htree_move_at = begin_htree_in_at + interconnect_latency
                begin_tile_at = begin_htree_move_at + input_pipeline_delay
                timings["htree_in"].append(
                    [
                        begin_htree_in_at,
                        begin_htree_in_at + interconnect_latency - 1,
                    ]
                )
                for parallel_leaf in range(
                    0, outputs_per_cluster, leaves_per_cluster // outputs_per_cluster
                ):
                    # Time for data to get to the clusters
                    timings["htree_to_cluster_leaf"][f"{leaf + parallel_leaf}_tile"][
                        leaf_tile
                    ].append(
                        [
                            begin_htree_move_at,
                            begin_htree_move_at + input_pipeline_delay - 1,
                        ]
                    )
                    # End of last data movement time
                    end_htree_in_time = begin_tile_at
                    # MVM takes mvm_latency to run
                    end_mvm_time = begin_tile_at + mvm_latency - 1
                    timings["mvm_leaf"][f"{leaf + parallel_leaf}_tile"][leaf_tile].append(
                        [
                            begin_tile_at,
                            end_mvm_time,
                        ]
                    )
                # If we have to tier switch after this MVM, begin the tier switch such that it
                # will be done as the data arrives to begin the MVM
                if tier_switch:
                    begin_tier_switch_at = begin_tile_at - tier_switch_latency
                    timings["tier_switch"].append(
                        [
                            begin_tier_switch_at,
                            begin_tile_at,
                        ]
                    )
                current_time = timings["htree_in"][-1][0] + interconnect_latency
                # Decrement remaining inputs, if we've sent all inputs, break
                current_input += 1
                if current_input == input_blocks:
                    break
            # We've iterated over all tiles dedicated to the output and are returning to the first
            # tile. Increment the current tier, and, if we are out of free tiers, indicate that
            # we have to incur a tier switch overhead
            current_tier = (current_tier + 1) % arrays_per_tile
            next_current_time = (
                timings["mvm_leaf"]["0_tile"][0][-1][1]
                + 1
                - input_pipeline_delay
                - 1
                - interconnect_latency
            )
            current_time = max(current_time, next_current_time)
            # current_time = min(current_time, )
            if current_tier == 0:
                tier_switch = True
            else:
                tier_switch = False
        # Step 2: Reduction and data out.
        # In 1L mode, there is no reduction, and data immediately starts leaving
        # In 2L mode, data is reduced between 2 leaves, then starts leaving
        # In 4L mode, data is reduced between 2 leaves, then those two leaves,
        # then starts leaving.
        # If the keep_output_on_chip argument is True, the data will be chained into
        # the next input. To simulate this, don't count the leaving time, as it will
        # be pipelined with the input of the next MVM. The delay for the data getting
        # out over the output H-Tree will be added to the next MVM's input delay.

        # Get the time at the end of the last MVM.
        current_time = end_mvm_time
        # Pay for the last Tile Reduction
        current_time += 8
        # We have 0 reduction levels for 1 leaf, 1 for 2 leaves, and 2 for 4 leaves
        reduction_levels = int(math.ceil(math.log2(leaves_per_output)))
        # Iterate over remaining outputs (another closed form solution plz.)
        remaining_outputs = output_blocks
        # Reduction uses the same adders as the affine units, meaning it takes
        # interconnect_latency cycles to perform 512 reductions. Reduction latency is
        # therefore interconnect_latency * 2
        if reduction_levels == 0:
            reduction_latency = 0
        else:
            reduction_latency = 8
        end_out_time = current_time
        end_reduction_time = end_out_time
        for reduction_level in range(reduction_levels):
            timings["reduction_level"][reduction_level].append(
                [current_time, current_time + reduction_latency - 1]
            )
            current_time += reduction_latency
            end_reduction_time = current_time
        # Pay for output activation
        current_time += 8
        end_reduction_time = current_time
        if not keep_output_on_chip:
            # Iterate over the outputs
            for output in range(remaining_outputs):
                will_start_at = current_time + (output % leaves_per_output) * interconnect_latency
                will_take = (output + 1) * interconnect_latency + pipeline_delay
                timings["htree_from_cluster"][
                    f"{(output*leaves_per_output)//leaves_per_cluster}_leaf"
                ][(output * leaves_per_output) % leaves_per_cluster].append(
                    [will_start_at, current_time + will_take - 1]
                )
                end_out_time = current_time + will_take
        # If this mapping resulted in a better time, store it
        total_latency = end_out_time
        htree_in_latency = end_htree_in_time
        mvms_latency = end_mvm_time - (pipeline_delay + interconnect_latency - 1)
        mvms_plus_reduction_latency = end_reduction_time - (
            pipeline_delay + interconnect_latency - 1
        )
        htree_out_latency = end_out_time - end_reduction_time
        if not best_timings or total_latency < best_total_latency:
            best_htree_in_latency = htree_in_latency
            best_mvm_latency = mvms_latency
            best_mvm_plus_reduction_latency = mvms_plus_reduction_latency
            best_htree_out_latency = htree_out_latency
            best_total_latency = total_latency
            best_map = leaves_per_output
            best_timings = timings
    tops = num_ops / (best_total_latency * 1e-9) * 1e-12
    tops_mm2 = tops / chip_size
    mmio_latency = (input_blocks + output_blocks) * pcie_latency_ns + mvm_latency
    mmio_tops = num_ops / (mmio_latency * 1e-9) * 1e-12
    mmio_tops_mm2 = mmio_tops / chip_size
    stats = {
        "htree_in_latency": best_htree_in_latency,
        "mvm_latency": best_mvm_latency,
        "mvm_plus_reduction_latency": best_mvm_plus_reduction_latency,
        "htree_out_latency": best_htree_out_latency,
        "latency": best_total_latency,
        "tops": tops,
        "tops_mm2": tops_mm2,
        "leaves": best_map,
        "mmio_latency": mmio_latency,
        "mmio_tops": mmio_tops,
        "mmio_tops_mm2": mmio_tops_mm2,
    }

    print(
        f"Best map/time for {M}({input_blocks})x{N}({output_blocks}): {best_map} leaves per output, {best_total_latency}"
    )
    return (
        [best_htree_in_latency, best_mvm_latency, best_htree_out_latency, best_total_latency],
        best_timings,
        stats,
    )
    # Jobs done


# This function calculates the time it takes to do a horizontal, inter-cluster reduction
# over the H-Tree. Data is moved from a leaf accumulator to the H-Tree then back into another
# leaf accumulator in the adjacent leaf. This can be done once or twice, and is hardcoded
# for a cluster size of 4 leaves and one buffer per tile width. Reduction could be done
# either as an addition or a multiplication, as needed for the gate+up reduction in llama
# This enables element-wise matrix operations on-chip.
def reduce_clusters(htree_freq_ghz, level=1):
    # Hardcoded to 3, should be a function of cluster size and
    # number of registers per tile
    buffers = 3
    # H-Tree latency for 512 bytes in nanoseconds
    htree_latency_ns = int(math.ceil(1 / htree_freq_ghz * 8))
    # Calculate cluster reduction as the number of times data has to be moved
    # over the H-Tree. Assuming FP16 data is being transfered, so double the
    # htrree_latency
    return (htree_latency_ns * 2 + buffers) * level


# This function adds a value coming from the PCIe to outputs stored in the accumulators.
# Primarily used for residual addition.
def external_add(
    M,
    tile_size=512,
    pcie_gps=64,
    htree_freq_ghz=1,
    addition_latency=1,
    clusters_per_chip=64,
    pipeline_every_tile=1,
):
    pipeline_delay = calculate_pipelining(clusters_per_chip, pipeline_every_tile)
    # PCIe latency for 512 bytes in nanoseconds. If the input is already
    # on-chip, set to 0
    pcie_latency_ns = 512 / (pcie_gps * (1024**3) / 1e9)
    # H-Tree latency for 512 bytes in nanoseconds
    htree_latency_ns = 1 / htree_freq_ghz * 8
    # Interconnect latency is the max of these two latencies.
    # Ultimately I will always try to balance these, so basically equal
    interconnect_latency = int(max(pcie_latency_ns, htree_latency_ns))
    input_blocks = int(math.ceil(M / tile_size))
    return interconnect_latency * input_blocks + pipeline_delay + addition_latency


# This function performs a distributed operation such as approximate max/argmax
# or layernorm. Each leaf gets 2 wires to transmit data to a centralized unit,
# where global operations are performed before (optionally) transmitting
# result data back to the leaves. This is optional because for example the
# max/argmax does not need to return data. If data needs to be transmitted
# off-chip, as in max/argmax, this will need to be modeled, I've not done it
# here.
def distributed_operation(
    return_result_to_leaf=True,
    global_op_latency=20,
    tile_size=512,
    pcie_gps=64,
    htree_freq_ghz=1,
    clusters_per_chip=64,
    pipeline_every_tile=1,
):
    pipeline_delay = calculate_pipelining(clusters_per_chip, pipeline_every_tile)
    # H-Tree latency for 512 bytes in nanoseconds
    htree_latency_ns = 1 / htree_freq_ghz * 8
    data_to_global = pipeline_delay + htree_latency_ns
    data_from_global = pipeline_delay + htree_latency_ns if return_result_to_leaf else 0
    return data_to_global + global_op_latency + data_from_global


def get_attention_time(job):
    batch_size = job.tensorSizes()[0][0].concreteInt()
    match batch_size:
        case 1:
            return 10579
        case 2:
            return 10535
        case 4:
            return 11891
        case 8:
            return 12012  # TODO: A guess
        case 16:
            return 24025
        case _:
            raise NotImplementedError(f"Invalid batch size {batch_size}")


def htree_in_latency(M, L, pipeline_stages):
    in_sticks = M // 512
    # num_tiles = L * 4
    # tiers = math.ceil(in_sticks / (num_tiles))
    data_latency = in_sticks * 8
    # tier_latency = (200 - num_tiles * 8 - 1) * (tiers - 1)
    # tier_switch_latency = 200 * math.floor((tiers - 1) / 4)
    # return data_latency + tier_latency + tier_switch_latency + pipeline_stages
    return data_latency + pipeline_stages


def get_htree_in_latency(interconnect, job):
    tensor_size = [dim.concreteInt() for dim in job.tensorSizes()[0]][1:]
    B = [dim.concreteInt() for dim in job.tensorSizes()[0]][0]
    bytes = math.prod(tensor_size)
    # L = job.inputArguments()["L"].constantValue()
    L = 0
    return htree_in_latency(bytes, L, interconnect.pipelineStages()) * B


def htree_out_latency(N, pipeline_stages):
    num_vecs = math.ceil(N / 512)
    return pipeline_stages + num_vecs * 8


def get_htree_out_latency(interconnect, job):
    tensor_size = [dim.concreteInt() for dim in job.tensorSizes()[0]][1:]
    B = [dim.concreteInt() for dim in job.tensorSizes()[0]][0]
    bytes = math.prod(tensor_size)
    return htree_out_latency(bytes, interconnect.pipelineStages()) * B


def mvm_latency(M, L):
    in_sticks = M // 512
    num_tiles = L * 4
    tiers = math.ceil(in_sticks / (num_tiles))
    data_latency = (in_sticks - 1) * 8
    tier_latency = (200 - num_tiles * 8 - 1) * (tiers - 1)
    tier_switch_latency = 200 * math.floor((tiers - 1) / 4)
    last_tile_latency = 200
    return data_latency + last_tile_latency + tier_latency + tier_switch_latency


def mvm_plus_reduction_latency(M, L):
    mvm_only_latency = mvm_latency(M, L)
    return mvm_only_latency + int(math.log(L, 2)) * 16


def get_mvm_plus_reduction_latency(resource, job):
    B = job.tensorSizes()[0][0].concreteInt()
    M = job.inputArguments()["self"].shape()[-1].concreteInt()
    N = job.tensorSizes()[0][-1].concreteInt()
    L = job.inputArguments()["L"].constantValue()
    latency_per_input_activation = 0 # resource.baseLatency()
    mvm_latency = mvm_plus_reduction_latency(M, L)
    latency_per_output_activation = 0 # htree_out_latency(N, resource.baseLatency())
    return (latency_per_input_activation + mvm_latency + latency_per_output_activation) * B
    #return total_latency if job.opName() != "tads_ops::grouped_mm" else total_latency #* 2


def get_mvm_latency(resource, job):
    pass


def llama_htree_run(M, N, L):
    import json
    from htree_plotting import plot_mvm_timings

    tile_size = 512
    tile_power = 60e-3
    leaf_size = 4
    leaves_per_cluster = 4
    clusters_per_chip = 64
    mvm_latency = 200
    max_power = 300
    tier_switch_latency = 200
    arrays_per_tile = 4
    pcie_gps = 64
    htree_freq_ghz = 1
    pipeline_every_tile = 1
    # Use M and N to set special mapping
    try:
        # 'x' = create for writing, fail if file exists
        with open("_cached_htree_latencies.json", "x") as f:
            json.dump({}, f)  # write an empty JSON object
    except FileExistsError:
        # file already exists—nothing to do
        pass
    with open("_cached_htree_latencies.json", "r") as f:
        jd = json.load(f)
    cached_latencies = {}
    for key, vals in jd.items():
        m, n, l = map(int, key.split(","))
        cached_latencies[(m, n, l)] = tuple(vals)
    if (M, N, L) in cached_latencies:
        # print(f"Loading latencies for MVM of size {M}x{N} from cache")
        latencies = cached_latencies[(M, N, L)]
    else:
        print(f"Latencies for MVM of size {M}x{N} not cached. Computing")
        latencies, timings, _ = map_mvm(
            M,
            N,
            False,
            False,
            tile_size,
            tile_power,
            leaf_size,
            leaves_per_cluster,
            clusters_per_chip,
            mvm_latency,
            max_power,
            tier_switch_latency,
            arrays_per_tile,
            pcie_gps,
            htree_freq_ghz,
            pipeline_every_tile,
            L,
        )
        plot_mvm_timings(timings, latencies[3], f"debug/{M}x{N}.svg")
        cached_latencies[(M, N, L)] = latencies
        jsonable = {
            f"{m},{n},{l}": [a, b, c, d] for (m, n, l), (a, b, c, d) in cached_latencies.items()
        }
        with open("_cached_htree_latencies.json", "w") as f:
            json.dump(jsonable, f)
    # print(
    #     f"M: {M}, N: {N} htree in: {latencies[0]}, mvm: {latencies[1]}, htree out: {latencies[2]}, total: {latencies[3]}"
    # )
    return latencies
