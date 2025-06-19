"""Utility functions module."""

import time
import threading
import functools
import os
import psutil
import matplotlib.pyplot as plt


def consecutive_groups(numbers: list[int]) -> list[list[int]]:
    """
    Splits list of numbers into consecutive groups

    Args:
      numbers: list of numbers

    Returns:
      list of consecutive groups
    """

    if not numbers:
        return []

    group = [numbers[0]]
    groups = []
    for i in range(1, len(numbers)):
        if numbers[i] - numbers[i - 1] == 1:
            group.append(numbers[i])
        else:
            groups.append(group)
            group = [numbers[i]]
    groups.append(group)

    return groups


def find_duplicate_indices(lst: list[int]) -> list[int]:
    """
    Finds the indices of duplicate elements in a list

    Args:
      lst: input list

    Returns:
      list of duplicate indices
    """

    seen = set()
    duplicates = []
    for i, num in enumerate(lst):
        if num in seen:
            duplicates.append(i)
        else:
            seen.add(num)

    return duplicates


def profile_performance():  # output_filename="performance_profile.png"):
    """
    A decorator that profiles the CPU and memory usage of the decorated function
    and plots the data as a time-series graph.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cpu_usages = []
            memory_usages = []  # in MB
            timestamps = []

            process = psutil.Process(os.getpid())

            monitoring_active = True

            def monitor_resources():
                """
                Function to collect resource usage data.
                """
                start_time = time.time()
                while monitoring_active:
                    try:
                        cpu_percent = process.cpu_percent(interval=None)
                        memory_info = process.memory_info()
                        rss_mb = memory_info.rss / (1024 * 1024)

                        current_time = time.time() - start_time

                        cpu_usages.append(cpu_percent)
                        memory_usages.append(rss_mb)
                        timestamps.append(current_time)

                        time.sleep(0.1)
                    except Exception as e:
                        print(f"Error during monitoring: {e}")
                        break

            print(f"Profiling function: {func.__name__}...")
            monitor_thread = threading.Thread(target=monitor_resources)
            monitor_thread.daemon = True
            monitor_thread.start()

            func_start_time = time.perf_counter()
            result = func(*args, **kwargs)
            func_end_time = time.perf_counter()
            total_execution_time = func_end_time - func_start_time
            print(
                f"Function {func.__name__} finished in {total_execution_time:.4f} seconds."
            )
            monitoring_active = False
            monitor_thread.join(timeout=1)
            print("Generating performance graph...")
            _, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(timestamps, cpu_usages, "b-", label="CPU usage (%)")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("CPU usage (%)", color="b")
            ax1.tick_params("y", colors="b")
            ax1.set_title("Performance profile")
            ax1.grid(True)
            ax2 = ax1.twinx()
            ax2.plot(timestamps, memory_usages, "r--", label="Memory usage (MB)")
            ax2.set_ylabel("Memory usage (MB)", color="r")
            ax2.tick_params("y", colors="r")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc="upper left")
            plt.tight_layout()

            return result

        return wrapper

    return decorator
