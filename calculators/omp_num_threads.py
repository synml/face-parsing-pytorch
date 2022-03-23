import os
import matplotlib.pyplot as plt
import timeit
import torch
import tqdm


def calculate_omp_num_threads():
    threads = [1] + [t for t in range(2, os.cpu_count() * 2, 2)]
    runtimes = []
    for t in tqdm.tqdm(threads):
        torch.set_num_threads(t)
        r = timeit.timeit(setup='import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)',
                          stmt='torch.mm(x, y)', number=100)
        runtimes.append(r)

    min_runtime = min(runtimes)
    thread_with_min_runtime = threads[runtimes.index(min_runtime)]

    plt.plot(threads, runtimes, label=f'min=({thread_with_min_runtime}: {min_runtime:.2f})')
    plt.xlabel('Threads')
    plt.ylabel('Runtimes')
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('omp_num_threads.png', dpi=400)


if __name__ == '__main__':
    calculate_omp_num_threads()
