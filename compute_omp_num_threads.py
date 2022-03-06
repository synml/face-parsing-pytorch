import os
import matplotlib.pyplot as plt
import timeit
import torch
import tqdm

if __name__ == '__main__':
    threads = [1] + [t for t in range(2, os.cpu_count() * 2, 2)]
    runtimes = []
    for t in tqdm.tqdm(threads):
        torch.set_num_threads(t)
        r = timeit.timeit(setup="import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)",
                          stmt="torch.mm(x, y)", number=100)
        runtimes.append(r)

    plt.plot(threads, runtimes)
    plt.xticks(threads)
    plt.grid(True)
    plt.savefig('omp_num_threads.png')
