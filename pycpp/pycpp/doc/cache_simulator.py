class CacheSimulator:
    def __init__(self, size=2048*1024, way=16):
        self.size = size
        self.way = way
        self.num_sets = (size // (way*64))
        self.sets = [0] * self.num_sets
        print(f"Cache {self.way} way {self.size/1024/1024:.1f} MB {len(self.sets)} sets")

    def test(self, K, stride_bytes):
        for i in range(self.num_sets):
            self.sets[i] = 0

        for k in range(K):
            offset_cache_line = (k * stride_bytes) // 64
            cache_set_index = offset_cache_line % self.num_sets
            self.sets[cache_set_index] += 1

        cached = 0
        for i in range(self.num_sets):
            num_cache_lines = self.sets[i]
            if num_cache_lines > 0:
                cached += self.way if num_cache_lines > self.way else num_cache_lines
                # print(f"set[{i}] : {self.sets[i]}")
        # print(f"total {K*64/1024:.1f} KB, cached {cached*64/1024:.1f} KB, hit-rate: {cached*100/K:.1f} % ") 
        return cached*64

if __name__ == "__main__":
    cs = CacheSimulator()
    K = 3520
    total_mem = K*64
    for i in range(1, 100):
        cached = cs.test(K, i*64)
        if cached != total_mem:
            print(f" stride {i} cache-lines: total {K*64/1024:.1f} KB, cached {cached/1024:6.1f} KB, hit-rate: {cached*100/(K*64):.1f} %")
