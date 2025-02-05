
'''
 - Ai has the ability simulate first, so it can proceed
 - Ai must have a goal
 - Ai searches all choices to try to go near the goal, this is the AI part, assume:
    it has to choose 2 reg from 16 regs: there are 240 choices
    it has to choose from instructions
    and it has to explore all possibilties in some depth to get to search for target value

    the choices of each step is b(broadness), and total steps(instructions) to generated is d(depth), 
    the total choices would be (b^d), that's too much to track!

    So the key of AI is :
        HOW to explore such big of search space efficiently and still find the solution !

    at first glance it seems to be impossible, but we can make it simpler by realizing that
    with the help of vperm2i128 instruction, we can simplify our target as making each 128bit SIMD lanes
    perfectly match the target 128bit lanes.

    heuristic search is the key to success, we need to develop a estimation function which estimate the best
    choice of current step w/o doing a full search of future choices, this `insightful` estimation function
    is the key, it narrows the wideness of the search a lot !!!

    Just as most decision tree problem, instinct is the key part which almost directly leads us to the solution.

    https://en.wikipedia.org/wiki/Heuristic_(computer_science)
    https://zh.wikipedia.org/wiki/%E5%90%AF%E5%8F%91%E5%BC%8F%E7%AE%97%E6%B3%95#%E5%95%9F%E7%99%BC%E5%BC%8F%E6%BC%94%E7%AE%97%E6%B3%95%E5%B0%8D%E9%81%8B%E7%AE%97%E6%95%88%E8%83%BD%E7%9A%84%E5%BD%B1%E9%9F%BF
'''
import numpy as np
import sys
import functools

np.set_printoptions(threshold=sys.maxsize, linewidth=1000, formatter={'int':hex})

class VirtualMachine(object):
    def __init__(self, reg_cnt, simd_lanes):
        self.simd_lanes = simd_lanes
        self.vregs = np.zeros(shape=(reg_cnt, simd_lanes)).astype(np.int32)
        self.logs = {}
        for i in range(reg_cnt):
            for k in range(simd_lanes):
                self.vregs[i, k] = i * 100 + k

        self.searching = False

    def log(self, dst, source_code):
        if self.searching: return
        if dst in self.logs:
            self.logs[dst] += source_code
        else:
            self.logs[dst] = source_code

    def to_str(self, v):
        ret = ""
        for k in range(len(v)):
            vec_id = v[k]//100
            name = f"{v[k]:03d}"
            ret += f"\x1b[{vec_id + 100}m{name}\x1b[0m "
        return ret

    def __repr__(self):
        ret = ""
        for i, v in enumerate(self.vregs):
            vname = f"v{i}"
            ret += f"{vname:3} : {self.to_str(v)}"
            if (i in self.logs):
                ret += "\t" + self.logs[i]
            ret += "\n"
        return ret

    def comment(self, txt = ""):
        self.log(self.vregs.shape[0]-1, "\n" + txt)

    def vpunpcklbw(self, dst, src1, src2):
        self._vpunpck(dst, src1, src2, 0, 1)
        self.log(dst, f"vpunpcklbw({dst}, {src1}, {src2})")

    def vpunpckhbw(self, dst, src1, src2):
        self._vpunpck(dst, src1, src2, 1, 1)
        self.log(dst, f"vpunpckhbw({dst}, {src1}, {src2})")

    def vpunpcklwd(self, dst, src1, src2):
        self._vpunpck(dst, src1, src2, 0, 2)
        self.log(dst, f"vpunpcklwd({dst}, {src1}, {src2})")

    def vpunpckhwd(self, dst, src1, src2):
        self._vpunpck(dst, src1, src2, 1, 2)
        self.log(dst, f"vpunpckhwd({dst}, {src1}, {src2})")

    def _vpunpck(self, dst, src1, src2, hi, ele_bytes):
        # https://www.felixcloutier.com/x86/punpcklbw:punpcklwd:punpckldq:punpcklqdq
        # 128bit : 16 bytes element
        offset = 8 if hi else 0
        new_vreg = np.zeros(shape=(1, self.simd_lanes)).astype(np.int32)
        for k0 in range(0, self.simd_lanes, 16):
            for k1 in range(0, 8, ele_bytes):
                dk0 = k0 + k1*2
                dk1 = dk0 + ele_bytes
                dk2 = dk1 + ele_bytes
                sk0 = k0 + k1 + offset
                sk1 = sk0 + ele_bytes
                new_vreg[0, dk0:dk1] = self.vregs[src1, sk0:sk1]
                new_vreg[0, dk1:dk2] = self.vregs[src2, sk0:sk1]
        self.assign(dst, new_vreg)

    def vperm2i128(self, dst, src1, src2, imm8):
        def get128(src1, src2, imm):
            if ((imm & 3) == 0):
                tmp = self.vregs[src1, 0:16]
            if ((imm & 3) == 1):
                tmp = self.vregs[src1, 16:]
            if ((imm & 3) == 2):
                tmp = self.vregs[src2, 0:16]
            if ((imm & 3) == 3):
                tmp = self.vregs[src2, 16:]
            if (imm & 4):
                tmp = np.zeros(shape=(1, 16)).astype(np.int32)
            return tmp

        new_vreg = np.zeros(shape=(1, self.simd_lanes)).astype(np.int32)
        new_vreg[0, 0:16] = get128(src1, src2, imm8 & 0xf)
        new_vreg[0, 16:] = get128(src1, src2, imm8 >> 4)
        self.assign(dst, new_vreg)
        self.log(dst, f"vperm2i128({dst}, {src1}, {src2}, {imm8})")

    def vpermq(self, dst, src1, imm8):
        def get64(src1, imm):
            if (imm == 0):
                tmp = self.vregs[src1, 0:8]
            if (imm == 1):
                tmp = self.vregs[src1, 8:16]
            if (imm == 2):
                tmp = self.vregs[src2, 16:24]
            if (imm == 3):
                tmp = self.vregs[src2, 24:32]
            return tmp
        new_vreg = np.zeros(shape=(1, self.simd_lanes)).astype(np.int32)
        new_vreg[0:8] = get64(src1, imm & 3)
        new_vreg[8:16] = get64(src1, (imm >> 2) & 3)
        new_vreg[16:24] = get64(src1, (imm >> 4) & 3)
        new_vreg[24:32] = get64(src1, (imm >> 6) & 3)
        self.assign(dst, new_vreg)
        self.log(dst, f"vpermq({dst}, {src1}, {imm8})")

    def vmovdqu(self, dst, values):
        new_vreg = np.zeros(shape=(1, self.simd_lanes)).astype(np.int32)
        new_vreg[0, :] = values
        self.assign(dst, new_vreg)
        self.log(dst, f"vmovdqu({dst}, ...)")

    def vpshufb(self, dst, src1, src2):
        new_vreg = np.zeros(shape=(1, self.simd_lanes)).astype(np.int32)
        for off in [0, 16]:
            for k in range(16):
                i0 = off + k
                index = self.vregs[src2, i0]
                j0 = off + (index & 0xF)
                if (index & 0x80):
                    new_vreg[0, i0] = 0
                else:
                    new_vreg[0, i0] = self.vregs[src1, j0]
        self.assign(dst, new_vreg)
        self.log(dst, f"vpshufb({dst}, {src1}, {src2})")

    def assign(self, dst, new_vreg):
        if (dst < self.vregs.shape[0]):
            self.vregs[dst, :] = new_vreg[0, :]
        elif (dst == self.vregs.shape[0]):
            self.vregs = np.vstack([self.vregs, new_vreg])
        else:
            raise Exception(f"dst register {dst} is not exist and not the next!")

    # this similarity function is the core for Heuristic searching, can be used for both choicing
    # the right sources & evaluate the goodness of result matching against target
    def vect_similarity(self, vtarget, vsrc):
        similarity = 0
        for i, v in enumerate(vsrc):
            # exact location match with 128bit lanes
            '''
            if v == vtarget[i]:
                similarity += 2
            elif i < 16 and v == vtarget[i + 16]:
                similarity += 1
            elif i >= 16 and v == vtarget[i - 16]:
                similarity += 1
            '''
            if v == vtarget[i]:
                similarity += 2
            
            if v in vtarget:
                similarity += 1
                next_idx = np.argwhere(vtarget == v) + 1
                # in right order, increase similarity
                if (next_idx < len(vtarget) and i + 1 < len(vsrc) and vsrc[i+1] == vtarget[next_idx]):
                    similarity += 1
        return similarity

    def vect_similarity_v1(self, vtarget, vsrc):
        similarity = 0
        for i, v in enumerate(vsrc):
            if v == vtarget[i]:
                similarity += 1
        return similarity

    def new_vregs(self, count):
        return np.zeros(shape=(count, self.simd_lanes)).astype(np.int32)

    def suggest(self, target_vreg_str):
        self.searching = True
        if isinstance(target_vreg_str, str):
            target_vreg = np.zeros(shape=(1, self.simd_lanes)).astype(np.int32)
            for i, v in enumerate(target_vreg_str.split(" ")):
                target_vreg[0, i] = int(v)
        else:
            target_vreg = target_vreg_str

        self.ILIST = [
            (self.vpunpcklbw, "vpunpcklbw({dst}, {src1}, {src2})"),
            (self.vpunpckhbw, "vpunpckhbw({dst}, {src1}, {src2})"),
            (self.vpunpcklwd, "vpunpcklwd({dst}, {src1}, {src2})"),
            (self.vpunpckhwd, "vpunpckhwd({dst}, {src1}, {src2})"),
        ]
        print("target: ", self.to_str(target_vreg[0]))
        #
        #  get src vregs list
        #  in this step, human instinct would do clever choice:
        #    - choose the src contains dst values first
        #    - choose the src with dst value in right order & sequence first
        #
        src_list = []
        for i, v in enumerate(self.vregs):
            score = self.vect_similarity(target_vreg[0], v)
            if score > 0:
                src_list.append((i, score))

        src_list.sort(key=lambda item: item[1], reverse=True)

        print(f"src_list: {src_list}")
        # choose two vreg & one instruction & sort results according to matching score.
        # and sort the result according to vect_similarity against target
        dst = self.vregs.shape[0]
        N = len(src_list)

        choices = []
        for i1 in range(0, N):
            for i2 in range(0, N):
                if (i1 == i2): continue
                src1 = src_list[i1][0]
                src2 = src_list[i2][0]
                for f in self.ILIST:
                    f[0](dst, src1, src2)
                    score = self.vect_similarity(target_vreg[0], self.vregs[dst, :])
                    choices.append((score, f[1].format(dst=dst, src1=src1, src2=src2)))
                # try vperm2i128 (src1 & src2 are exchangable)
                if (i2 >= i1):
                    scores = []
                    scores.append(self.vect_similarity_v1(target_vreg[0, 0:16], self.vregs[src1, 0:16]))
                    scores.append(self.vect_similarity_v1(target_vreg[0, 0:16], self.vregs[src1, 16:]))
                    scores.append(self.vect_similarity_v1(target_vreg[0, 0:16], self.vregs[src2, 0:16]))
                    scores.append(self.vect_similarity_v1(target_vreg[0, 0:16], self.vregs[src2, 16:]))
                    imm_lo = scores.index(max(scores))
                    scores = []
                    scores.append(self.vect_similarity_v1(target_vreg[0, 16:], self.vregs[src1, 0:16]))
                    scores.append(self.vect_similarity_v1(target_vreg[0, 16:], self.vregs[src1, 16:]))
                    scores.append(self.vect_similarity_v1(target_vreg[0, 16:], self.vregs[src2, 0:16]))
                    scores.append(self.vect_similarity_v1(target_vreg[0, 16:], self.vregs[src2, 16:]))
                    imm_hi = scores.index(max(scores))
                    best_imm = (imm_hi << 4) + imm_lo
                    self.vperm2i128(dst, src1, src2, best_imm)
                    score = self.vect_similarity(target_vreg[0], self.vregs[dst, :])
                    choices.append((score, f"vperm2i128({dst}, {src1}, {src2}, 0x{best_imm:02x})"))

        choices.sort(key=lambda item: item[0], reverse=True)
        for c in choices[:10]:
            print(c[0], c[1])
        self.searching = False


vm = VirtualMachine(4, 32)
vm.comment()
vm.vpunpcklbw(4, 0, 1)
vm.vpunpcklbw(5, 2, 3)
vm.vpunpcklwd(6, 4, 5)
#vm.vpunpcklwd(5, 4, 0)

vm.suggest("000 100 200 300 001 101 201 301 002 102 202 302 003 103 203 303 004 104 204 304 005 105 205 305 006 106 206 306 007 107 207 307")
print(vm); raise 0

vm.comment()
vm.vpunpcklbw(4, 0, 1)
vm.vpunpcklbw(5, 2, 3)
vm.vpunpckhbw(6, 0, 1)
vm.vpunpckhbw(7, 2, 3)

vm.comment()
print(vm)
raise 0

vm.vpunpcklwd(8, 4, 5)
vm.vpunpcklwd(9, 6, 7)
vm.vpunpckhwd(10, 4, 5)
vm.vpunpckhwd(11, 6, 7)

vm.comment()
print(vm)

vm.suggest("000 100 200 300 001 101 201 301 002 102 202 302 003 103 203 303 004 104 204 304 005 105 205 305 006 106 206 306 007 107 207 307")
#vm.vperm2i128(12, 8, 10, 0x20)
#vm.vperm2i128(13, 9, 11, 0x20)
#vm.vperm2i128(14, 8, 10, 0x31)
#vm.vperm2i128(15, 9, 11, 0x31)


