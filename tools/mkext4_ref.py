#!/usr/bin/env python3
# From-scratch ext2/ext4-family mkfs + file writer prototype, iterated against
# e2fsck (the oracle) before porting to no_std Rust. Block-mapped files (12
# direct + single/double/triple indirect) so large files (the model) work.
# Feature set kept minimal: filetype + large_file + sparse_super, 128-B inodes.
import struct, sys, os

BS = 4096                # block size
INODE_SIZE = 128
BPG = BS * 8             # blocks per group (32768)
INO_PER_GROUP = 4096     # inodes per group (tunable)
FIRST_INO = 11
ROOT_INO = 2

# inode i_mode bits
S_IFDIR = 0x4000
S_IFREG = 0x8000

# feature flags
COMPAT = 0
INCOMPAT_FILETYPE = 0x0002
ROCOMPAT_SPARSE_SUPER = 0x0001
ROCOMPAT_LARGE_FILE  = 0x0002

def sparse_group_has_super(g):
    if g in (0, 1): return True
    def pow_of(b):
        x = b
        while x < g: x *= b
        return x == g
    return pow_of(3) or pow_of(5) or pow_of(7)

class Ext:
    def __init__(self, total_blocks):
        self.total_blocks = total_blocks
        self.ngroups = (total_blocks + BPG - 1) // BPG
        self.img = bytearray(total_blocks * BS)
        self.inode_size = INODE_SIZE
        self.ipg = INO_PER_GROUP
        self.itable_blocks = (self.ipg * INODE_SIZE + BS - 1) // BS
        # gdt size
        desc_per = self.ngroups
        self.gdt_blocks = (desc_per * 32 + BS - 1) // BS
        # per-group metadata block counts (super+gdt only in sparse groups)
        self.meta_overhead_sparse = 1 + self.gdt_blocks  # super + gdt
        # bitmaps(2) + itable
        self.per_group_fixed = 2 + self.itable_blocks
        # allocation cursors per group
        self.next_ino = FIRST_INO  # inodes 1..10 reserved, root=2 handled specially
        self.free_blocks = 0
        self.free_inodes = 0
        self.layout()

    def group_start(self, g):
        return g * BPG

    def layout(self):
        # Compute where bitmaps / itable / first data block live in each group.
        self.g_block_bitmap = []
        self.g_inode_bitmap = []
        self.g_inode_table = []
        self.g_first_data = []
        self.block_alloc = []  # per-group next-free data block (absolute)
        for g in range(self.ngroups):
            start = self.group_start(g)
            off = 0
            if sparse_group_has_super(g):
                off += self.meta_overhead_sparse
            bb = start + off; off += 1
            ib = start + off; off += 1
            it = start + off; off += self.itable_blocks
            self.g_block_bitmap.append(bb)
            self.g_inode_bitmap.append(ib)
            self.g_inode_table.append(it)
            self.g_first_data.append(start + off)
            self.block_alloc.append(start + off)
        # group 0 first_data_block is 0 for 4K blocks (block 0 holds super)
        # total blocks in last group may be < BPG
        self.used_blocks = set()

    def blocks_in_group(self, g):
        start = self.group_start(g)
        end = min(start + BPG, self.total_blocks)
        return end - start

    def alloc_block(self, g=None):
        # allocate one data block, preferring group g then any
        order = range(self.ngroups) if g is None else [g] + [x for x in range(self.ngroups) if x != g]
        for gg in order:
            end = self.group_start(gg) + self.blocks_in_group(gg)
            b = self.block_alloc[gg]
            if b < end:
                self.block_alloc[gg] = b + 1
                self.used_blocks.add(b)
                return b
        raise RuntimeError("out of blocks")

    def alloc_inode(self):
        ino = self.next_ino
        self.next_ino += 1
        return ino

    def wr(self, off, data):
        self.img[off:off+len(data)] = data

    def blk(self, b):
        return b * BS

    # ---- inode ----
    def write_inode(self, ino, mode, size, blocks_list, nblocks, links=1, is_dir=False):
        # place inode in its group's table
        idx = ino - 1
        g = idx // self.ipg
        within = idx % self.ipg
        it = self.g_inode_table[g]
        off = self.blk(it) + within * INODE_SIZE
        i = bytearray(INODE_SIZE)
        struct.pack_into('<H', i, 0, mode)
        struct.pack_into('<H', i, 2, 0)      # uid
        struct.pack_into('<I', i, 4, size & 0xffffffff)  # size lo
        struct.pack_into('<I', i, 8, 0)      # atime
        struct.pack_into('<I', i, 12, 0)     # ctime
        struct.pack_into('<I', i, 16, 0)     # mtime
        struct.pack_into('<H', i, 26, links) # links_count
        # i_blocks in 512-byte units = (data + indirect blocks) * (BS/512)
        struct.pack_into('<I', i, 28, nblocks * (BS // 512))
        # block pointers: 15 slots at offset 40 (12 direct + 3 indirect)
        # blocks_list here is the list of DATA block numbers; caller must have
        # already built indirect blocks and pass the 15 i_block slots instead.
        for k in range(15):
            v = blocks_list[k] if k < len(blocks_list) else 0
            struct.pack_into('<I', i, 40 + k*4, v)
        if size >= (1<<31) and not is_dir:
            struct.pack_into('<I', i, 108, size >> 32)  # size_high (dir_acl/size_high)
        self.wr(off, i)

    def build_iblocks(self, data_blocks):
        self._indirect_count = 0
        """Given the list of DATA block numbers, produce the 15 i_block slots,
        allocating indirect blocks as needed (single/double/triple)."""
        slots = [0]*15
        n = len(data_blocks)
        ptrs_per = BS // 4
        # 12 direct
        for k in range(min(12, n)):
            slots[k] = data_blocks[k]
        idx = 12
        if n > idx:
            # single indirect
            si = self.alloc_block(); self._indirect_count += 1
            arr = data_blocks[12:12+ptrs_per]
            self.write_ptr_block(si, arr)
            slots[12] = si
            idx = 12 + len(arr)
        if n > idx:
            # double indirect
            di = self.alloc_block(); self._indirect_count += 1
            singles = []
            rem = data_blocks[idx:]
            for chunk_start in range(0, len(rem), ptrs_per):
                chunk = rem[chunk_start:chunk_start+ptrs_per]
                si = self.alloc_block(); self._indirect_count += 1
                self.write_ptr_block(si, chunk)
                singles.append(si)
                if len(singles) == ptrs_per:
                    break
            self.write_ptr_block(di, singles)
            slots[13] = di
            idx += sum(min(ptrs_per, len(rem)-c) for c in range(0, min(len(rem), ptrs_per*ptrs_per), ptrs_per))
        # (triple indirect omitted; not needed at our sizes)
        return slots

    def write_ptr_block(self, b, ptrs):
        off = self.blk(b)
        buf = bytearray(BS)
        for k, p in enumerate(ptrs):
            struct.pack_into('<I', buf, k*4, p)
        self.wr(off, buf)

    # ---- directory ----
    def make_dir_block(self, entries):
        """entries: list of (inode, name, filetype). Pack into one 4K block."""
        buf = bytearray(BS)
        off = 0
        for j,(ino,name,ft) in enumerate(entries):
            nb = name.encode()
            reclen = 8 + ((len(nb)+3)//4)*4
            if j == len(entries)-1:
                reclen = BS - off  # last entry spans to end
            struct.pack_into('<I', buf, off, ino)
            struct.pack_into('<H', buf, off+4, reclen)
            buf[off+6] = len(nb)
            buf[off+7] = ft
            buf[off+8:off+8+len(nb)] = nb
            off += reclen
        return bytes(buf)


def mkfs(path, size_mb, files):
    total_blocks = size_mb*1024*1024 // BS
    fs = Ext(total_blocks)

    # allocate + write regular files first, collect (name,ino) for root dir
    root_entries = [(ROOT_INO, '.', 2), (ROOT_INO, '..', 2)]
    used_inodes = set([1,2,3,4,5,6,7,8,9,10,11])  # 1..10 reserved + root(2); we count later
    file_inodes = []
    for (name, data) in files:
        nblocks = (len(data)+BS-1)//BS
        dblocks = [fs.alloc_block() for _ in range(nblocks)]
        for k,b in enumerate(dblocks):
            chunk = data[k*BS:(k+1)*BS]
            fs.wr(fs.blk(b), chunk)
        ino = fs.alloc_inode()
        slots = fs.build_iblocks(dblocks)
        fs.write_inode(ino, S_IFREG|0o644, len(data), slots, len(dblocks)+fs._indirect_count, links=1)
        root_entries.append((ino, name, 1))  # filetype 1 = regular
        file_inodes.append(ino)

    # root directory data block
    rootblk = fs.alloc_block(0)
    fs.wr(fs.blk(rootblk), fs.make_dir_block(root_entries))
    fs.write_inode(ROOT_INO, S_IFDIR|0o755, BS, [rootblk]+[0]*14, 1, links=2)

    # reserved inodes 1..10: inode 1 (bad blocks) empty ok; others zero. Only
    # need lost+found? e2fsck wants lost+found (inode 11) ideally but not fatal.
    # Write bitmaps + inode table already placed. Now block/inode bitmaps + GDT + SB.
    finalize(fs, used_file_inodes=file_inodes)
    open(path,'wb').write(fs.img)


def finalize(fs, used_file_inodes):
    total_used_inodes = 10 + 1 + len(used_file_inodes)  # 1..10 + root + files (root is #2 within 1..10 range! adjust)
    # inodes 1..10 reserved (includes root=2). files start at 11.
    used_inodes = set(range(1,11)) | set(used_file_inodes)
    # ---- block bitmaps ----
    for g in range(fs.ngroups):
        start = fs.group_start(g)
        nb = fs.blocks_in_group(g)
        bm = bytearray(BS)
        # mark metadata + allocated data blocks used
        for b in range(start, start+nb):
            used = (b in fs.used_blocks)
            # metadata blocks:
            if sparse_group_has_super(g) and b < start + fs.meta_overhead_sparse:
                used = True
            if b == fs.g_block_bitmap[g] or b == fs.g_inode_bitmap[g]:
                used = True
            if fs.g_inode_table[g] <= b < fs.g_inode_table[g] + fs.itable_blocks:
                used = True
            if b < fs.block_alloc[g] and b >= fs.g_first_data[g]:
                # data block region up to alloc cursor: used only if actually allocated
                used = used
            if used:
                bit = b - start
                bm[bit//8] |= (1 << (bit%8))
        # pad bits beyond nb as used
        for bit in range(nb, BS*8):
            bm[bit//8] |= (1 << (bit%8))
        fs.wr(fs.blk(fs.g_block_bitmap[g]), bm)
    # ---- inode bitmaps ----
    for g in range(fs.ngroups):
        bm = bytearray(BS)
        base = g*fs.ipg
        for within in range(fs.ipg):
            ino = base + within + 1
            if ino in used_inodes:
                bm[within//8] |= (1 << (within%8))
        for bit in range(fs.ipg, BS*8):
            bm[bit//8] |= (1 << (bit%8))
        fs.wr(fs.blk(fs.g_inode_bitmap[g]), bm)

    # ---- group descriptors ----
    total_free_blocks = 0
    total_free_inodes = 0
    gdt = bytearray(fs.gdt_blocks*BS)
    for g in range(fs.ngroups):
        nb = fs.blocks_in_group(g)
        used_b = sum(1 for b in range(fs.group_start(g), fs.group_start(g)+nb)
                     if (b in fs.used_blocks)
                     or (sparse_group_has_super(g) and b < fs.group_start(g)+fs.meta_overhead_sparse)
                     or b==fs.g_block_bitmap[g] or b==fs.g_inode_bitmap[g]
                     or (fs.g_inode_table[g] <= b < fs.g_inode_table[g]+fs.itable_blocks))
        free_b = nb - used_b
        base = g*fs.ipg
        used_i = sum(1 for within in range(fs.ipg) if (base+within+1) in used_inodes)
        free_i = fs.ipg - used_i
        dirs = 1 if g==0 else 0
        total_free_blocks += free_b; total_free_inodes += free_i
        e = g*32
        struct.pack_into('<I', gdt, e+0, fs.g_block_bitmap[g])
        struct.pack_into('<I', gdt, e+4, fs.g_inode_bitmap[g])
        struct.pack_into('<I', gdt, e+8, fs.g_inode_table[g])
        struct.pack_into('<H', gdt, e+12, free_b)
        struct.pack_into('<H', gdt, e+14, free_i)
        struct.pack_into('<H', gdt, e+16, dirs)
    # write GDT in every sparse group (right after its superblock)
    for g in range(fs.ngroups):
        if sparse_group_has_super(g):
            gdt_start = fs.group_start(g) + 1
            fs.wr(fs.blk(gdt_start), gdt)

    # ---- superblock (+ backups) ----
    sb = bytearray(1024)
    total_inodes = fs.ipg * fs.ngroups
    struct.pack_into('<I', sb, 0, total_inodes)          # s_inodes_count
    struct.pack_into('<I', sb, 4, fs.total_blocks)       # s_blocks_count
    struct.pack_into('<I', sb, 8, fs.total_blocks//20)   # r_blocks
    struct.pack_into('<I', sb, 12, total_free_blocks)    # free blocks
    struct.pack_into('<I', sb, 16, total_free_inodes)    # free inodes
    struct.pack_into('<I', sb, 20, 0)                    # first_data_block (0 for 4K)
    struct.pack_into('<I', sb, 24, 2)                    # log_block_size (4K)
    struct.pack_into('<I', sb, 28, 2)                    # log_cluster_size
    struct.pack_into('<I', sb, 32, BPG)                  # blocks_per_group
    struct.pack_into('<I', sb, 36, BPG)                  # clusters_per_group
    struct.pack_into('<I', sb, 40, fs.ipg)               # inodes_per_group
    struct.pack_into('<H', sb, 56, 0xEF53)               # magic
    struct.pack_into('<H', sb, 58, 1)                    # state = clean
    struct.pack_into('<H', sb, 60, 1)                    # errors = continue
    struct.pack_into('<I', sb, 76, 1)                    # rev_level = dynamic
    struct.pack_into('<I', sb, 84, FIRST_INO)            # first_ino
    struct.pack_into('<H', sb, 88, INODE_SIZE)           # inode_size
    struct.pack_into('<I', sb, 92, COMPAT)               # feature_compat
    struct.pack_into('<I', sb, 96, INCOMPAT_FILETYPE)    # feature_incompat
    struct.pack_into('<I', sb, 100, ROCOMPAT_SPARSE_SUPER|ROCOMPAT_LARGE_FILE) # ro_compat
    for g in range(fs.ngroups):
        if sparse_group_has_super(g):
            # backup group number field s_block_group_nr @ 90
            struct.pack_into('<H', sb, 90, g)
            fs.wr(fs.blk(fs.group_start(g)) + (1024 if g==0 else 0), bytes(sb))
    # note: for g>0 the superblock sits at the very start of the group block


if __name__ == '__main__':
    files = [('hello.txt', b'hello from a hand-built ext filesystem\n'),
             ('big.bin', bytes((i*7) & 0xff for i in range(200000)))]  # ~200KB -> indirect
    mkfs('/tmp/mine.img', 64, files)
    print("wrote /tmp/mine.img")
