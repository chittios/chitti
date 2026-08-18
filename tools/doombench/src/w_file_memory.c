/* A memory-backed WAD backend, substituted for upstream's `w_file_stdc.c`.
 *
 * Why substitute rather than add: `w_file.c` reaches the backend as
 * `stdc_wad_file.OpenFile(path)` -- a concrete symbol, not a registry -- so the
 * way to replace it without editing upstream (VENDORING.md's rule) is to leave
 * `w_file_stdc.c` out of the build and define `stdc_wad_file` here. That is the
 * same substitution the platform layer makes for `doomgeneric_*.c`.
 *
 * Why memory at all: on ChittiOS there is no file descriptor to give Doom. A wasm
 * guest reads the WAD with `host_fs_read`, which returns *bytes into linear
 * memory*; a ring-3 tenant gets the same thing in its arena. So the real port must
 * serve lumps out of a buffer no matter which route it takes, and the bench should
 * measure that rather than a host `fopen` neither will ever do.
 *
 * Setting `mapped` is what makes this fast: with a non-NULL `mapped`, Doom reads
 * lumps straight out of the buffer instead of copying each one through `W_Read`.
 * Both builds get it, so the comparison stays like-for-like.
 */

#include <stdlib.h>
#include <string.h>

#include "doomtype.h"
#include "w_file.h"

static const unsigned char *wad_base = NULL;
static unsigned int wad_size = 0;

/* Staging for the WAD, in `.bss`.
 *
 * The wasm side needs a *stable address inside guest linear memory* to write the
 * WAD to before Doom starts, and it cannot call `malloc` (not exported) or guess a
 * free region. A `.bss` array gives the host an address it can ask for and the
 * guest an address it owns, with no allocator involved -- and it is the same code
 * natively, so the two builds stay like-for-like.
 *
 * Freedoom Phase 1 is ~29 MB; 40 MB leaves room without making the module's
 * initial memory absurd.
 */
static unsigned char wad_storage[40u << 20];

__attribute__((export_name("dg_wad_storage")))
unsigned char *dg_wad_storage(void) { return wad_storage; }

__attribute__((export_name("dg_wad_capacity")))
unsigned int dg_wad_capacity(void) { return (unsigned int)sizeof wad_storage; }

/* Called by the harness before `dg_create`, naming the bytes it just staged.
 * Deliberately takes a buffer rather than a path: the caller owns the bytes for
 * the process's life, which is true of `wad_storage` in both builds.
 */
__attribute__((export_name("dg_set_wad")))
void dg_set_wad(const unsigned char *base, unsigned int len) {
    wad_base = base;
    wad_size = len;
}

extern wad_file_class_t stdc_wad_file;

static wad_file_t *mem_OpenFile(char *path) {
    (void)path; /* There is one WAD and it is already in memory. */
    if (wad_base == NULL || wad_size == 0) {
        return NULL;
    }
    wad_file_t *f = malloc(sizeof(wad_file_t));
    if (f == NULL) {
        return NULL;
    }
    f->file_class = &stdc_wad_file;
    /* Non-const cast: Doom's struct predates const-correctness and never writes
     * through this pointer -- `mapped` is only ever read (w_wad.c caches lumps
     * from it). */
    f->mapped = (byte *)wad_base;
    f->length = wad_size;
    return f;
}

static void mem_CloseFile(wad_file_t *f) {
    /* The buffer is the caller's; only the handle is ours. */
    free(f);
}

static size_t mem_Read(wad_file_t *f, unsigned int offset,
                       void *buffer, size_t buffer_len) {
    /* Every bound is checked rather than trusted, and a read that runs past the
     * end is *clamped to the available bytes* the way a short file read would be
     * -- Doom checks the returned count. Returning the requested length would
     * hand it uninitialised memory as WAD data, which is a mis-parse rather than
     * an error and would surface much later as a corrupt lump.
     */
    if (f == NULL || offset >= f->length) {
        return 0;
    }
    size_t avail = (size_t)(f->length - offset);
    size_t n = buffer_len < avail ? buffer_len : avail;
    memcpy(buffer, f->mapped + offset, n);
    return n;
}

wad_file_class_t stdc_wad_file = {
    mem_OpenFile,
    mem_CloseFile,
    mem_Read,
};
