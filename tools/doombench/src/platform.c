/* The bench's `doomgeneric` platform layer: the six DG_ functions, plus a small
 * export surface the harness drives.
 *
 * This is deliberately the *same file* for the native and wasm sides. The whole
 * point of the harness is to price the interpreter, so anything that differs
 * between the two builds pollutes the ratio — including the platform layer. It is
 * also intentionally close in shape to the real ChittiOS platform layer, so what
 * is measured here is what will run there:
 *
 *   - no display: the frame is left in DG_ScreenBuffer for the host to fetch,
 *     exactly as the real port leaves it for `ui_present` to bounds-check and blit.
 *   - no sleeping: the host paces frames, so DG_SleepMs must not block. On the OS
 *     a blocking sleep inside a frame would freeze the cooperative scheduler.
 *   - a monotonic millisecond clock, which is all `host_now_ms()` provides.
 *   - keys arrive as (pressed, key) *edges*, which is what DG_GetKey wants and
 *     what the OS cannot currently deliver — see the plan's input section.
 */

#include <stdint.h>
#include <string.h>

#include "doomgeneric.h"
#include "doomkeys.h"

#define KEYQ 64

/* wasi-libc has no `system()`, and `i_system.c` references it for the external
 * music-player option (`snd_musiccmd`). Stubbed here rather than upstream: it is a
 * gap in the *target's* libc, not a defect in Doom, and there is no shell on
 * ChittiOS to run a command with either. Returning -1 is what a real `system()`
 * reports when it cannot spawn, which is the honest answer on both.
 */
#ifdef __wasm__
int system(const char *cmd) { (void)cmd; return -1; }
#endif

/* Virtual clock. A real clock would make the benchmark non-deterministic *and*
 * let Doom's own frame skipping react to how fast the interpreter is, which would
 * silently flatter the slow side: at 30x, wasm would run fewer game tics per frame
 * and each frame would get cheaper. Advancing a fixed step per frame pins both
 * builds to the same work. Doom runs at 35 Hz internally (TICRATE).
 */
static uint32_t now_ms = 0;
static const uint32_t MS_PER_FRAME = 1000 / 35;

static struct { int pressed; unsigned char key; } keyq[KEYQ];
static int kq_head = 0, kq_tail = 0;

void DG_Init(void) {}

/* The frame is already in DG_ScreenBuffer; there is nothing to present to. */
void DG_DrawFrame(void) {}

/* Must not block -- but time must still move, or Doom livelocks.
 *
 * `TryRunTics` waits for the game clock to reach the next tic by looping
 * `{ I_Sleep(1); NetUpdate(); }`, and `doomgeneric_Create` reaches it (via
 * `D_DoomLoop`) *before* it returns. So a platform whose clock only advances
 * per-frame spins here forever: the first sample of this harness sat in
 * TryRunTics -> NetUpdate -> DG_GetTicksMs with the clock pinned at 0.
 *
 * Advancing by the requested amount is both the fix and the honest semantics:
 * Doom asked to sleep `ms`, and in virtual time it did. It stays deterministic
 * because Doom's own loop asks for the same sleeps in the same order every run.
 *
 * This is a real hazard for the ChittiOS port too, and in a nastier form: there
 * the clock is `host_now_ms()` and *does* advance, so instead of a livelock the
 * frame pump would spin inside one guest call, burning fuel and never yielding to
 * `upkeep()` -- a frozen shell rather than a frozen game. The port's DG_SleepMs
 * must return immediately and let the pump own pacing.
 */
void DG_SleepMs(uint32_t ms) {
    now_ms += ms > 0 ? ms : 1;
}

uint32_t DG_GetTicksMs(void) { return now_ms; }

int DG_GetKey(int *pressed, unsigned char *key) {
    if (kq_head == kq_tail) return 0;
    *pressed = keyq[kq_tail].pressed;
    *key = keyq[kq_tail].key;
    kq_tail = (kq_tail + 1) % KEYQ;
    return 1;
}

void DG_SetWindowTitle(const char *title) { (void)title; }

/* ---- the harness surface ------------------------------------------------- */

/* Start straight into a level and render the real 3D view.
 *
 * `-warp 1 1` rather than `-timedemo`: a timedemo is the better *oracle* (it is
 * deterministic and prints a gametic count, which is what Phase 3 compares), but
 * it calls `I_Quit` -> `exit()` when it finishes, and under wasm that is
 * `proc_exit` trapping the instance mid-measurement. Warping to a map needs no
 * input, renders the full 3D view every frame -- which is the thing being priced
 * -- and never terminates, so the harness decides how many frames to run.
 *
 * The virtual clock (see above) is what makes it deterministic without a demo:
 * both builds advance the same 28 ms per frame regardless of how long a frame
 * actually took, so both simulate identical game state.
 */
/* The IWAD path, set by the harness before `dg_create`.
 *
 * Needed even though the bytes are already in memory: Doom picks its *game mode*
 * from the IWAD's **filename** (`D_IdentifyIWADByName` -- freedoom1.wad is
 * doom1-shaped, four episodes), and `D_FindWADByName` checks the file exists
 * before opening it. So the path is not how the data is loaded, it is how the game
 * knows which game it is. Without it Doom reports "Game mode indeterminate".
 */
static char iwad_path[512];

__attribute__((export_name("dg_iwad_path_buf")))
char *dg_iwad_path_buf(void) { return iwad_path; }

__attribute__((export_name("dg_iwad_path_cap")))
unsigned int dg_iwad_path_cap(void) { return (unsigned int)sizeof iwad_path; }

__attribute__((export_name("dg_create")))
void dg_create(void) {
    static char a0[] = "doom";
    static char a1[] = "-iwad";
    static char a3[] = "-warp";
    static char a4[] = "1";
    static char a5[] = "1";
    char *argv[] = {a0, a1, iwad_path, a3, a4, a5, 0};
    doomgeneric_Create(6, argv);
}

__attribute__((export_name("dg_tick")))
void dg_tick(void) {
    now_ms += MS_PER_FRAME;
    doomgeneric_Tick();
}

/* Where the pixels are and how many. The real port reports exactly this triple
 * through `ui_present`, and the kernel bounds-checks every number against live
 * guest memory rather than trusting it — the image tenant's rule. Reporting it
 * the same way here keeps the two ports honest with each other.
 */
__attribute__((export_name("dg_frame_ptr")))
uint32_t dg_frame_ptr(void) { return (uint32_t)(uintptr_t)DG_ScreenBuffer; }

__attribute__((export_name("dg_frame_len")))
uint32_t dg_frame_len(void) { return (uint32_t)(DOOMGENERIC_RESX * DOOMGENERIC_RESY); }

/* A cheap content hash of the current frame, so native and wasm can be compared
 * for *agreement* and not only for speed. A faster build that renders something
 * else has changed the program, not won — the rule `/html bench` follows when it
 * reports agreement before it reports a ratio.
 *
 * FNV-1a over the paletted frame: the palette is applied by the host, so the
 * 8-bit indices are the decoder's actual output.
 */
__attribute__((export_name("dg_frame_hash")))
uint64_t dg_frame_hash(void) {
    const uint8_t *p = (const uint8_t *)DG_ScreenBuffer;
    uint64_t h = 1469598103934665603ULL;
    uint32_t n = DOOMGENERIC_RESX * DOOMGENERIC_RESY;
    for (uint32_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

/* The simulation clock. Two builds that agree here agree on the whole game
 * simulation, which is a far stronger claim than agreeing on a frame hash --
 * gametic advancing identically means every tic ran the same code path.
 */
__attribute__((export_name("dg_gametic")))
int32_t dg_gametic(void) { extern int gametic; return gametic; }
