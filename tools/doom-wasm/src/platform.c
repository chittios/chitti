/* The ChittiOS `doomgeneric` platform layer.
 *
 * The bench's platform (`tools/doombench/src/platform.c`) proved the game builds,
 * runs and is bit-exact; this is the same six functions wired to the OS instead
 * of to a harness. Everything OS-facing goes through the host imports Phase 1
 * added — no new authority, and nothing here touches a device.
 */

#include <stdint.h>
#include <string.h>

#include "doomgeneric.h"
#include "doomkeys.h"
#include "i_video.h"
#include "doomstat.h"

/* ---- host imports (see kernel/src/agent/wasm_rt.rs) ---------------------- */

#define IMPORT(n) __attribute__((import_module("chitti"), import_name(n)))

IMPORT("host_ui_present")  extern int32_t host_ui_present(const void *px, int32_t len, int32_t w, int32_t h);
IMPORT("host_surface_size") extern int64_t host_surface_size(void);
IMPORT("host_keys_held")   extern int32_t host_keys_held(void *out, int32_t len);
IMPORT("host_now_ms")      extern int64_t host_now_ms(void);
IMPORT("host_fs_read")     extern int32_t host_fs_read(const char *p, int32_t pl, void *out, int32_t cap);
IMPORT("host_fs_exists")   extern int32_t host_fs_exists(const char *p, int32_t pl);
IMPORT("host_home")        extern int32_t host_home(void *out, int32_t cap);
IMPORT("host_log")         extern void    host_log(const char *p, int32_t l);
IMPORT("host_audio_submit") extern int32_t host_audio_submit(const void *pcm, int32_t frames, int32_t rate, int32_t ch);
IMPORT("host_audio_free")  extern int32_t host_audio_free(void);

static void logs(const char *s) { host_log(s, (int32_t)strlen(s)); }

/* wasi-libc has no `system()`, and `i_system.c` references it for the external
 * music-player option (`snd_musiccmd`). Stubbed here rather than upstream: it is a
 * gap in the target's libc, not a defect in Doom, and there is no shell on
 * ChittiOS to run a command with either. -1 is what a real `system()` reports when
 * it cannot spawn, which is the honest answer on both counts. */
int system(const char *cmd) { (void)cmd; return -1; }

/* ---- frame ---------------------------------------------------------------
 *
 * Doom renders 8-bit paletted (CMAP256). The palette is applied here rather than
 * in the kernel: `ui_present` takes one pixel format and nothing else, so a
 * palette would have meant either a second primitive to upload it or a format
 * enum threaded through the gate. Expanding 64000 pixels through a 256-entry LUT
 * costs a rounding error against the ~2.3 M fuel a frame already spends.
 */
/* Under CMAP256 there is no palette callback: `i_video.c` exposes `colors[256]`
 * and sets `palette_changed` when Doom swaps it (damage flashes, item pickups,
 * the invulnerability sphere). Both are declared in `i_video.h`. The LUT is
 * rebuilt only when that flag says to — a palette swap is rare and rebuilding it
 * every frame would be 256 pointless conversions per frame forever. */
static uint32_t palette[256];
static uint32_t framebuf[DOOMGENERIC_RESX * DOOMGENERIC_RESY];

void DG_Init(void) {
    /* Doom sets a palette during startup, before the first frame; seed the LUT so
     * a frame drawn before the first swap is not solid black. */
    palette_changed = true;
}

void DG_DrawFrame(void) {
    if (palette_changed) {
        for (int i = 0; i < 256; i++)
            palette[i] = ((uint32_t)colors[i].r << 16)
                       | ((uint32_t)colors[i].g << 8)
                       | (uint32_t)colors[i].b;
        palette_changed = false;
    }
    const uint8_t *src = (const uint8_t *)DG_ScreenBuffer;
    const int n = DOOMGENERIC_RESX * DOOMGENERIC_RESY;
    for (int i = 0; i < n; i++) framebuf[i] = palette[src[i]];
    host_ui_present(framebuf, (int32_t)sizeof framebuf, DOOMGENERIC_RESX, DOOMGENERIC_RESY);
}

/* ---- time ----------------------------------------------------------------
 *
 * **Must return immediately.** `TryRunTics` waits for the next tic in a
 * `{ I_Sleep(1); NetUpdate(); }` loop, so a blocking sleep here spins inside one
 * guest call — burning the frame's fuel and never yielding to `upkeep()`, which
 * freezes the clock, the mouse and the net stack. That is a frozen *shell*, not a
 * frozen game. The frame pump owns pacing; this is the finding doombench cost a
 * hung process to learn.
 */
void DG_SleepMs(uint32_t ms) { (void)ms; }

uint32_t DG_GetTicksMs(void) { return (uint32_t)host_now_ms(); }

void DG_SetWindowTitle(const char *t) { (void)t; }

/* ---- input ---------------------------------------------------------------
 *
 * The OS reports *held* state; `DG_GetKey` wants *edges*. So each frame we take a
 * snapshot and diff it against the previous one, which is also exactly what makes
 * held-key movement work (a byte stream cannot express W and A held together —
 * software typematic replaces the held key rather than adding to it).
 *
 * A press and release inside one frame is missed. At 28 ms that is not reachable
 * by a human, and the alternative — an edge queue — needs ordering and overflow
 * policy for a case that cannot occur.
 */
static uint32_t held[8], prev_held[8];
static struct { int pressed; unsigned char key; } kq[64];
static int kq_head, kq_tail;

/* HID usage -> Doom key. Only what the game binds; anything else is ignored so a
 * keystroke meant for the shell is not swallowed. */
static unsigned char doom_key_for(int usage) {
    switch (usage) {
        case 0x52: return KEY_UPARROW;
        case 0x51: return KEY_DOWNARROW;
        case 0x50: return KEY_LEFTARROW;
        case 0x4f: return KEY_RIGHTARROW;
        case 0x1a: return KEY_UPARROW;      /* w */
        case 0x16: return KEY_DOWNARROW;    /* s */
        case 0x04: return KEY_STRAFE_L;     /* a */
        case 0x07: return KEY_STRAFE_R;     /* d */
        case 0x2c: return KEY_USE;          /* space */
        case 0xe0: case 0xe4: return KEY_FIRE;   /* ctrl */
        case 0xe1: case 0xe5: return KEY_RSHIFT; /* shift = run */
        case 0x28: return KEY_ENTER;
        case 0x29: return KEY_ESCAPE;
        case 0x2b: return KEY_TAB;
        case 0x1e: return '1';
        case 0x1f: return '2';
        case 0x20: return '3';
        case 0x21: return '4';
        case 0x22: return '5';
        case 0x23: return '6';
        case 0x24: return '7';
        default:   return 0;
    }
}

static void push_key(int pressed, unsigned char k) {
    int next = (kq_head + 1) % 64;
    if (next == kq_tail) return;   /* full: drop, never overwrite the oldest */
    kq[kq_head].pressed = pressed;
    kq[kq_head].key = k;
    kq_head = next;
}

static void poll_input(void) {
    if (host_keys_held(held, (int32_t)sizeof held) != (int32_t)sizeof held) return;
    for (int w = 0; w < 8; w++) {
        uint32_t diff = held[w] ^ prev_held[w];
        while (diff) {
            int b = __builtin_ctz(diff);
            diff &= diff - 1;
            int usage = w * 32 + b;
            unsigned char k = doom_key_for(usage);
            if (k) push_key((held[w] >> b) & 1, k);
        }
        prev_held[w] = held[w];
    }
}

int DG_GetKey(int *pressed, unsigned char *key) {
    if (kq_head == kq_tail) return 0;
    *pressed = kq[kq_tail].pressed;
    *key = kq[kq_tail].key;
    kq_tail = (kq_tail + 1) % 64;
    return 1;
}

/* ---- the package-UI export surface ---------------------------------------
 *
 * The ordinary app ABI: `(args_ptr, args_len) -> (ptr << 32) | len`, with
 * `chitti_alloc` for host-written arguments. No new binding, no new agent kind —
 * this is the same contract snake and chess use, which is why nothing downstream
 * needed changing.
 */

/* A small bump arena for the host's argument strings and our replies. Reset per
 * call, exactly as `tools/apps-wasm/src/guest.rs` does: guest statics are the
 * state, the heap is scratch. Doom's own allocations go through its zone
 * allocator over malloc and are untouched by this. */
static uint8_t argheap[4096];
static uint32_t argheap_at;

__attribute__((export_name("chitti_alloc")))
void *chitti_alloc(int32_t n) {
    argheap_at = 0;                       /* per-call reset */
    if (n < 0 || (uint32_t)n > sizeof argheap) return 0;
    argheap_at = (uint32_t)n;
    return argheap;
}

static int64_t reply(const char *s) {
    uint32_t n = (uint32_t)strlen(s);
    if (n > sizeof argheap) n = sizeof argheap;
    memcpy(argheap, s, n);
    return ((int64_t)(uint32_t)(uintptr_t)argheap << 32) | n;
}

/* Where the WAD is looked for, in order.
 *
 * The bundled Freedoom in this agent's own `assets/` comes first, so the game
 * works on a fresh boot with no network and nothing to fetch. `/downloads` is
 * searched after it so someone who owns the original game data can drop
 * `doom1.wad` or `doom.wad` there and have it win over the bundled one — a real
 * IWAD should take precedence over the replacement. */
static char wad_paths[4][160];
static int wad_path_count;

static void build_wad_paths(void) {
    char home[96];
    int hn = host_home(home, (int32_t)sizeof home - 1);
    if (hn < 0) hn = 0;
    home[hn] = 0;

    wad_path_count = 0;
    /* Owned game data first, then the bundled replacement. */
    static const char *const downloads[] = {
        "/downloads/doom.wad", "/downloads/doom1.wad", "/downloads/freedoom1.wad",
    };
    for (unsigned i = 0; i < sizeof downloads / sizeof *downloads; i++) {
        unsigned n = (unsigned)strlen(downloads[i]);
        if (n >= sizeof wad_paths[0]) continue;
        memcpy(wad_paths[wad_path_count], downloads[i], n + 1);
        wad_path_count++;
    }
    if (hn > 0 && wad_path_count < 4) {
        char *d = wad_paths[wad_path_count];
        unsigned n = 0;
        for (int i = 0; i < hn && n < sizeof wad_paths[0] - 24; i++) d[n++] = home[i];
        const char *tail = "/assets/freedoom1.wad";
        for (unsigned i = 0; tail[i]; i++) d[n++] = tail[i];
        d[n] = 0;
        wad_path_count++;
    }
}

/* Doom needs the WAD resident: `w_file_memory.c` serves every lump from this
 * buffer via `wad_file_t::mapped`, so nothing streams across the boundary. */
static uint8_t wad[40u << 20];
extern void dg_set_wad(const uint8_t *base, unsigned int len);
static char iwad_path[128];

static int started;

__attribute__((export_name("doom_start")))
int64_t doom_start(int32_t p, int32_t n) {
    (void)p; (void)n;
    if (started) return reply("ok");

    build_wad_paths();
    const char *found = 0;
    int32_t got = 0;
    for (int i = 0; i < wad_path_count; i++) {
        const char *path = wad_paths[i];
        int32_t pl = (int32_t)strlen(path);
        if (!host_fs_exists(path, pl)) continue;
        got = host_fs_read(path, pl, wad, (int32_t)sizeof wad);
        /* `host_fs_read` returns the file's **full** length and writes
         * min(len, cap) — so a value over our capacity means the WAD is real but
         * does not fit, which is a different problem from not finding one and is
         * reported as such. */
        if (got > (int32_t)sizeof wad) {
            return reply("ask:The WAD I found is larger than the 40 MB I can hold. Tell the user.");
        }
        if (got > 0) { found = path; break; }
    }
    if (!found) {
        /* The bundled WAD is written into this agent's home at install, so this
         * should be unreachable. It means the asset did not land — worth saying
         * plainly rather than blaming the user for not fetching something that
         * ships with the OS. */
        logs("doom: no WAD found, including the bundled one in my own assets");
        return reply(
            "ask:I could not find a WAD, not even the Freedoom one bundled in my own "
            "assets folder — so my install may be incomplete. Tell the user they can "
            "also drop doom.wad or doom1.wad into /downloads/.");
    }

    dg_set_wad(wad, (unsigned)got);
    /* Doom picks its *game mode* from the IWAD's filename, so the path is needed
     * even though the bytes are already in memory. Without it: "Game mode
     * indeterminate", and it refuses to start. */
    unsigned pl = (unsigned)strlen(found);
    if (pl >= sizeof iwad_path) pl = sizeof iwad_path - 1;
    memcpy(iwad_path, found, pl);
    iwad_path[pl] = 0;

    static char a0[] = "doom";
    static char a1[] = "-iwad";
    char *argv[] = {a0, a1, iwad_path, 0};
    doomgeneric_Create(3, argv);
    started = 1;
    logs("doom: started");
    return reply("ok");
}

__attribute__((export_name("tick")))
int64_t tick(int32_t p, int32_t n) {
    (void)p; (void)n;
    if (!started) return reply("ok");
    poll_input();
    /* Suppress the screen-melt wipe. `D_Display` takes the wipe branch when
     * `gamestate != wipegamestate`, and that branch is a **blocking spin loop**:
     *
     *     do { do { nowtime = I_GetTime(); tics = nowtime - wipestart;
     *               I_Sleep(1); } while (tics <= 0);
     *          ... I_FinishUpdate(); } while (!done);
     *
     * It runs the whole ~1 s melt inside a single `doomgeneric_Tick`, spinning on
     * the clock thousands of times per tic. That is exactly the pathology
     * `DG_SleepMs` is written to avoid, and here it is worse: one guest call that
     * neither returns nor yields, so it exhausts the frame's fuel *and* freezes
     * `upkeep()` — the clock, the mouse and the net stack — for a second.
     *
     * Holding the two equal keeps Doom on its normal per-frame path. The cost is
     * a cosmetic transition; the alternative is a frozen shell. Done here rather
     * than by editing `d_main.c`, per VENDORING.md.
     */
    wipegamestate = gamestate;
    doomgeneric_Tick();
    return reply("ok");
}
