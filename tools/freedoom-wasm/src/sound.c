/* Freedoom's audio backend for ChittiOS: an 8-channel software mixer over the
 * WAD's own `DS*` lumps, submitted through the `audio_submit` primitive.
 *
 * Doom reaches audio through a `sound_module_t` vtable and a `music_module_t`,
 * both declared in `i_sound.h` and selected by `snd_sfxdevice`. With
 * `FEATURE_SOUND` undefined the module table is empty and the game runs silent
 * by a supported path, which is where the port started. This file is the module,
 * so nothing upstream is edited (VENDORING.md).
 *
 * Why the mixing happens *here* rather than in the kernel: Doom already knows
 * which sounds are playing, at what volume, and panned where — a kernel-side
 * mixer would be a second implementation of bookkeeping the game has already
 * done, plus sample-bank lifetimes to manage. `audio_submit` takes a finished
 * stream, exactly as `speech_pump` and the video player already feed
 * `sound::play_ch`.
 */

#include <stdint.h>
#include <string.h>

#include "doomtype.h"
#include "i_sound.h"
#include "deh_str.h"
#include "w_wad.h"
#include "z_zone.h"

/* On wasm these are the kernel's host imports. Natively they are provided by
 * `tools/doombench`, which is how the mixer is tested at all: a headless guest
 * has no sound device, so `host_audio_free()` returns 0, the mixer correctly
 * does nothing, and a boot can prove only that the module initialised. The
 * arithmetic — the DMX 16-byte skip, the fixed-point resample, the pan law, the
 * clamp — is checkable only off-hardware. Same reason `pngbench` and `h264diff`
 * exist. */
#ifdef __wasm__
#define IMPORT(n) __attribute__((import_module("chitti"), import_name(n)))
IMPORT("host_audio_submit") extern int32_t host_audio_submit(const void *pcm, int32_t frames, int32_t rate, int32_t ch);
IMPORT("host_audio_free")   extern int32_t host_audio_free(void);
IMPORT("host_log")          extern void    host_log(const char *p, int32_t l);
#else
extern int32_t host_audio_submit(const void *pcm, int32_t frames, int32_t rate, int32_t ch);
extern int32_t host_audio_free(void);
extern void    host_log(const char *p, int32_t l);
#endif

/* Doom's own default. `snd_channels` can differ but 8 is what it ships with and
 * what `S_Init` allocates for. */
#define NCHAN 8

/* Output rate. Doom's effects are 11025 Hz, so 22050 is one clean doubling — a
 * whole-number ratio keeps the resampler exact and avoids the shimmer a
 * fractional step puts on short, percussive samples. */
#define OUT_RATE 22050

/* One tic is 1000/35 = ~28.6 ms. Two tics of stereo headroom is enough to ride
 * out a slow frame without building the latency that makes a gunshot feel
 * detached from the trigger pull. */
#define MAX_FRAMES 2048

typedef struct {
    const uint8_t *data;   /* unsigned 8-bit samples, in the cached lump */
    uint32_t len;          /* sample count */
    uint32_t step;         /* 16.16 source-samples per output frame */
    uint32_t pos;          /* 16.16 position into `data` */
    int vol;               /* 0..127, Doom's scale */
    int sep;               /* 0..254, 128 = centre */
    boolean active;
} chan_t;

/* `i_sound.c` binds these as config variables under FEATURE_SOUND, and they were
 * defined in `i_sdlsound.c` — which we exclude. Something has to own them or the
 * link fails with two bare undefined symbols that say nothing about sound being
 * newly enabled. Both keep the SDL backend's defaults; we do our own resampling
 * with a fixed-point step, so neither is read by anything here. */
int use_libsamplerate = 0;
float libsamplerate_scale = 0.65f;

static chan_t chans[NCHAN];
static boolean sfx_prefix;
static int16_t mixbuf[MAX_FRAMES * 2];

/* ---- lump decoding ------------------------------------------------------- */

/* The DMX sound format, exactly as `i_sdlsound.c` reads it:
 *   u16 format (must be 3), u16 sample rate, u32 sample count, then samples.
 *
 * Two details that are not guessable and both come from that reference. The
 * library **skips 16 bytes at each end** of the sample data (reason unknown, but
 * every port does it, and not skipping them clicks at the start of every sound).
 * And a lump whose header claims more samples than the lump holds is *invalid*
 * rather than clamped — a length that disagrees with the file means the lump is
 * not what it says it is, and reading the remainder as audio produces a noise
 * burst that sounds like a driver fault.
 */
static boolean sfx_from_lump(int lumpnum, const uint8_t **out, uint32_t *count, uint32_t *rate) {
    if (lumpnum < 0) return false;
    int lumplen = W_LumpLength(lumpnum);
    if (lumplen < 8) return false;
    const uint8_t *lump = (const uint8_t *)W_CacheLumpNum(lumpnum, PU_STATIC);
    if (lump == NULL) return false;
    if (lump[0] != 0x03 || lump[1] != 0x00) return false;

    uint32_t r = (uint32_t)lump[2] | ((uint32_t)lump[3] << 8);
    uint32_t len = (uint32_t)lump[4] | ((uint32_t)lump[5] << 8)
                 | ((uint32_t)lump[6] << 16) | ((uint32_t)lump[7] << 24);
    if (len > (uint32_t)(lumplen - 8) || len <= 48) return false;
    if (r == 0) return false;

    *out = lump + 8 + 16;   /* skip the header, then DMX's leading 16 */
    *count = len - 32;      /* and its trailing 16 */
    *rate = r;
    return true;
}

/* ---- the sound module --------------------------------------------------- */

static boolean I_Chitti_InitSound(boolean use_sfx_prefix) {
    sfx_prefix = use_sfx_prefix;
    memset(chans, 0, sizeof chans);
    host_log("freedoom: audio up", 18);
    return true;
}

static void I_Chitti_ShutdownSound(void) {
    memset(chans, 0, sizeof chans);
}

/* Doom prefixes effect lumps with `DS`; Heretic and Hexen do not, which is what
 * `use_sfx_prefix` distinguishes. A `link` means this effect is an alias for
 * another, so the name has to be taken from the target. */
static int I_Chitti_GetSfxLumpNum(sfxinfo_t *sfx) {
    char buf[16];
    if (sfx->link != NULL) sfx = sfx->link;
    const char *name = DEH_String(sfx->name);
    unsigned n = 0;
    if (sfx_prefix) { buf[n++] = 'd'; buf[n++] = 's'; }
    for (unsigned i = 0; name[i] && n < sizeof buf - 1; i++) buf[n++] = name[i];
    buf[n] = 0;
    return W_GetNumForName(buf);
}

static int I_Chitti_StartSound(sfxinfo_t *sfxinfo, int channel, int vol, int sep) {
    if (channel < 0 || channel >= NCHAN) return -1;
    const uint8_t *data;
    uint32_t count, rate;
    if (!sfx_from_lump(sfxinfo->lumpnum, &data, &count, &rate)) return -1;

    chan_t *c = &chans[channel];
    c->data = data;
    c->len = count;
    /* 16.16 fixed point. At 11025 -> 22050 this is exactly 0x8000, i.e. every
     * source sample held for two output frames. */
    c->step = (uint32_t)(((uint64_t)rate << 16) / OUT_RATE);
    c->pos = 0;
    c->vol = vol;
    c->sep = sep;
    c->active = true;
    return channel;
}

static void I_Chitti_StopSound(int handle) {
    if (handle >= 0 && handle < NCHAN) chans[handle].active = false;
}

static boolean I_Chitti_SoundIsPlaying(int handle) {
    if (handle < 0 || handle >= NCHAN) return false;
    return chans[handle].active;
}

static void I_Chitti_UpdateSoundParams(int handle, int vol, int sep) {
    if (handle < 0 || handle >= NCHAN) return;
    chans[handle].vol = vol;
    chans[handle].sep = sep;
}

/* Called once per tic from `S_UpdateSounds` via `I_UpdateSound`.
 *
 * Pacing is the `speech_pump` loop: ask how much room the device has, produce
 * that much, submit. Never more — a queue this fills faster than it drains ends
 * up seconds of audio behind the picture, which is worse than a dropout.
 */
static void I_Chitti_UpdateSound(void) {
    int active = 0;
    for (int i = 0; i < NCHAN; i++) if (chans[i].active) active++;
    /* Nothing playing: submit nothing. Pushing silence every tic would cost an
     * audited primitive call 35 times a second to say "still quiet", and Doom's
     * effects are bursty — most tics have none. */
    if (active == 0) return;

    int free_samples = host_audio_free();
    int frames = free_samples / 2;              /* stereo */
    if (frames > MAX_FRAMES) frames = MAX_FRAMES;

    /* **A machine with no sound device must still finish its sounds.**
     *
     * `host_audio_free()` returns 0 when nothing is up — a headless guest, or a
     * backend that failed to bind. Returning early there leaves every channel
     * `active` forever: positions never advance, nothing ever expires, and after
     * eight effects `s_sound.c` believes all eight channels are still busy and
     * stops starting new ones. The game's own state would then depend on whether
     * a speaker exists, which is the wrong thing to make it depend on.
     *
     * So when there is nowhere to submit, advance the channels by one tic anyway
     * and retire the finished ones. Silent, but correct.
     */
    if (frames <= 0) {
        for (int ci = 0; ci < NCHAN; ci++) {
            chan_t *c = &chans[ci];
            if (!c->active) continue;
            c->pos += c->step * (uint32_t)(OUT_RATE / 35);
            if ((c->pos >> 16) >= c->len) c->active = false;
        }
        return;
    }

    memset(mixbuf, 0, (unsigned)frames * 2 * sizeof(int16_t));

    for (int ci = 0; ci < NCHAN; ci++) {
        chan_t *c = &chans[ci];
        if (!c->active) continue;

        /* Doom's `sep` is 0..254 with 128 centre. Constant-sum panning: the two
         * gains add to the same total at every position, so panning a sound does
         * not change how loud it is — which a naive linear pan does, making
         * anything centred sound quieter than the same thing to one side. */
        int left = 254 - c->sep;
        int right = c->sep;
        if (left < 0) left = 0;
        if (right < 0) right = 0;

        for (int f = 0; f < frames; f++) {
            uint32_t idx = c->pos >> 16;
            if (idx >= c->len) { c->active = false; break; }
            /* Unsigned 8-bit, silence at 128. Centre first, then widen. */
            int32_t s = ((int32_t)c->data[idx] - 128) << 7;
            s = (s * c->vol) / 127;
            int32_t l = (s * left) / 254;
            int32_t r = (s * right) / 254;
            int32_t ol = mixbuf[f * 2] + l;
            int32_t orr = mixbuf[f * 2 + 1] + r;
            /* Clamp rather than wrap: eight channels can exceed the range, and a
             * wrapped sample is a loud click where a clipped one is merely loud. */
            if (ol > 32767) ol = 32767; else if (ol < -32768) ol = -32768;
            if (orr > 32767) orr = 32767; else if (orr < -32768) orr = -32768;
            mixbuf[f * 2] = (int16_t)ol;
            mixbuf[f * 2 + 1] = (int16_t)orr;
            c->pos += c->step;
        }
    }

    host_audio_submit(mixbuf, frames, OUT_RATE, 2);
}

/* We parse on demand and the lump cache does the rest, so there is nothing to
 * precache. Kept as a no-op rather than omitted because the vtable slot is
 * called unconditionally. */
static void I_Chitti_CacheSounds(sfxinfo_t *sounds, int num_sounds) {
    (void)sounds; (void)num_sounds;
}

static snddevice_t sound_devices[] = {
    SNDDEVICE_SB, SNDDEVICE_PAS, SNDDEVICE_GUS,
    SNDDEVICE_WAVEBLASTER, SNDDEVICE_SOUNDCANVAS, SNDDEVICE_AWE32,
};

sound_module_t DG_sound_module = {
    sound_devices,
    sizeof(sound_devices) / sizeof(*sound_devices),
    I_Chitti_InitSound,
    I_Chitti_ShutdownSound,
    I_Chitti_GetSfxLumpNum,
    I_Chitti_UpdateSound,
    I_Chitti_UpdateSoundParams,
    I_Chitti_StartSound,
    I_Chitti_StopSound,
    I_Chitti_SoundIsPlaying,
    I_Chitti_CacheSounds,
};

/* Test hooks. The module is reached through its vtable in the game; the harness
 * needs the same entry points without going through `S_StartSound`. */
int dgsnd_out_rate(void) { return OUT_RATE; }
int dgsnd_channels(void) { return NCHAN; }
const int16_t *dgsnd_mixbuf(void) { return mixbuf; }

/* ---- music -------------------------------------------------------------- */

/* **A deliberate no-op, for now.**
 *
 * Doom's music is MUS: a compact event format written for the OPL2, which needs
 * either FM synthesis or a General MIDI instrument bank to sound like itself.
 * `mus2mid.c` is vendored and would convert it to MIDI, but MIDI still needs a
 * synth, and a synth needs instruments — that is a decoder-sized project of its
 * own, not a few lines beside the effects mixer.
 *
 * The alternative would be square-wave oscillators, which produce a recognisable
 * chiptune rendition of tunes people know very well. That may be worth doing and
 * is easy to bolt onto the mixer above — but it is a *different* sound from the
 * one the WAD describes, so it is a decision to take deliberately rather than a
 * gap to fill quietly.
 *
 * Registering as a working module and playing nothing is the honest shape: Doom
 * asks whether music is available and behaves correctly either way. Returning a
 * non-NULL handle matters — `S_ChangeMusic` treats NULL as a failure and retries.
 */
static boolean I_Chitti_InitMusic(void) { return true; }
static void I_Chitti_ShutdownMusic(void) {}
static void I_Chitti_SetMusicVolume(int volume) { (void)volume; }
static void I_Chitti_PauseSong(void) {}
static void I_Chitti_ResumeSong(void) {}
static void I_Chitti_UnRegisterSong(void *handle) { (void)handle; }
static void I_Chitti_PlaySong(void *handle, boolean looping) { (void)handle; (void)looping; }
static void I_Chitti_StopSong(void) {}
static boolean I_Chitti_MusicIsPlaying(void) { return false; }
static void I_Chitti_PollMusic(void) {}

static void *I_Chitti_RegisterSong(void *data, int len) {
    (void)data; (void)len;
    /* A stable non-NULL token. Never dereferenced by us, and Doom only ever
     * hands it back to the calls above. */
    static int token = 1;
    return &token;
}

static snddevice_t music_devices[] = {
    SNDDEVICE_PAS, SNDDEVICE_GUS, SNDDEVICE_WAVEBLASTER,
    SNDDEVICE_SOUNDCANVAS, SNDDEVICE_GENMIDI, SNDDEVICE_AWE32,
};

music_module_t DG_music_module = {
    music_devices,
    sizeof(music_devices) / sizeof(*music_devices),
    I_Chitti_InitMusic,
    I_Chitti_ShutdownMusic,
    I_Chitti_SetMusicVolume,
    I_Chitti_PauseSong,
    I_Chitti_ResumeSong,
    I_Chitti_RegisterSong,
    I_Chitti_UnRegisterSong,
    I_Chitti_PlaySong,
    I_Chitti_StopSong,
    I_Chitti_MusicIsPlaying,
    I_Chitti_PollMusic,
};
