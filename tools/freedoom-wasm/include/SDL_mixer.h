/* Stub. `i_sound.c` includes <SDL_mixer.h> under FEATURE_SOUND but never calls a
 * single `Mix_*` function — the only other SDL references in that file are
 * comments about historical SDL_mixer bugs. So enabling sound needs this header
 * to exist and nothing more.
 *
 * A stub rather than pretending to be DJGPP (the other branch of that `#if`),
 * because we control the include path and lying about the platform would change
 * unrelated code paths. And a stub rather than a patch to `i_sound.c`, per
 * VENDORING.md: upstream is never edited. */
#ifndef CHITTI_SDL_MIXER_STUB_H
#define CHITTI_SDL_MIXER_STUB_H
#endif
