// The board's microphone and speaker, for BabyTalk's AtomVM NIFs: the only hardware-specific
// part of BabyTalk. board_audio.c implements it for boards built from ES7210/ES8311 codecs,
// with the pins, I2C addresses and amp/rail switches chosen at run time
// (board_audio_config.h, babytalk:audio_config/1). A board with other audio chips implements
// these functions in its own component and selects CONFIG_BABYTALK_BOARD_CUSTOM (see Kconfig).
//
// The contract: 16 kHz mono signed 16-bit capture; mono playback at any rate the board can
// do (8..48 kHz). The mic and the speaker may share one I2S port, so the NIFs never overlap
// them: rx_start .. rx_stop and play take turns. All calls come from one FreeRTOS task
// (BabyTalk's audio task), so implementations need not be thread-safe -- but if the board's
// I2C bus is shared with other drivers, use that bus's handle rather than creating one.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bring the audio chips up (rails on, amp off). Once; later calls return 0 at once.
// mic_gain: the codec's own gain step, or -1 for the board's default. Negative on failure
// (board_audio.c: -1 I2C bus, -2 rail/amp switch, -3 mic codec, -4 speaker codec).
int board_audio_init(int mic_gain);

// Capture at 16 kHz; discards the first 200 ms (ADC/filter settling).
int board_audio_rx_start(void);
int board_audio_rx_read(int16_t *mono, int frames);  // fills all frames, blocking: 0, or negative on failure
void board_audio_rx_stop(void);                      // frees the port

// Play mono PCM at `rate` Hz through the speaker, blocking. volume 0..100 (DAC; 75 = 0 dB).
// The amp is on only while playing (after the stream starts: no turn-on pop).
int board_audio_play(const int16_t *mono, int n, int rate, int volume);
// DMA underruns during the last play (each one a short gap of silence).
int board_audio_underruns(void);

// Drive one pin (0..7, level 0/1) of a PI4IOE5V6408 IO expander, independently of the audio
// configuration -- for a board part that is a switched expander pin, not an amp or a rail.
// Brings the I2C bus up if the audio board hasn't (the first caller's pins win, and a later
// board bring-up then shares that bus). 0, else -1 (bus, pins, address or pin) or -2 (nothing
// answered). Not present with CONFIG_BABYTALK_BOARD_CUSTOM.
int board_audio_pin_set(int i2c_port, int sda, int scl, int hz, uint8_t addr, int pin, int level);

#ifdef __cplusplus
}
#endif
