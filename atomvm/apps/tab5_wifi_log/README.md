# tab5_wifi_log

Join your WiFi (credentials from `wifi_creds`, like the other apps) and serve the
application's log over HTTP, so you can watch the board from another machine. It does two
things at once:

- a **WiFi test** for the M5Stack Tab5: it powers the on-board ESP32-C6's rail (the pin nothing
  else drives — see the main README, "A switched board part on an expander pin") and joins your
  AP, reporting each attempt;
- a **log server** on port 8080: `GET /` in a browser, `GET /logs` as a Server-Sent Events
  stream.

The log lines are `listen_demo`'s: the same listener on the `m5stack_tab5` preset reports what
it hears, the command it got and the reply it spoke, and every one of those lines is byte for
byte what `listen_demo_responder:handle_info/2` prints — plus this app's own startup and WiFi
lines.

## 1. Configure

WiFi credentials come from `wifi_creds`, exactly like every other app here
([Example apps](../../README.md#example-apps)):

    atomvm/gen_wifi_creds.sh atomvm/apps/tab5_wifi_log

writes a git-ignored `src/wifi_creds.erl` from the git-ignored `stt/main/wifi_secrets.h`. Or,
by hand: `cp src/wifi_creds.erl.example src/wifi_creds.erl` and fill in your network. Without
it the app stops at startup with an `undef` from `wifi_creds:get/0`.

## 2. Build and flash

The firmware has to be a Tab5 build — the C6's SDIO pins are a build-time setting
(`esp32p4_pre_c6_tab5` for pre-v3.0 P4 silicon, `esp32p4_c6_tab5` otherwise; see
[Build and flash](../../README.md#build-and-flash)):

    cd atomvm
    IDF_PATH=~/esp/esp-idf-5.5.5 ./build.sh esp32p4_pre_c6_tab5
    esptool.py --chip esp32p4 -p /dev/ttyACM0 write_flash 0x0 \
        ~/build/atomvm-out-esp32p4_pre_c6_tab5/atomvm-babytalk.img

then the app (it went where every other app goes, `main.avm` at 0xA90000):

    cd atomvm/apps/tab5_wifi_log
    mix atomvm.packbeam
    esptool.py --chip esp32p4 -p /dev/ttyACM0 write_flash 0xA90000 tab5_wifi_log.avm

## 3. Read the log

The app prints the address it got, and then you can follow it from another machine:

    tab5_wifi_log: log on http://192.168.4.42:8080/ -- and GET /logs as a stream

    $ curl -N http://192.168.4.42:8080/logs
    data: tab5_wifi_log: C6 rail on (expander 0x44 pin 0) -> ok
    data: tab5_wifi_log: log on http://192.168.4.42:8080/ -- and GET /logs as a stream
    data: tab5_wifi_log: listening (listen_demo's log lines)
    data:   heard <<"wake up tomato face">> (score -20.9)
    data: AWAKE (score -21.0) -- listening for a command
    data: COMMAND <<"turn the light on">>
    data: SAID <<"You said: turn the light on">>

`curl -N` streams the events as they happen (`-N` turns off curl's buffering). A browser at
`http://192.168.4.42:8080/` shows the same log, including the last 100 lines first — everything the
app logged before you connected — then follows along. No log lines are retained while a client is
connected to receive them, if a client disconnects the missed logged lines will be saved (up to the
most recent 100) and sent the next time a client connects.

## What it logs

- `tab5_wifi_log: C6 rail on (expander 0x44 pin 0) -> ok` — the C6 powered, which has to
  happen in the first seconds after reset (below).
- `tab5_wifi_log: joining <ssid> (attempt N, up to T ms)` / `wifi up: ip ...` — the join, once
  per attempt; a failed attempt is retried every 15 s, so a late AP needs no reset.
- `tab5_wifi_log: log on http://<ip>:8080/ ...` — the server is up.
- `tab5_wifi_log: listening (listen_demo's log lines)` — the listener started.
- `  heard <text> (score <score>)` — each scored wake window.
- `AWAKE (score <score>) -- listening for a command` — the wake phrase was heard.
- `COMMAND <text>` — the transcribed message.
- `SAID <text>` — the reply was spoken (the sink answers a command exactly as
  `listen_demo_responder` does).
- `  <term>` — anything else the listener reports.

## If WiFi doesn't come up or has poor performance

- An `undef` error from `wifi_creds:get/0` means step 1 was skipped: there is no
  `src/wifi_creds.erl`.
- `network:wait_for_sta/2` failing with a timeout is usually the SSID/PSK or the AP's band: the
  C6 is 2.4 GHz only.
- If the link never comes up at all, look for `eh_sdio: ... card init failed` /
  `bring-up timed out`: the C6 isn't powered or isn't answering. This app drives the rail as its
  first act, but the C6 needs about a second to boot and esp_hosted only keeps trying for about
  five, so an app that starts late (or a rail that was never driven on this boot) can still miss
  that window.
- The console (115200 on the USB port) has the C6's own story under the app's lines. A C6 whose
  slave firmware doesn't match the host shows `esp-hosted fw versions: host=3.0.9 coprocessor=0.0.0`
  and `major version mismatch — OTA coprocessor from host`: on a Tab5 that is **not** fatal — the
  C6 still configures and connects the station (measured: `set_config iface=STA ssid="..."`
  accepted, events delivered, an address obtained), so treat it as the upgrade recommendation it
  is. What stops a join is a `StaDisconnected ... reason=N` line: reason 2 (AUTH_EXPIRE) came and
  went in our runs — attempts 1 and 2 failed, attempt 3 connected. A failure that never clears is
  usually the PSK, or an SSID the C6 can't see (it is 2.4 GHz only).  Everything over that link is
  slow and jittery while the old slave is in place: the console says
  `CP without SDIO SW_AGGR; compatible streaming mode enabled`, so the SDIO data path falls back to
  packet mode, and the listener is transcribing at the same time. Measured on a Tab5: 1-10s per
  small request, an occasional `Connection reset by peer`. A 3.x update of the firmware on on the
  C6 slave will fix the slowness and instability.
