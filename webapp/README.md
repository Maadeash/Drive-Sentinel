# DriveSentinel — technician web app

*Early fault detection for elevator drives.* The product presentation of the project:
sign-in, a monitored fleet, alerts with a Why? view, a live monitor, and the edge-hardware
story.

> **The fleet view is driven by recorded public datasets.** Every signal, status and alert
> comes from recordings of public laboratory test rigs, streamed through the project's real
> detection, fusion and alert code. No lift is connected.
> The FPGA accelerator has run on a PYNQ-Z2 and served this app's
> bearing stage (2026-09-24, `docs/claims_audit.md` §1.26–1.27). The lift names, sites and drive tags are fictional. The full mapping is
> [below](#where-the-fleet-comes-from). The pages present this as a working product, by
> design. The Engineering view (footer link) and this file are where the provenance lives.

**Frozen** as of 2026-09-22, re-frozen after the polish pass and the bearing-engine card (board connection). Only fixes, no new features.

## Start it

```bash
.venv/Scripts/python.exe -m webapp
```

Open <http://127.0.0.1:8000> and sign in with the **demo credentials**:

| Username | Password |
|---|---|
| `technician` | `demo` |

To set your own, use environment variables. They are read at every sign-in, and when
they are unset the defaults above apply:

```bash
DRIVESENTINEL_USER=operator DRIVESENTINEL_PASSWORD='choose-one' .venv/Scripts/python.exe -m webapp
```

In PowerShell: `$env:DRIVESENTINEL_USER="operator"; $env:DRIVESENTINEL_PASSWORD="choose-one"`.
Setting them replaces the default account. It does not add a second one.

Options: `--host 0.0.0.0` (so a phone on the same network can reach it), `--port 8000`,
`--pace 0.25` (seconds per step of the background drive stream).

Nothing beyond the project's `.venv` is needed (Starlette, uvicorn, itsdangerous and
python-multipart are already installed). Pages are plain HTML/CSS/JS. There is no build step,
no CDN and no external request.

On start the app streams every drive stage in the background through the same fusion and
alert code a drive would run. Alerts arrive over HTTP as they happen. The full stream takes
about 7½ minutes at the default pace, and the two winding stages are the long ones.
**Settings → Restart drive stream** starts it again.

## Sign-in

* Starlette `SessionMiddleware` with an itsdangerous-signed cookie `ds_session`: HttpOnly,
  SameSite=Lax, 8-hour lifetime.
* `webapp/auth.py` `AuthGate` sits in front of every route. Without a session, a page
  redirects to `/signin?next=…` and an API call returns 401. `next` accepts local paths only.
* Unauthenticated routes are `/signin`, `/static/*` (CSS, JS and icon only; page HTML lives in
  `webapp/pages/`, which is not under `/static`) and **`/api/alerts/ingest`**. Ingest is the
  device channel: a drive's outbox POSTs 21-byte alert packets there and has no browser
  session. It accepts only a packet of exactly the wire size. It is **not otherwise
  authenticated**. A real deployment would put a device key on it.
* The session secret is `DS_SESSION_SECRET` when set. Otherwise it is random per process, so
  a restart signs everyone out.
* Sign-out is on Settings. The header shows only the product name and the navigation.

## The pages

| Page | URL | What it shows |
|---|---|---|
| Sign in | `/signin` | Product mark and name; username, password, Sign in |
| **Fleet** | `/fleet` | KPI tiles, then one card per lift: status, the four stages, the latest alert |
| Lift health | `/lifts/<lift>` | The lift's four stages with rating, status, live signal strip and events |
| **Alerts** | `/alerts` | The feed, newest first. Filters for status, stage and lift, kept in the URL |
| Why? | `/alerts/<id>` | Signal, confidence, evidence build-up and action. **Technical details** are collapsed: rating, validation method in plain words, validation score, independent validation groups, the alarm-ready criterion, the raw packet |
| **Live monitor** | `/monitor?lift=…&stage=…` | Pick a lift and a drive stage. The page follows the server's stream -- the same one that raises the Alerts -- and asks for new observations every second: it shows what the stream has computed so far (`service.live_records`), never a precomputed copy. For the bearing it names the engine and where the signal processing ran ("FPGA (PYNQ-Z2) … signal processed on the board") |
| **Board live** | `/board` | A view of the bearing feed the whole app runs on (no controls). When the PYNQ-Z2 answers with its DSP front end, `webapp/boardlive.py` `BoardFeed` sends the raw 64 kHz current and vibration behind each bearing lift, in turn, 4 s at a time, to `POST /process`; the board filters, computes the spectra on its ARM cores, quantises and runs the FPGA, and those verdicts are every lift's bearing observations -- on Fleet, Alerts, Live monitor and the Why? pages. The page shows which lift the board is on, each lift's bearing status, the signal sent, the spectrum the board computed, the board's time per stage, the bearing alerts, and a check of every window against the laptop. Without the board, the bearing stage runs on its stored spectra and the page says why. `claims_audit.md` §1.29 |
| Hardware | `/hardware` | Verified bit-exact in simulation, synthesised for Zynq-7020, resources (core and full system), fixed cycles per verdict, V-1, and the board: "Running on PYNQ-Z2" with its record when `board/board_run.json` is committed (it is, since 2026-09-24), otherwise "Board deployment: next step" |
| About | `/about` | Short factual sections: overview, stages, alert levels, decision rule, edge hardware, roadmap |
| Settings | `/settings` | Data source ("Drive stream": Active; "Drive interface — PYNQ-Z2": **Connected · bearing engine** while the board is serving, otherwise Ready for connection), **Bearing engine** (active engine **Software** or **FPGA (PYNQ-Z2)**, board state, windows per engine, board address), alert delivery, account |
| Engineering view | `/engineering` | **Footer link only.** Metrics with protocol names, the pre-registered floor, the authority table, winding held-out figures and the session-only baseline, pooled leave-one-bearing-out, the KI05 miss, the order-resolution table. Content unchanged from the previous version |

Every page has its own URL. Back/forward and bookmarks work because the pages are real pages,
not a client-side router. Each page fades in on load. That is turned off under
`prefers-reduced-motion`. Old addresses (`/replay`, `/lift`, `/alert`) redirect.

**"Connected · bearing engine"** means the PYNQ-Z2 is computing every bearing verdict right
now; the data still comes from the drive stream, so the row is not the active source.
**"Ready for connection"** means this: the data-source seam (`drivesentinel/sources.py`,
the `DataSource` protocol) is what a board-side source would plug into, and the bitstream
exists (`board/`). The board-side source itself, `LiveSensorSource`, is a skeleton that
raises `NotImplementedError`.

**Bearing engine.** Every bearing window in the drive stream goes through
`drivesentinel/engines.py` `EngineRouter`. It uses the FPGA on a PYNQ-Z2 running
`board/server.py` when a board address is set and the board answers with build ID
`0x23eb56bf`. Otherwise it uses the bit-accurate software model, whose register values the
FPGA must reproduce. Fallback and recovery are automatic, and Settings shows which engine is
active. A verdict is labelled "FPGA (PYNQ-Z2)" only when the board produced it. The bearing
stage's status comes from that verdict -- the deployed network's -- whichever engine computes
it (the owner's decision of 2026-09-24; identical alerts either way, tested). On the recorded
data that verdict is **in-sample**, because the deployed network was trained on these bearings
(`claims_audit.md` §1.28). Set the address on Settings, or with
`DS_BOARD_ADDRESS=host:port`. **A PYNQ-Z2 served the web app on 2026-09-24**: every bearing
window, bit-exact against the software model (`claims_audit.md` §1.27).

## Where the fleet comes from

`webapp/fleet.py` `LIFTS` assigns each recorded unit to one stage slot of one lift. A test
asserts that every unit is used **exactly once** and in a slot of its own stage. Nothing is
duplicated to look busier and nothing is dropped. A slot with no unit shows NO DATA. Each
lift's stages come from **different** test rigs, because no public dataset covers a whole
drive. The engineering dashboard calls this COMPOSITE REPLAY.

| Lift | Supply | Inverter | Winding | Bearing |
|---|---|---|---|---|
| Lift 07 · Tower B | Thomas FILE 2 (phase lost running) | Bacha F2 (open circuit) | KAIST 1000 W inter-turn ramp | Paderborn KA04 (outer race) |
| Lift 03 · Tower A | Thomas FILE 5 (single-phasing at start) | — | KAIST 1000 W inter-coil ramp | Paderborn KI18 (inner race) |
| Lift 12 · Tower B | — | Bacha F3 (short circuit) | — | Paderborn K001 (healthy) |
| Lift 01 · Core 1 | — | Bacha F6 (over-temperature) | — | Paderborn KI05 (inner race; ALARM since 2026-09-24, **in-sample**) |

What the pages do not say, and where it is said instead:

* **Every bearing alert on this data is in-sample.** The bearing stage runs the deployed
  network, which was trained on all four demo bearings. Lift 01's bearing (KI05) raises an
  ALARM; a model that never saw KI05 misses it on 98 % of its windows. The Engineering view
  shows that fold, and the honest figure for a new bearing stays macro-F1 0.7852.
* **The winding type call is crossed on both units.** The inter-turn ramp is called
  inter-coil and the inter-coil ramp is called inter-turn. That is why a winding alert says
  only "Winding anomaly detected" (decision 2). The fault code still travels in the packet.
* **Three of four stages are ADVISORY** because they fail the pre-registered group floor.
  About presents that as the roadmap. The Engineering view shows the floor.

## Ratings and statuses

| Stage rating | Alert type it can raise |
|---|---|
| **ALARM-READY** (today: bearing) | ALARM — schedule inspection |
| **ADVISORY** (power supply, inverter, motor winding) | ADVISORY — early sign, check at next visit |

Status colours: ALARM red, ADVISORY amber, NORMAL green, NO DATA grey. **An ADVISORY stage
never raises an ALARM.** Two independent guards enforce this, both unchanged by this
revision. `fusion.BranchState` caps an INDICATIVE branch at Warning, and `alerts.build_alert`
raises rather than emit an ALARM without Fault authority. Both are tested.

## What each alert says (decisions of 2026-09-21)

| Stage | Headline | Where | How bad |
|---|---|---|---|
| Power supply | phase lost while running / started with a phase missing | Phase L1/L2/L3, from the collapsed current (measured) | running or stalled (measured) |
| Inverter | open circuit / short circuit / overheating | "Location: inverter power stage" | omitted |
| Motor winding | **"Winding anomaly detected"**, never a type | "Motor stator winding" | **omitted** (decision 1: the signal does not track severity, max \|ρ\| 0.34) |
| Bearing | outer-race / inner-race damage | Motor bearing | confidence |

A field a stage does not determine is **left out** rather than filled with a sentence
saying so.

## The alert pipeline, the data source, the packet

These are unchanged. See `drivesentinel/alerts.py` for the status-change emitter, the
21-byte packet (`<BHBBBBBbIII`) against 10,240 bytes for one model input window, and the
on-disk store-and-forward outbox with idempotent resend. See `drivesentinel/sources.py` for
`RecordedScenarioSource` (bearing units use the fold model that held that bearing out;
winding and inverter use held-out predictions only).

## Tests and screenshots

```bash
.venv/Scripts/python.exe -m pytest -q tests/test_webapp.py
```

These tests cover sign-in on every page and API, sign-out, the open-redirect guard, the cookie
flags, and the sign-in form field names. They check every user-facing page, script and API
response for the banned vocabulary and for claims of a connected lift or a board run. They
also cover winding headlines, inverter location, fleet placement, ADVISORY never ALARM, and
every number against its JSON. The engineering view is exempt from the vocabulary check.

```bash
.venv/Scripts/python.exe scripts/webapp/screenshots.py
```

With the app running and its stream finished, this writes every page at 1280 px and 390 px to
`docs/screenshots/`, including the sign-in page. It also runs back/forward navigation and
three Live monitor sessions (a bearing ALARM, an inverter ADVISORY and a winding advisory).
Per-shot console errors, horizontal overflow and a banned-word scan of the rendered text go to
`docs/screenshots/console.json`. It drives the Edge that ships with Windows over the DevTools
protocol. It gets its session by POSTing the demo credentials to `/signin` and handing the
cookie to the browser. It does not type a password into the page.
