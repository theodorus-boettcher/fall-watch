# Fall watch: handoff

One page for whoever runs this next. How to set it up is in the [README](README.md).

## What was set up

- **One room.** A Galaxy S5 hangs high on the bedroom wall, looking down at the room and the bed. It runs
  the Fall watch app 1.1 with the fall-v3 model. The motion gate is on, the preview is off, and it uses 4
  threads. The app is device owner, the phone's other apps are hidden, and it starts by itself after a reboot.
- **One gateway.** A laptop on the same Wi-Fi runs `fall-gateway` with one bed zone. Staff watch the live page.
- **Roboflow.** The project `fall-watch` (keypoint detection) holds the training data. Its version 4 is
  CAUCAFall and GMDCSA-24 plus 1,053 pictures of this room, with the room pictures given 3 times the weight
  in training. A second, box-only project was used for SAM 3 Auto Label.

## How well it works

- **In this room:** 8 of 8 acted floor falls raised an alert after 5 s, and 3 normal clips raised none.
  One of those clips is a fall onto the bed. A live test on 2026-10-22 gave the same: an alert about 5 s
  after lying down on the floor, and none on the bed.
- **On public clips from rooms and people it never trained on:** 16 of 17 falls on GMDCSA-24 and 10 of 10
  on CAUCAFall.
- **The room pictures were what made it work.** The same model trained only on public data caught 0 of the 8
  room falls. The public videos are filmed at body height, and this phone looks down from high on the wall.
  Seen from above, a person on the floor looks different.

## What fails, and why

- **Floor exercise looks like a fall.** Push-ups and sitting on the floor raised 4 false alarms on GMDCSA-24,
  up from 2 before the room training. Those clips are scored with a 1 s alert because they're short. The
  push-ups last 4 to 5 s, under the room's 5 s alert time, but the sitting clip lasts 8 s, so it might still
  alert. A care home would use about 30 s, which cuts these further.
- **One room, one person.** Everything was tested in one room with one person. A new room or a new camera
  position needs its own pictures and a new bed zone. Until it gets them, it may do as badly as the model
  trained only on public data did here.
- **No night test.** The S5's camera needs light. A real room needs a night light or an infrared camera.
- **The alert is only on the live page.** There's no sound and no message to a staff phone.
- **No login.** Anyone on the home network can open the live page and clear alerts. The phone sends plain
  HTTP, so it's for a closed network only.
- **Wi-Fi.** On a weak link, frames pile up on the phone, and the alerts come late, up to minutes. The live
  page shows "waiting on phone" in red when this happens. The fix is to pin the phone to the router's
  stronger radio at its final spot (see `deploy/provision.sh`). The pin is lost on reboot.
- **People get a new number when they come back.** The tracker only matches positions and movement. It
  can't recognise a person who leaves the picture and returns. That doesn't affect the alerts. A person
  the model misses for more than 3.5 s does restart their 5 s timer, though.
- **Someone lying very still may stop being found.** Their alert stays on until they're seen up again or
  staff clear it.
- **Heat.** With the preview on and someone moving, the phone reached 88 to 91 °C. Keep the preview off
  except while setting up. A 2014 battery should also be checked for swelling.

## Speed and power on the phone

- 0.2 to 0.33 s per frame. That's 3 to 5 frames a second while something moves, and 1 a second when
  the room is still.
- After 3 h 15 min with the motion gate on: 55 °C, no dropped frames, and the battery went from 25% to 23%
  while plugged into a laptop's USB. Without the motion gate it was 64 to 73 °C within an hour.
- The camera's own frame rate isn't lowered. That might save more heat, but it hasn't been measured.

## How to retrain for a new room

1. Turn the preview on and record `room-*` clips from the live page: standing, walking, sitting, in bed,
   and lying on the floor in several spots. Record a few `test-fall-*` and `test-*` clips apart, for testing.
2. `fall_data --room` labels them with the RF-DETR Keypoint teacher. Pictures it misses go to SAM 3 Auto
   Label with `roboflow_io --autolabel-room`, which costs Roboflow credits. Upload them and make a new
   version.
3. Train from fall-v3 (`train --base fall-v3.pt --repeat-room 2`, with the weights from the release). Then check it with `eval_falls --captures` on the test
   clips. Ship it only if it catches at least as many falls with no new false alarms.
4. Copy the model files into the app's assets, build it, and run `deploy/provision.sh`.

## Next steps for a real care home

- An alert time of about 30 s, and a message to the staff phone.
- A login on the gateway, and one gateway for many rooms.
- Tests with more people, more rooms, and at night.

## Licences

The code is AGPL-3.0, because it builds on Ultralytics YOLO26. The datasets and other parts are listed in
the [README](README.md#licences).
