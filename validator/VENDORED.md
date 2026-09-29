# Vendored validator code

`validator/modules/video_inconsistency/` holds an unmodified copy of these files from
[FLock-io/FLock-validator](https://github.com/FLock-io/FLock-validator), branch
`feat/video-inconsistency-task`, commit `ba9c152`:

- `issue_types.py`: the 10 issue types and the suite version
- `manifest.py`: label, decoy and clip models, and the time convention
- `errors.py`, `predictions.py`: detector output parsing, exactly as the validator does it
- `scoring.py`: the validator's scorer (mAP, localization, clip accuracy)
- `video_io.py`: H.264 encode and decode through `imageio-ffmpeg`
- `synthesis.py`: the clip generator used to build every official dataset and validation package

The trainer scripts use them to generate extra training data and to calibrate thresholds against
the real scorer. To update, copy the same files from a newer validator commit and bump the hash
above. Your submission must not import them, because the validator sandbox cannot read this
package.
