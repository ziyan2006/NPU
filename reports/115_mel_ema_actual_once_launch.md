# Actual once-only fixed500 EMA launch

Actual launch occurred2026-10-05T04:32:06Z (12:32local), after225 metadata
preflight passed11 actual gates/2914 bindings/freeVRAM5452MiB/disk>=12GiB/no
other scoped Python task. SharedGPU utilization28% was authorized, not a fault.
224 ran with WindowsPowerShell5.1; foreground session25930 completed native0
(start197a4f/completion57cab7). That is launch success, NOT training EXIT0.

Actual WMI helper79852 has WmiPrvSE.exe parent9480 and in_job_object=false.
Native supervisor shim123036 / actual115300 spawned training worker shim126552
/ actual119804. The shims are one chain, not duplicate training. No user task
was terminated or system setting changed. Activation SHA:
e0c8d20fe23d936712f844198b8faccd5d07d204ffb56907d4bef75223fafe21.

Launch receipt:
results/mel_ema_single_trajectory_launch_20261005/detached_launch_20261005_043206_1397102.json
SHA7499410941d9a2665ef353c4d89e8a440ed3f3eefac3ebddb92457df0e310ede.
Actual WMI request receipt SHA:
0a0116be88e5d89509b6c0c886cca095a8f854e24ad56100d6011606e0015cf1.
New launcher224 SHA:
d27f45989dee1f0283f29d34c51788b3ca35d74d0d4fa4c50c399ae8075832d0.
Complete plan/activation builder225 SHA:
ae7d23a5bfff5c28cd7398af402eb188e96400e5c05dc1ce568ede4e98941ddb.

Real run directory: results/mel_ema_single_trajectory_20261005.
At04:40:07Z the whole31/177 DEV input preparation finished and the worker
saved NONRELEASE_EMA_step_4500.pt SHA:
49ecc9e0cbce4cb91d681673f94add1a565b712e48566afc499042bb0eccc144.
This is the initial full CUDA restore container, NOT a new training update.
Status entered training, errornull, additional_steps0. At04:41:07Z the
native journal reached sequence586 and the first actual TRAIN draw was in
progress. At04:41:24.487008Z the worker really committed4501:6 raw forwards
started/completed,6 backwards completed and1 Adam started/completed;errornull.
The corresponding updates.jsonl row records4501/LR6.28001045799771e-5.
The first new formal update is now observed, not merely launch success.
Latest full disk checkpoint remains4500 until the planned new stage. No new
DEV score or final exit/completion is claimed.

At2026-10-05T04:48:39Z a read-only follow-up observed updates.jsonl step4507,
LR6.271570764446376e-5 and formal_student_update=true. Actual worker119804
and supervisor115300 remain live in the same launch chain; D free24.271904GiB.
Periodic run_status still records4501/errornull, while the append-only update
journal supplies the newer committed step. This is7/500 formal updates, not a
new checkpoint or completed stage. No matching detached exit is present yet.

During open-file writes, directory Length/LastWriteTime snapshots sometimes
remained0/old even though directly reading the journal showed native events
and accepted decoder exits. Monitor actual file content and dynamic PID/CPU,
not a single directory-length or unchanged step snapshot.

Budget remains one raw/original Adam500 updates4501..5000, hard5000 and full
raw/EMA DEV pairs4750/5000. No EMA feedback, historical rerun, new dataset,
teacher inference, old stop edit, qualification reset or release. On failure
retain actual exposure/last full checkpoint, do not restart automatically.
Completion requires the full budget, both full stages, matching genuine
detached_exit0 and no live worker before an independent final review.
NONRELEASE/listening/independent real acceptance remain PENDING. The observed
automation remains PAUSED; manual launch did not silently resume it.
