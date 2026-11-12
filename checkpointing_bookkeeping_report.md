# Parallel HDF5 checkpointing: population bookkeeping and synchronization fixes

Branch: `maintenance/checkpointing` (upstream `Helmholtz-AI-Energy/propulate`, base commit `31d285b`).
Scope: debugging and stabilizing the population bookkeeping that the new parallel HDF5 checkpointing introduced
(migration, pollination, intra-island synchronization, restart from checkpoint), plus the tests covering it.

## 1. Summary

The intermittent "workers do not end up with identical, complete populations" failures had several root causes of
one class: the bookkeeping result depended on the order in which independent MPI messages arrived, or on the state
of whichever *copy* of an individual a worker happened to hold. On top of that, a restart from a checkpoint could
breed an already finished generation a second time (reusing an identity and overwriting its checkpoint row), the
end-of-run synchronization relied on `iprobe` after a barrier (which MPI does not guarantee to see in-flight
messages), and the test suite's population comparisons were vacuous.

The fix expresses all bookkeeping as events that are applied exactly once to a population entry created by
whichever message mentions the individual first, makes the "responsible worker" a pure function of identity and
island, drains outstanding messages by exchanged message counts, and derives the restart generation from the
checkpoint file rather than from the population. Repeated runs went from 0/10 passing (original code, migration
enabled) to 30/30, with a 12-rank / 3-island stress scenario passing 10/10.

## 2. Conceptual model and invariants

| Concept | Invariant |
|---|---|
| Identity | `(birth island, birth island rank, generation)`. It is the key of `Propulator.population` and the individual's slot `f[island][rank][generation]` in the checkpoint. It never changes; every copy that travels between workers or islands keeps it. |
| Location | Tracked separately from identity: in the checkpoint by `active_on_island[generation, island]`, locally by the `active` field of the worker's population entry. |
| Events | A worker learns about an individual through independent messages: it was bred on this island (`INDIVIDUAL_TAG`), it migrated in (`MIGRATION_TAG`), it migrated out or was replaced (`SYNCHRONIZATION_TAG`). MPI orders messages only per (source, destination); messages about the same individual from different sources arrive in any order. |
| Bookkeeping | `_upsert_individual` creates the entry (with `active = 0`) on the first message mentioning the identity, or completes an unevaluated placeholder loaded from a checkpoint; the caller then applies its event. Every event is applied exactly once. |
| Migrator | `active = (+1 if bred here) + (+1 per immigration) + (−1 per emigration)`. The operations commute, so the final value is order-independent (it may transiently be 2 or −1 while messages are outstanding). The worker treats the individual as active while `active >= 1`; at quiescence `active ∈ {0, 1}`. |
| Pollinator | Breeding and arrival of copies are idempotent (activate only when the entry is created); replacement is final (`active = 0`, a later copy does not reactivate). The responsible worker replaces one individual per immigrant that was *new* to the island, so an island holds exactly as many active individuals as it bred. Any worker may send pollen copies of any active individual; only replacement is uniquely assigned. |
| Responsibility | `_responsible_rank(ind)`: birth rank on the birth island, `generation % island_size` elsewhere. Pure function → all workers of an island agree regardless of which copy they hold. Only this worker emigrates/replaces the individual on its island, and only it writes the individual's activity bit *of its own island* into the checkpoint. (With real migration the sender additionally sets the target island's bit when it sends; see §8 for the ordering assumption this makes.) |
| Checkpoint | At quiescence, the set of active identities on an island equals the set of rows with `active_on_island[:, island] == 1` (checked by `population_consistency_check`). An unevaluated row (NaN loss) lives on its birth island only. Multi-cell transitions (migration: target bit on, source bit off; pollination: immigrant on, replaced off) are ordered so that an interruption in between leaves an extra active copy, never a lost individual. |
| End of run | `propulate()` ends with `_final_synchronization`: per tag, workers exchange how many messages they sent to each other (alltoall) and blocking-receive exactly that many, in rounds (bred → migration → notices, since processing immigrants can generate notices). Afterwards all workers of an island hold the same active population. |
| Restart | The next generation to breed is `generations[rank] + 1` if that row's loss is set, else `generations[rank]` (the placeholder is evaluated first). This does not depend on whether the individual is still in the population. |

## 3. Root causes and the sequences that triggered them

1. **Intra-island receive overwrote existing entries** (`_receive_intra_island_individuals`).
   Sequence: worker C breeds X at generation g and `isend`s it to A and B. A receives X, emigrates it to island 1
   (sends A→B "deactivate X"); island 1's responsible worker sends X back (island 1→B "X immigrates"). If B
   processes either of those before C's message, B creates X from the copy; C's message then *replaced* the
   entry with `active = 1`, discarding the earlier events. B ends with X active while A and C do not (or vice
   versa). The same overwrite reactivated individuals already replaced under pollination.
2. **Immigrant added with the sender's `active` counter** (`Migrator._receive_immigrants`). A worker whose
   counter for X was transiently 2 (return seen before the deactivation notice) emigrated X; the target island
   stored the copy with `active = 2`, so after X left again it still counted as active there → extra individual.
3. **Responsibility was a mutable attribute of copies** (`migrator_island_rank`). `SelectMin` returns references,
   so `_send_emigrants` changed the attribute on the sender's own entries, while other workers kept the old value
   and workers that first saw X as an immigrant kept the sender-assigned one. After a round trip two workers of
   one island could both consider themselves responsible → X emigrated twice, deactivated twice (`active = −1`),
   later a return left it at 0 → missing individual. After a restart, immigrants were loaded with their birth
   rank, which can exceed the island size → nobody responsible.
4. **Restart bred a finished generation again.** `propulate()` advanced past `generations[rank]` only if that
   individual was present in the population. If the worker's last individual had migrated away or been replaced,
   it is not loaded on its birth island, the generation was bred again, `_pre_eval_checkpoint` overwrote the
   row, and two different individuals shared one identity (the other island's copy and the new native). This was
   the residual flake after the fixes above (island 0 ended with 121 instead of 120 active individuals, all
   workers and the checkpoint agreeing on 121).
4b. **Resuming an unfinished evaluation deadlocked multi-rank workers** (found by review). The worker's root
   rank evaluated the placeholder loaded from the checkpoint on its own, while the worker's other ranks went
   straight into the breed/evaluate loop and waited in `worker_sub_comm.bcast`; a collective loss function
   (e.g. `parallel_sphere`'s `allreduce`) then mismatched collectives. Pre-existing on the branch.
5. **Final synchronization was probe-based and test-only.** `final_synch` did barrier + `iprobe`; MPI does not
   guarantee a non-blocking probe sees a message sent before a barrier, and `propulate()` itself never drained,
   so the library's own end state was not synchronized.
6. **Pollinator checkpoint race.** Every target worker wrote `active_on_island[g, T] = 1` when it processed an
   arrival, possibly after the responsible worker had already written 0 for the same row (immediate
   replacement), leaving the file claiming an inactive individual is active.
7. Smaller issues: `ind.island is not self.island_idx` (identity comparison on numpy ints — effectively "all but
   own-bred objects"); blocking `send` for pollen and deactivation notices (deadlock risk once messages exceed
   the eager limit, e.g. with surrogate data); `generation = −1` after a restart for a worker that had never
   started would index the last row; unevaluated rows were loaded on non-birth islands as permanent inactive
   placeholders; `Individual.__init__` ignored its `island_rank`/`prop_rank` arguments.
8. **Tests.** `deepdiff.DeepDiff` treats an `Individual` as an iterable of its parameter names
   (`Individual.__iter__`), so it never sees position, loss, velocity or identity attributes: list comparisons
   were vacuous and dict comparisons could only notice changed dict keys. Two "all losses written" checks
   applied `np.isnan` to the result of `.any()` and could never fail. `test_islands_checkpointing_incomplete` faked a crash by NaN-ing an individual that may
   already have been pollinated elsewhere, an impossible state that produced the observed `assert 3 == 2`.
   The real-migration variant of the island tests was disabled.

Why these were flaky: all of them need a particular interleaving (a message about X arriving before the message
that bred X, a counter being transiently 2, the last individual of a worker having left before the run ends, a
replacement of a just-arrived immigrant), and a failing rank leaves the other ranks blocked in collectives, which
shows up as a hang.

## 4. Code changes

`propulate/propulator.py`
- Documented identity/location model; `_responsible_rank`, `_upsert_individual`, `_process_bred_individual`
  (idempotent activation; the Migrator overrides it with `+1`).
- Unified non-blocking sends with accounting (`_send_message`, `_send_cleanup`, `_sent`/`_received` per tag),
  `_drain_messages` (count-based), `_final_synchronization` called at the end of `propulate()`, and the
  `_migration_step` hook so the generation loop exists once (the Migrator/Pollinator `propulate` overrides are gone).
- `load_checkpoint`: restart generation derived from `generations[rank]` and the row's loss; responsibility
  recomputed; unevaluated rows ignored on non-birth islands; never-started workers start at 0.
- `propulate()`: only evaluates a loaded placeholder, no longer increments based on population contents; the
  worker's root rank broadcasts the resumed individual (or `None`) to the worker's other ranks so that all ranks
  evaluate it together before the ordinary loop.

`propulate/migrator.py`, `propulate/pollinator.py`
- Bookkeeping rewritten as `_process_immigrants` / `_process_emigrants` / `_process_replaced_individuals`
  event handlers; pending `emigrated`/`replaced` lists and `==`-based lookups removed; copies are never mutated
  in place; all sends non-blocking; Pollinator eligible emigrants = all active individuals; only the responsible
  worker writes activity bits.

`propulate/population.py` (constructor arguments honored, `active` documented), `propulate/islands.py`
(`island_idx` as plain `int`).

`propulate/utils/consistency_checks.py`
- `final_synch` delegates to `_final_synchronization`; `population_consistency_check` compares the active
  individuals of all workers of an island by identity *and* content (loss, position, velocity), checks that
  every entry has `active ∈ {0, 1}` and every active one is evaluated, that the active set equals the
  checkpoint's island column, and — for real migration — that every bred identity is active on exactly one
  island (duplicates and losses that balance out in a total count are caught); `assert_populations_match`
  (identity, position, loss, velocity) replaces DeepDiff.

Nine source/test files changed (eight modified, `tests/test_bookkeeping.py` new), plus this report.

## 5. Why the fix is correct under asynchronous execution

- A worker's view of an individual is a function of the *set* of events it has processed, not their order:
  Migrator counters are sums of commuting increments; Pollinator creation is idempotent and replacement is
  absorbing. Each message is processed exactly once (no deferral lists, no matching by content).
- Decisions that must be unanimous (who emigrates/replaces X) depend only on X's identity and the island, never
  on local state or on a copy's attributes. Because the responsible worker is the only one that emigrates or
  replaces X on its island, its own counter is exact up to arrivals it has not seen yet — it can only
  under-count, so it never emigrates an individual that is not on the island.
- An island's own activity bits are written only by that island's responsible worker. The one cross-rank case is
  real migration, where the sender sets the target island's bit and the target island's responsible worker may
  clear it later; the two writes are causally ordered through the migration message (see §8 for what that
  relies on).
- Completion of the final drain is decided by counts exchanged after every worker left its loop; MPI guarantees
  delivery of sent messages and the blocking probe/receive observes them.
- The restart generation is read from the file, independent of which individuals happen to be active at home.

## 6. Tests added or modified

- `tests/test_bookkeeping.py` (new, 8 tests; run on one rank and under `mpirun`): replays bred / emigrated /
  returned / emigrated and bred / returned / replaced in every permutation (also starting from a checkpoint
  placeholder, and with copies carrying stale `active`/`migrator_island_rank`), checks responsibility is
  independent of copies, that the responsible worker replaces once per new immigrant and never reactivates a
  replaced individual, placeholder completion, and the restart re-breed (`assert 9 == 10` on the original code).
- `tests/test_island.py`: real-migration cases re-enabled; DeepDiff assertions replaced by
  `assert_populations_match`; the "incomplete" checkpoint manipulation made consistent (an unfinished individual
  exists only at home; counts on other islands compensated), expectations made mode-aware.
- `tests/test_propulator.py`, `tests/test_pso.py`: DeepDiff assertions replaced (the PSO one now really checks
  restored velocities); the two `np.isnan(...any())` checks in `test_island.py` fixed.
- `tests/test_multi_rank_workers.py`: new `test_multi_rank_workers_resume_unfinished` (2 ranks per worker,
  every worker's last evaluation unfinished in the checkpoint; deadlocked before the fix).
- `population_consistency_check` strengthened as described above.

## 7. Repeated-run results

8 ranks (`mpirun -n 8 --oversubscribe --bind-to none`), whole `tests/test_island.py`, 60 s timeout per run.

| Code | Result |
|---|---|
| Original branch, migration cases enabled | **0 / 10** (6 hangs after a rank's assertion, 4 assertion failures: `test_islands_checkpointing[True]` after restart, `unequal_populations_bla[True]`, `incomplete[True/False]`) |
| Bookkeeping/sync fixes only (before the restart fix) | 26 / 30 (one `121 != 120` after restart, three hangs of the same kind under CPU load) |
| After all fixes (before review) | 30 / 30; full suite 75 passed on 8 ranks and 48 passed sequentially; `incomplete` tests alone 40 / 40 |
| Final (after review fixes) | **30 / 30** island module; **10 / 10** multi-rank module (incl. the new unfinished-resume test); full suite on 8 ranks exit 0 (76 tests), sequential 48 passed |
| Stress: 12 ranks, islands of 5/3/4 workers, 2 migrants per message, per-island migration probability 1, 60 + 60 generations with restart, both modes | **10 / 10** |

## 8. Remaining assumptions and fragile areas

- **Checkpoint write ordering across ranks (real migration).** When X moves A→B, A's responsible worker sets
  `X.B = 1`; if X later moves B→C, B's responsible worker clears the same cell. The two writes are causally
  ordered only through the migration message. HDF5's MPI-IO driver issues raw-data writes without caching, so
  in practice the later write reaches the file system later and wins on a POSIX-coherent or lock-coherent
  (Lustre/GPFS) file system — but the MPI-IO standard's default consistency semantics do not *guarantee* this
  ordering for independent writes from different processes, and nothing in the code forces it (`H5Fflush` /
  `MPI_File_sync` are collective and cannot be issued by one rank). A fully standard-compliant design would make
  island `c`'s responsible worker the only writer of column `c` (set on arrival, cleared on departure), at the
  price of a larger crash window in which an in-flight migrant is lost, or add an explicit commit protocol.
  This was flagged as a blocker by the independent review and is deliberately left as a documented limitation.
- **Crash windows in multi-cell transitions.** Migration (target on, then source off) and pollination (immigrant
  on, then replaced off) are not atomic. An interruption between the two writes leaves an extra active copy
  after restart (never a loss); the next `population_consistency_check` would report it.
- Pollination of an individual already known on the target island (including one replaced there) is
  intentionally a no-op; such an individual can only become active there again after a restart, because
  inactive rows are not loaded.
- `population_consistency_check`'s size formulas (`generations × workers` in total for migration, per island for
  pollination) assume a clean start or a count-preserving checkpoint manipulation, as the tests now do.
- `test_islands_checkpointing_incomplete` still hard-codes per-island counts for pollination (`[20, 53]`,
  `[18, 49]`); they follow from the fixed `started_first_generations` but are brittle to edits of that array.
- A rank that fails inside an MPI test leaves the other ranks blocked (pytest-mpi); repeated runs need `timeout`.
- Messages are deep-copied per destination; mpi4py already pickles at `isend` time, so this is only a
  performance cost.
- Requires an MPI-enabled h5py. CI installs it via `libhdf5-mpi-dev`; locally HDF5 1.14.6 was built with
  `--enable-parallel --enable-hl` into `~/.local/hdf5-mpi` and h5py 3.15.1 reinstalled against it.
- Not covered: restart of CMA-ES propagator state; the regression tests in `test_bookkeeping.py` exercise the
  event handlers and the restart logic on one rank, not the message-count drain under MPI (that is covered by
  the repeated island runs and the stress scenario only).

## 9. Independent review

The report and the full change set were reviewed with Codex CLI (`gpt-5.6-sol`, reasoning effort high,
read-only sandbox). Its findings and what was done with them:

| Finding | Disposition |
|---|---|
| Blocker: migration/pollination checkpoint transitions are not crash-atomic, and the "single writer per cell" claim is false for the migrator's target column; MPI-IO does not formally order cross-rank writes. | Confirmed. Write order changed so an interruption can only leave an extra copy, never lose an individual. The cross-rank ordering is documented as an assumption with the standard-compliant alternative (§8); a transactional protocol is out of scope for this change set. |
| Major: resuming an unfinished evaluation deadlocks multi-rank workers (root evaluates alone, sub-ranks wait in `bcast`). | Confirmed and fixed (`propulate()` broadcasts the resumed individual); regression test added. |
| Major: `population_consistency_check` accepted `active == -1` entries, only checked a total count for real migration, and ignored content differences between workers. | Confirmed and fixed: `active ∈ {0,1}` for every entry, exact partition of all bred identities over islands for real migration, per-identity content comparison (loss, position, velocity) across workers. |
| Minor: `test_pso_checkpointing` still used DeepDiff; two `np.isnan(....any())` checks could never fail; velocity not compared; `test_bookkeeping.py` used `COMM_SELF` so foreign responsibility was `generation % 1`. | All fixed; `_responsible_rank` now uses `island_sizes[island_idx]` so the single-rank tests exercise the real modulo. |
| Report inaccuracies (formula sign, single-writer overstatement, DeepDiff nuance, file count, missing multi-rank issue). | Corrected in this version. |
