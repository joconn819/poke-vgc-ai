# Fresh-plan audit: opponent information and Round583

Audit of the current checkout on 2026-11-13. This is a source inspection and artifact review, not a completed differential information-boundary verification.

## Opponents

- **Legacy Counterplay, evaluation:** `scripts/evaluate_native.py` passes the opposing player's mirrored battle object to `choose_counterplay_action`, along with a predicted Max Damage action. `src/baselines/native_damage.py` reads the opposing battle's team and active Pokémon when scoring responses. This opponent is privileged; its scores are stress tests, not legal closed-sheet opponent results. The evaluated policy's own access must be audited separately.
- **Legacy Counterplay, training:** `scripts/run_headless_showdown_self_play.py` passes both `match.battles[other_index]` and `actions[agent_names[other_index]]` to `choose_counterplay_action`. This opponent can use private opposing state and the actual already selected simultaneous action. Its behavior is not a legal simultaneous closed-sheet policy.
- **Max Damage:** both inspected call sites give `choose_max_damage_action` the acting player's own battle object and action mask. The scorer accesses own active Pokémon and moves and `battle.opponent_active_pokemon`, and its damage calculation reads target attributes such as stats and abilities. No opposing battle object or actual opposing action is passed directly. Whether those attributes contain exclusively publicly available information in every runtime path has not yet been verified; do not certify it without a hidden-state perturbation and protocol-view audit.
- **Public Counterplay:** the inspected call sites pass the acting player's battle object and public history, not the opposing battle object or already chosen action. The scorer constructs threats from public history and set priors. This is a promising public-view design, but the provenance of all fields and priors and the history boundaries still require differential checks.

## Round583

The retry completed a 20,000-game rollout and produced `checkpoints/latest.pt` on 2026-11-01. Its stored preflight audit has `ready_for_ppo: true`, no audit errors, 40,000 safe player-view rows, 177,989 trainable transitions and 10,003/9,997 trainable games in P1/P2. This audit addresses recorded data checks; it does not certify the opponent's information boundary or the overall training objective. Its evaluation field is null and the evaluation directory is empty: no held-out performance claim is supported.

The run included 14,114 legacy Counterplay games (70.57%), 2,927 Public Counterplay, 825 Max Damage and 2,134 historical-policy games. Consequently most training games used the privileged Counterplay opponent described above. Collection took 544.12 seconds (36.76 games/s); the recorded elapsed time to checkpoint was 955.83 seconds (20.92 games/s). These are historical measurements, not current machine benchmarks.

## Round582 interpretation

The matrix includes the privileged legacy Counterplay opponent. Its restricted-game value is a result against a pool that includes a privileged response; it does not establish a legal closed-sheet equilibrium or a score against only legal strategies. One timeout or error was recorded in Round531 versus Round536, and 127 team clusters were common to all cells. The Max Damage and Public Counterplay columns remain provisional until their information boundaries are independently checked.

## Next checks

1. Generate two otherwise identical visible states with different unrevealed sets and actions. Verify observations, Max Damage and Public Counterplay action probabilities/outputs are invariant to those hidden changes for both seats, including preview and forced-switch requests.
2. Trace public-history construction and player-specific protocol messages from local server through headless parsing. Compare equivalent player views against the server.
3. Freeze legal-opponent implementations and team/seed manifests before reproducing the old baseline. Track excluded games, errors, and per-seat counts, and retain an untouched team split.

## Initial regression coverage

`HeadlessShowdownProcess._apply_state` feeds public history from the shared protocol log and sends each parser its corresponding player-specific log and request. This is a promising separation in source, but it has not yet been verified end to end by perturbing hidden teams.

On 2026-11-13, 30 existing targeted tests passed: four native damage tests, nine public-history tests, one recipient-specific split-log projection test, and sixteen closed-sheet belief tests. A stale fake order in the native-damage speed-cache test lacked fields now required by `DoubleBattleOrder.join_orders`; the fixture was corrected before rerunning. These tests do not prove hidden-state invariance of live baseline or policy decisions.

## Hidden-set perturbation: initial headless battle state

A new integration regression (`tests/env/test_headless_hidden_invariance.py`) starts equivalent local headless battles and changes an opponent's unrevealed moves and item, once for a lead and once for a selected bench Pokémon. It checks both seats. All four cases produced identical player observations, legal masks, Max Damage choices and Public Counterplay choices after team preview. The fixture uses modified sets as controlled perturbations and does not claim team legality. It does not yet cover the trained policy's complete inputs, preview itself, revealed-information updates, forced switches, ability changes, or live-server parity. The information-boundary gate remains open.

The same four initial-state perturbations were extended to instantiate the Round430 policy with both complete team dictionaries. Its deterministic selected action, log probability and value also remained identical, as did the baseline choices and observations. This exercises the species-only opponent preview and default initial policy inputs, but does not cover later public-context/history features or live-server parity. Command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -m pytest tests/env/test_headless_hidden_invariance.py -q` (4 passed).

The opening-state regression now also varies two non-announcing abilities on each side's active and bench Pokémon. All eight combinations of seat, slot and hidden change passed. These are controlled simulator perturbations, not proof that the fixture sets are legal teams or that later states and training inputs respect the information boundary.

## Hidden-set perturbation after a completed turn

The regression now also advances equivalent battles by one full turn while changing the moves/items or a non-announcing ability on an unrevealed benched Pokémon. It checks both acting seats and compares the player observation, legal mask, Max Damage action, Public Counterplay action, encoded 73-feature public-mechanics context, and Round430 action/log probability/value. All 12 cases pass: eight opening checks (both seats, active and bench, moves/item and ability) and four later-turn bench checks (both seats and both hidden-field variants). The transition choices are selected using the public Counterplay baseline and replayed identically in both worlds. If the parser marks a side as waiting, the test checks the no-choice sentinel instead of calling a decision heuristic. This extends coverage to one later state; it does not yet cover multiple revealed-history updates, forced replacements, live-server parity, legal roster perturbations, or every training feature path. The boundary gate remains open. Command: `PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 venv/bin/python -m pytest tests/env/test_headless_hidden_invariance.py -q` (12 passed).
