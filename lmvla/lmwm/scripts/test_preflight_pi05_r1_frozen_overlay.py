import pytest

from preflight_pi05_r1_frozen_overlay import reverse_eval_launcher_amendment


def test_reverse_eval_launcher_amendment_is_exact() -> None:
    frozen = (
        "PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r1_protocol_v1.json\n"
        "before\n"
        '  --repo "$REPO" --protocol "$PROTOCOL" \\\n'
        '  --output "$REPO/logs/r1/protocol_eval_${CONDITION}_s${SEED}.json"\n'
        "after\n"
    )
    runtime = (
        "PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r1_protocol_v1.json\n"
        "VERIFY_REPO=${R1_VERIFY_REPO:-$REPO}\n"
        "PROTOCOL_OUTPUT_DIR=${R1_PROTOCOL_OUTPUT_DIR:-$REPO/logs/r1_runtime}\n"
        "before\n"
        'if [[ "$VERIFY_REPO" != "$REPO" ]]; then\n'
        '  test -s "$VERIFY_REPO/READY"\n'
        "fi\n"
        'mkdir -p "$PROTOCOL_OUTPUT_DIR"\n'
        '  --repo "$VERIFY_REPO" --protocol "$PROTOCOL" \\\n'
        '  --output "$PROTOCOL_OUTPUT_DIR/protocol_eval_${CONDITION}_s${SEED}.json"\n'
        "after\n"
    )
    assert reverse_eval_launcher_amendment(runtime) == frozen


def test_reverse_eval_launcher_amendment_rejects_extra_shape() -> None:
    with pytest.raises(ValueError, match="unexpected R1 evaluator amendment"):
        reverse_eval_launcher_amendment("unrelated\n")
