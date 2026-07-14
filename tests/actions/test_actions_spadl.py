"""Pin our mirrored SPADL vocabulary against socceraction's real one.

:mod:`src.actions.spadl` re-declares SPADL's enums and column set so the stage
can run without importing socceraction (which drags in pandera/xgboost; every
other stage here is pandas/numpy only). A mirrored vocabulary is a promise that
can rot silently -- an enum reordered upstream would turn every ``type_id`` we
emit into a *different, valid-looking* action.

These tests are what keep that promise. They are the reason the mirror is safe,
and they skip cleanly when socceraction isn't installed (``pip install -e
".[spadl]"``).
"""

import pytest

pytest.importorskip(
    "socceraction",
    reason='socceraction not installed; pip install -e ".[spadl]"',
)

from socceraction.spadl import config as sa_config  # noqa: E402
from socceraction.spadl.schema import SPADLSchema  # noqa: E402

from src.actions import spadl  # noqa: E402


def test_action_type_vocabulary_matches_socceraction_exactly():
    """Order is load-bearing: type_id is an INDEX into this list."""
    assert spadl.ACTIONTYPES == list(sa_config.actiontypes)


def test_result_vocabulary_matches_socceraction_exactly():
    assert spadl.RESULTS == list(sa_config.results)


def test_bodypart_vocabulary_matches_socceraction_exactly():
    assert spadl.BODYPARTS == list(sa_config.bodyparts)


def test_the_ids_we_emit_resolve_to_the_actions_we_mean():
    """A round-trip through socceraction's own lookup tables."""
    types = sa_config.actiontypes_df().set_index("type_id")["type_name"].to_dict()
    results = sa_config.results_df().set_index("result_id")["result_name"].to_dict()
    parts = sa_config.bodyparts_df().set_index("bodypart_id")["bodypart_name"].to_dict()

    for name in spadl.EMITTED_ACTIONTYPES:
        assert types[spadl.type_id(name)] == name
    for name in ("success", "fail"):
        assert results[spadl.result_id(name)] == name
    assert parts[spadl.bodypart_id("foot")] == "foot"


def test_our_column_set_is_exactly_what_the_schema_declares():
    """SPADLSchema is ``strict``: a stray column is a hard failure, not a warning.

    This is also *why* the confidence/occlusion flags live in a separate
    provenance table rather than as extra columns on the actions table.
    """
    schema_columns = set(SPADLSchema.to_schema().columns)
    assert set(spadl.SPADL_COLUMNS) == schema_columns
    # ...and nothing required is missing from what we always write
    required = {
        name for name, col in SPADLSchema.to_schema().columns.items()
        if col.required
    }
    assert required <= set(spadl.SPADL_REQUIRED_COLUMNS)


def test_the_pitch_we_emit_into_is_spadls_own_pitch():
    """105x68 is SPADL's default, which is why this layer applies NO rescale.

    The prerequisites already rescale into the target frame. If socceraction ever
    changed its field size, our coordinates would silently be in the wrong pitch.
    """
    assert spadl.FIELD_LENGTH_M == sa_config.field_length
    assert spadl.FIELD_WIDTH_M == sa_config.field_width


def test_an_empty_actions_table_still_validates():
    """A clip with no transitions must not produce a malformed table."""
    SPADLSchema.validate(spadl.empty_actions())
