"""Tests for the Speech-to-Text category in `hermes tools` (tools_config).

Covers the STT provider picker rows, config writes (stt.provider /
use_gateway), the model picker catalog, config-only checklist exclusion,
and the faster_whisper post-setup readiness hook.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hermes_cli.tools_config import (
    STT_MODEL_CATALOG,
    TOOL_CATEGORIES,
    _checklist_toolset_keys,
    _configure_stt_model,
    _is_provider_active,
    _write_provider_config,
    apply_provider_selection,
)

def _stt_cat():
    return TOOL_CATEGORIES["stt"]

def _stt_provider_named(name):
    return next(p for p in _stt_cat()["providers"] if p["name"] == name)

class TestConfigWrites:
    def test_write_provider_config_sets_stt_provider(self):
        config = {"stt": {"use_gateway": True}}
        prov = _stt_provider_named("Groq")
        _write_provider_config(prov, config, managed_feature=None)
        assert config["stt"]["provider"] == "groq"
        # Legacy key is popped so the read-time shim can't override the pick.
        assert "use_gateway" not in config["stt"]

    def test_apply_provider_selection_stt(self):
        config = {}
        with patch(
            "hermes_cli.tools_config.get_nous_subscription_features"
        ) as feats:
            feats.return_value = MagicMock(
                nous_auth_present=False, account_info=None
            )
            apply_provider_selection("stt", "OpenAI", config)
        assert config["stt"]["provider"] == "openai"

class TestActiveDetection:
    def test_active_matches_config(self):
        config = {"stt": {"provider": "groq"}}
        assert _is_provider_active(_stt_provider_named("Groq"), config)
        assert not _is_provider_active(_stt_provider_named("OpenAI"), config)

    def test_unset_provider_defaults_to_local(self):
        assert _is_provider_active(_stt_provider_named("Local Whisper"), {})

class TestModelPicker:

    def test_every_model_picker_lists_the_runtime_catalog(self):
        """Dashboard selects and Desktop suggestions offer exactly STT_MODEL_CATALOG per provider."""
        import re

        from hermes_cli.web_server_config import _SCHEMA_OVERRIDES
        from tools.transcription_common import BUILTIN_STT_PROVIDERS, STT_MODEL_CONFIG_KEY

        desktop = (Path(__file__).resolve().parents[2] / "apps/desktop/src/app/settings/constants.ts").read_text()
        for provider, models in STT_MODEL_CATALOG.items():
            key = f"stt.{provider}.{STT_MODEL_CONFIG_KEY.get(provider, 'model')}"
            listed = re.search(rf"'{re.escape(key)}': \[([^\]]*)\]", desktop)
            assert listed, key
            desktop_models = re.findall(r"'([^']+)'", listed.group(1))
            # Local sizes are listed smallest-first in the UIs; the catalog puts the default first.
            same = set if provider == "local" else list
            assert same(_SCHEMA_OVERRIDES[key]["options"]) == same(models), key
            assert same(desktop_models) == same(models), key
        assert set(_SCHEMA_OVERRIDES["stt.provider"]["options"]) == BUILTIN_STT_PROVIDERS - {"local_command"}

    def test_english_only_sizes_sit_beside_their_multilingual_sibling(self):
        """Every `.en` size keeps its multilingual sibling, immediately before it.

        Both pickers read the catalog in order, so an `.en` entry that drifts away from its
        sibling is a UI regression; a `.en` with no sibling has no size to specialise.
        """
        local = STT_MODEL_CATALOG["local"]
        assert len(local) == len(set(local))
        english_only = [model for model in local if model.endswith(".en")]
        assert english_only, "the local catalog must offer the English-only sizes"
        for model in english_only:
            assert local.index(model) == local.index(model.removesuffix(".en")) + 1

    def test_local_model_normalization_keeps_english_only_sizes(self):
        """A catalog `.en` value reaches faster-whisper unchanged instead of falling back."""
        from tools.transcription_local import _normalize_local_model

        assert _normalize_local_model("base.en") == "base.en"
        assert _normalize_local_model("whisper-1") == "base"  # cloud-only names still fall back

    def test_configure_stt_model_defaults_to_current(self):
        config = {"stt": {"openai": {"model": "gpt-transcribe"}}}
        with patch(
            "hermes_cli.tools_config._prompt_choice", return_value=0
        ) as pc:
            _configure_stt_model("openai", config)
        # default index should point at the currently configured model
        args = pc.call_args[0]
        assert args[2] == STT_MODEL_CATALOG["openai"].index("gpt-transcribe")

class TestConfigOnlyExclusion:

    def test_stt_excluded_from_checklist_universe(self):
        assert "stt" not in _checklist_toolset_keys("cli")
        # sanity: tts (a real toolset) stays in
        assert "tts" in _checklist_toolset_keys("cli")
