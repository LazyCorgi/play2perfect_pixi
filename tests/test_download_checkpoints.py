import hashlib
from types import SimpleNamespace

import huggingface_hub
import pytest

import download_checkpoints as module


def test_tls_failure_reports_network_cause_and_never_publishes_model(tmp_path, monkeypatch):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        raise OSError('[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol')
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    destination = tmp_path/'pretrained_policy'
    with pytest.raises(RuntimeError, match='SSL: UNEXPECTED_EOF') as failure:
        module._download(module.DownloadArgs(), 'play', destination)
    assert 'Environment installation and CUDA checks need not be repeated' in str(failure.value)
    assert len(calls) == 3
    assert not (destination/'model.pth').exists()


def test_verified_download_uses_selected_endpoint_and_handles_existing_directory(tmp_path, monkeypatch):
    cached = tmp_path/'cached.pth'; cached.write_bytes(b'valid official weights')
    monkeypatch.setitem(module.OFFICIAL_SHA256, 'play', hashlib.sha256(cached.read_bytes()).hexdigest())
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(cached)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    dest = tmp_path/'pretrained_policy'; dest.mkdir()
    args = module.DownloadArgs(endpoint='https://models.example.test')
    module._download(args, 'play', dest)
    assert (dest/'model.pth').read_bytes() == cached.read_bytes()
    assert calls[0]['endpoint'] == args.endpoint and calls[0]['filename'] == 'play/model.pth'
    module._download(args, 'play', dest)
    assert len(calls) == 1
    assert not list(dest.glob('*.tmp'))


def test_corrupt_cached_and_offline_copied_weights_are_not_accepted(tmp_path, monkeypatch):
    cached = tmp_path/'bad.pth'; cached.write_bytes(b'partial download')
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda **kw: str(cached))
    dest = tmp_path/'pretrained_policy'
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        module._download(module.DownloadArgs(), 'play', dest)
    assert not (dest/'model.pth').exists()
    dest.mkdir(); (dest/'model.pth').write_bytes(b'also incorrect')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        module._download(module.DownloadArgs(), 'play', dest)


def test_missing_remote_file_is_not_misreported_as_transient_network_failure(tmp_path, monkeypatch):
    calls=[]
    class Missing(Exception):
        response=SimpleNamespace(status_code=404)
    def download(**kwargs):
        calls.append(kwargs)
        raise Missing('missing model')
    monkeypatch.setattr(huggingface_hub,'hf_hub_download', download)
    with pytest.raises(RuntimeError,match='HTTP 404'):
        module._download(module.DownloadArgs(),'play',tmp_path/'out')
    assert len(calls)==1
