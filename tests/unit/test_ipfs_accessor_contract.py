import pytest

from prsm.core import ipfs_client as core_ipfs


@pytest.mark.asyncio
async def test_create_ipfs_client_validates_configuration_inputs():
    with pytest.raises(ValueError, match="Invalid api_url"):
        await core_ipfs.create_ipfs_client(api_url="localhost:5001")

    with pytest.raises(ValueError, match="Invalid gateway_url"):
        await core_ipfs.create_ipfs_client(gateway_url="localhost:8080")

    with pytest.raises(ValueError, match="timeout must be > 0"):
        await core_ipfs.create_ipfs_client(timeout=0)


@pytest.mark.asyncio
async def test_create_ipfs_client_initializes_configured_client(monkeypatch):
    initialized = {"called": False}

    async def _fake_initialize(self):
        initialized["called"] = True
        self.connected = True

    monkeypatch.setattr(core_ipfs.IPFSClient, "initialize", _fake_initialize)

    client = await core_ipfs.create_ipfs_client(
        api_url="http://127.0.0.1:5001",
        gateway_url="http://127.0.0.1:8080",
        timeout=12.5,
    )

    assert initialized["called"] is True
    assert isinstance(client, core_ipfs.IPFSClient)
    assert client.config.api_url == "http://127.0.0.1:5001"
    assert client.config.gateway_url == "http://127.0.0.1:8080"
    assert client.config.timeout == 12.5


def test_get_ipfs_client_returns_global_singleton_instance():
    accessor_client = core_ipfs.get_ipfs_client()
    assert accessor_client is core_ipfs.ipfs_client


@pytest.mark.asyncio
async def test_legacy_helper_aliases_delegate_to_canonical_methods(monkeypatch):
    client = core_ipfs.IPFSClient()

    initialize_called = {"value": False}
    upload_args = {}
    download_args = {}

    async def _fake_initialize(self):
        initialize_called["value"] = True

    async def _fake_upload_content(self, content, filename=None, pin=True, progress_callback=None):
        upload_args.update(
            {
                "content": content,
                "filename": filename,
                "pin": pin,
            }
        )
        return core_ipfs.IPFSResult(success=True, cid="cid-1", size=4)

    async def _fake_download_content(
        self,
        cid,
        output_path=None,
        progress_callback=None,
        verify_integrity=True,
    ):
        download_args.update(
            {
                "cid": cid,
                "verify_integrity": verify_integrity,
            }
        )
        return core_ipfs.IPFSResult(success=True, cid=cid, size=4, metadata={"content": b"test"})

    monkeypatch.setattr(core_ipfs.IPFSClient, "initialize", _fake_initialize)
    monkeypatch.setattr(core_ipfs.IPFSClient, "upload_content", _fake_upload_content)
    monkeypatch.setattr(core_ipfs.IPFSClient, "download_content", _fake_download_content)

    await client.connect()
    upload_result = await client.add_content(content=b"test", filename="x.bin", pin=False)
    download_result = await client.get_content(cid="cid-1", verify_integrity=False)

    assert initialize_called["value"] is True
    assert upload_result.success is True
    assert download_result.success is True
    assert upload_args == {"content": b"test", "filename": "x.bin", "pin": False}
    assert download_args == {"cid": "cid-1", "verify_integrity": False}
