"""Verify the shared/private smoke matrix and summarize cache backing bytes."""

import argparse
import json
from pathlib import Path


def compare(directory):
    cases = []
    for weights in ("fp8", "mxfp4"):
        for dp in (1, 2):
            pool_ids = set()
            private_bytes = shared_bytes = 0
            for rank in range(dp):
                suffix = (
                    f"{weights}.json" if dp == 1 else f"dp2-{weights}.json.dp{rank}"
                )
                private_path = directory / f"private-{suffix}"
                shared_path = directory / f"shared-{suffix}"
                assert json.loads(private_path.read_text()) == json.loads(
                    shared_path.read_text()
                ), suffix
                private = json.loads(
                    Path(str(private_path) + ".cache.json").read_text()
                )
                shared = json.loads(Path(str(shared_path) + ".cache.json").read_text())
                assert len(private) == len(shared) == 2
                assert not any(item["shared"] for item in private)
                assert all(item["shared"] for item in shared)
                assert len({item["bytes"] for item in private + shared}) == 1
                assert len({item["num_blocks"] for item in private + shared}) == 1
                assert shared[0]["inode"] == shared[1]["inode"]
                assert shared[0]["cache_sha256"] == shared[1]["cache_sha256"]
                assert sum(item["writer"] for item in shared) == 1
                if dp == 2:
                    assert all(item["ep_size"] == 4 for item in private + shared)
                pool_ids.add(shared[0]["inode"])
                private_bytes += sum(item["bytes"] for item in private)
                shared_bytes += shared[0]["bytes"]
            assert len(pool_ids) == dp
            assert private_bytes == shared_bytes * 2
            cases.append(
                dict(
                    weights=weights,
                    dp=dp,
                    tp=2,
                    expert_parallel=dp == 2,
                    tokens_and_logprobs_exact_match=True,
                    independent_shared_pools=len(pool_ids),
                    private_cache_backing_bytes=private_bytes,
                    shared_cache_backing_bytes=shared_bytes,
                )
            )
    return dict(model="reduced seven-layer GLM5.2, dummy weights", cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.directory), indent=2))
