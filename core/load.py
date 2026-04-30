from pathlib import Path

DEFAULT_DATA_DIR = Path("./data")


def test_video(state):
    """Sets the data directory and returns a specific test video id."""
    data_dir = state.get("data_dir", DEFAULT_DATA_DIR)

    return {
        "data_dir": data_dir,
        "video_ids": ["14QbqkeiSDtU62syzgaOVXhXRzBJWhaNN",]
    }


def all_videos(state):
    """Sets the data directory and finds every video folder."""
    data_dir = state.get("data_dir", DEFAULT_DATA_DIR)
    video_ids = [d.name for d in data_dir.iterdir() if d.is_dir()]

    return {
        "data_dir": data_dir,
        "video_ids": sorted(video_ids)
    }


def selected_videos(state, ids=None):
    """Sets the data directory and returns a manually curated list of video ids."""
    data_dir = state.get("data_dir", DEFAULT_DATA_DIR)
    return {
        "data_dir": data_dir,
        "video_ids": ids or []
    }
