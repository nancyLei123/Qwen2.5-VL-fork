import functools
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = load("demo_server", ROOT / "docs/room_demo/start_demo.py")
package = load("demo_package", ROOT / "scripts/package_room_reconstruction_demo.py")


class DemoTests(unittest.TestCase):
    def test_ranges(self):
        for header, expected in [("bytes=2-4", (2,4)), ("bytes=2-", (2,9)),
                                 ("bytes=-3", (7,9)), ("bytes=0-100", (0,9))]:
            self.assertEqual(server.byte_range(header, 10), expected)
        for header in ("bytes=10-", "bytes=-0", "bytes=4-2", "bytes=0-1,4-5", "bytes=-"):
            with self.assertRaises(ValueError):
                server.byte_range(header, 10)

    def test_excerpt_keeps_original_pose_and_clock(self):
        original = {"trial_id":"RoomScan3c", "fps":30, "bounds":[], "labels_not_applicable":True,
                    "scene_assembly_note":"static", "poses":[[i]*7 for i in range(100)],
                    "recording_times":[i/30+.01 for i in range(100)],
                    "pose_status":["mps_supported"]*100,
                    "scenes":[{"id":"room", "start_s":0, "end_s":4}],
                    "provenance":{"graph_uid":"room", "model_sha256":"test"}}
        original["poses"][32] = None
        original["pose_status"][32] = "unavailable"
        result = package.excerpt(original, 30, 60)
        self.assertEqual(result["poses"], original["poses"][30:90])
        self.assertEqual(result["recording_times"], original["recording_times"][30:90])
        self.assertIsNone(result["poses"][2])
        self.assertEqual(result["duration_s"], 2)
        self.assertEqual(result["scenes"][0]["video_start_s"], 0)
        self.assertEqual(result["provenance"]["valid_pose_count"], 59)
        with self.assertRaises(ValueError):
            package.excerpt(original, 70, 60)

    def test_server_video_seek_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "video.mp4").write_bytes(b"0123456789")
            handler = functools.partial(server.Handler, directory=directory)
            http = server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{http.server_port}/video.mp4"
            try:
                with urlopen(Request(url, headers={"Range":"bytes=3-5"})) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.headers["Content-Range"], "bytes 3-5/10")
                    self.assertEqual(response.read(), b"345")
                with urlopen(Request(url, method="HEAD", headers={"Range":"bytes=-2"})) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.headers["Content-Length"], "2")
                    self.assertEqual(response.read(), b"")
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url, headers={"Range":"bytes=100-"}))
                self.assertEqual(error.exception.code, 416)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url, method="POST", data=b"no writes"))
                self.assertEqual(error.exception.code, 501)
                self.assertEqual(Path(directory, "video.mp4").read_bytes(), b"0123456789")
            finally:
                http.shutdown()
                http.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
