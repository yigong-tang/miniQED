import os
import tempfile
from mini_qed.logging import PipelineLogger, TokenTracker


class TestPipelineLogger:
    def test_creates_status_file(self):
        with tempfile.TemporaryDirectory() as d:
            logger = PipelineLogger(d, "Test Phase")
            logger.update_status(1, 9, "Proof Search", "RUNNING", "Running agent...")
            assert os.path.exists(os.path.join(d, "AUTO_RUN_STATUS.md"))
            assert os.path.exists(os.path.join(d, "AUTO_RUN_LOG.txt"))

    def test_log_appends(self):
        with tempfile.TemporaryDirectory() as d:
            logger = PipelineLogger(d, "Test")
            logger.log("line one")
            logger.log("line two")
            log_path = os.path.join(d, "AUTO_RUN_LOG.txt")
            with open(log_path) as f:
                content = f.read()
            assert "line one" in content
            assert "line two" in content


class TestTokenTracker:
    def test_record_and_save(self):
        with tempfile.TemporaryDirectory() as d:
            tracker = TokenTracker(d, "deepseek-v4-pro")
            tracker.record("Proof Search R1", 1000, 500, 5.2,
                          provider="deepseek", model="deepseek-v4-pro")
            tracker.record("Verification R1", 800, 300, 3.1,
                          provider="deepseek", model="deepseek-v4-flash")

            assert tracker.total_input == 1800
            assert tracker.total_output == 800
            assert len(tracker.calls) == 2

            # Verify md output
            assert os.path.exists(tracker.md_path)
            with open(tracker.md_path) as f:
                md = f.read()
            assert "Proof Search R1" in md
            assert "deepseek-v4-pro" in md

            # Verify json output
            assert os.path.exists(tracker.json_path)
