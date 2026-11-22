"""
Python wrapper for TexasSolver console version.
Generates config files, runs the solver, and parses results.
"""
import subprocess
import json
import os
import tempfile


class SolverConfig:
    """Configuration builder for TexasSolver."""

    def __init__(self):
        self.pot = 0
        self.effective_stack = 0
        self.board = ""
        self.range_oop = ""
        self.range_ip = ""
        self.bet_sizes = []
        self.allin_threshold = 0.67
        self.thread_num = os.cpu_count() or 8
        self.accuracy = 0.5
        self.max_iteration = 200
        self.print_interval = 10
        self.use_isomorphism = 1
        self.dump_rounds = 1

    def set_spot(self, pot, eff_stack, board, range_oop, range_ip):
        self.pot = pot
        self.effective_stack = eff_stack
        self.board = board
        self.range_oop = range_oop
        self.range_ip = range_ip
        return self

    def add_bet_size(self, pos, street, action, size=None):
        if size is not None:
            self.bet_sizes.append(f"set_bet_sizes {pos},{street},{action},{size}")
        else:
            self.bet_sizes.append(f"set_bet_sizes {pos},{street},{action}")
        return self

    def set_default_bet_sizes(self):
        for pos in ["oop", "ip"]:
            for street in ["flop", "turn", "river"]:
                self.add_bet_size(pos, street, "bet", 50)
                self.add_bet_size(pos, street, "bet", 75)
                self.add_bet_size(pos, street, "bet", 100)
                self.add_bet_size(pos, street, "raise", 50)
                self.add_bet_size(pos, street, "raise", 100)
                self.add_bet_size(pos, street, "allin")
                if pos == "oop" and street in ["turn", "river"]:
                    self.add_bet_size(pos, street, "donk", 50)
                    self.add_bet_size(pos, street, "donk", 75)
        return self

    def generate_config(self, output_path):
        lines = []
        lines.append(f"set_pot {self.pot}")
        lines.append(f"set_effective_stack {self.effective_stack}")
        lines.append(f"set_board {self.board}")
        lines.append(f"set_range_oop {self.range_oop}")
        lines.append(f"set_range_ip {self.range_ip}")
        if not self.bet_sizes:
            self.set_default_bet_sizes()
        for bs in self.bet_sizes:
            lines.append(bs)
        lines.append(f"set_allin_threshold {self.allin_threshold}")
        lines.append("build_tree")
        lines.append(f"set_thread_num {self.thread_num}")
        lines.append(f"set_accuracy {self.accuracy}")
        lines.append(f"set_max_iteration {self.max_iteration}")
        lines.append(f"set_print_interval {self.print_interval}")
        lines.append(f"set_use_isomorphism {self.use_isomorphism}")
        lines.append("start_solve")
        lines.append(f"set_dump_rounds {self.dump_rounds}")
        lines.append(f"dump_result {output_path}")
        return "\n".join(lines)


class TexasSolverWrapper:
    """Wrapper for TexasSolver console binary."""

    def __init__(self, solver_path=None, resource_dir=None):
        # the repo root: this file is archive/src/solver/wrapper.py, and tools/ stays at the root
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        if solver_path is None:
            solver_path = os.path.join(
                base, "tools", "TexasSolver", "build", "console_solver")
        if resource_dir is None:
            resource_dir = os.path.join(
                base, "tools", "TexasSolver", "resources")
        self.solver_path = solver_path
        self.resource_dir = resource_dir
        if not os.path.exists(self.solver_path):
            raise FileNotFoundError(
                f"Solver not found: {self.solver_path}")
        if not os.path.isdir(self.resource_dir):
            raise FileNotFoundError(
                f"Resources not found: {self.resource_dir}")

    def solve(self, config, output_file=None):
        """Run the solver with given configuration."""
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False,
                prefix="solver_cfg_") as cfg_f:
            cfg_path = cfg_f.name
            if output_file is None:
                output_file = tempfile.mktemp(
                    suffix=".json", prefix="solver_out_")
            cfg_f.write(config.generate_config(output_file))

        try:
            result = subprocess.run(
                [self.solver_path,
                 "--input_file", cfg_path,
                 "--resource_dir", self.resource_dir],
                capture_output=True, text=True, timeout=1800
            )
            log = result.stdout + result.stderr
            iters, expl, time_s = self._parse_output(log)
            strategy = {}
            if os.path.exists(output_file):
                with open(output_file) as f:
                    strategy = json.load(f)
            return {
                "success": result.returncode == 0,
                "iterations": iters,
                "exploitability": expl,
                "time_seconds": time_s,
                "strategy": strategy,
                "output_file": output_file,
                "solver_log": log
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Solver timed out after 30 minutes",
                "solver_log": ""
            }
        finally:
            os.unlink(cfg_path)

    def _parse_output(self, output):
        """Parse solver console output for final stats."""
        iters, expl, time_s = 0, 100.0, 0.0
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("Iter:"):
                try:
                    iters = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "Total exploitability" in line:
                try:
                    expl = float(line.split()[2])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("time used:"):
                try:
                    time_s = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
        return iters, expl, time_s

    def solve_spot(self, pot, eff_stack, board, range_oop, range_ip,
                   accuracy=0.5, max_iterations=200, threads=None):
        """Convenience method to solve a spot with default bet sizes."""
        config = SolverConfig()
        config.set_spot(pot, eff_stack, board, range_oop, range_ip)
        config.set_default_bet_sizes()
        config.accuracy = accuracy
        config.max_iteration = max_iterations
        if threads:
            config.thread_num = threads
        return self.solve(config)


def test_solver():
    """Quick test to verify solver works."""
    try:
        solver = TexasSolverWrapper()
        print(f"Solver binary: {solver.solver_path}")
        print(f"Resources: {solver.resource_dir}")
        print("Solver wrapper initialized OK!")
        print(f"CPU threads available: {os.cpu_count()}")
        return True
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    test_solver()