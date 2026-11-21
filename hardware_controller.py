import argparse
import os
import queue
import subprocess
import threading
import time

GPIO_BACKEND = "MOCK"
try:
    from gpiozero import OutputDevice
    GPIO_BACKEND = "GPIOZERO"
except ImportError:
    try:
        import RPi.GPIO as GPIO
        GPIO_BACKEND = "RPIGPIO"
    except ImportError:
        GPIO_BACKEND = "MOCK"


# --- PIN MAPPING (BCM Numbering) ---
BOOST_PWM_PIN = 18     # Physical Pin 12 (Arduino Pin 10 = signalpin)

BUTTON_PINS = {
    "LEFT": 17,        # Physical Pin 11 (Arduino sol1 = Pin 2)
    "RIGHT": 27,       # Physical Pin 13 (Arduino sol2 = Pin 3)
    "ROTATE_CW": 22,   # Physical Pin 15 (Arduino sol3 = Pin 4 -> SNES 'A' button)
    #"ROTATE_CCW": 23,  # Physical Pin 16 (DISABLED)
    "DROP": 23,        # Physical Pin 18 (Arduino sol5 = Pin 6 -> SNES D-pad DOWN)
}

ENABLE_DROP_SOLENOID = True
ENABLE_CCW_SOLENOID = False
ENABLE_BOOST_PWM = False  # OFF while running directly on the 12V power supply!

# Hard coil-safety & overshoot limits for the DROP solenoid (seconds)
MIN_DROP_HOLD_S = 0.080
MAX_DROP_HOLD_S = 0.450

# --- BOOST CONVERTER PWM PRESETS ---
ARDUINO_490HZ_PERIOD_NS = 2040816
ARDUINO_490HZ_DUTY_PCT = 45.0

FAST_31KHZ_PERIOD_NS = 32000
FAST_31KHZ_DUTY_PCT = 15.0

MAX_SAFE_DUTY_PCT = 45.0
DEFAULT_BOOST_DUTY_PCT = ARDUINO_490HZ_DUTY_PCT

# --- TIMING CONFIGURATION (Seconds) ---
LED_MODE_TIMINGS = {
    "dpad_press_s": 0.20,
    "rot_press_s": 0.20,
    "release_s": 0.10,
    "consecutive_same_extra_s": 0.05,
    "drop_hold_s": 0.25,
}

# Tuned for direct 12V supply: firm strikes + fast enough to finish multi-press moves before landing
SOLENOID_MODE_TIMINGS = {
    "dpad_press_s": 0.080,             # 80ms firm strike for D-pad
    "rot_press_s": 0.075,              # 75ms firm strike for A button
    "release_s": 0.120,                # 120ms reset between different buttons
    "consecutive_same_extra_s": 0.040, # +40ms (160ms total) for repeated same-button hits
    "drop_hold_s": 0.180,              # fallback if DROP is dispatched without an explicit hold duration
}


def _parse_action(item):
    """Normalizes either 'LEFT' or ('DROP', 0.28) into (action_name, custom_hold_or_None)."""
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return str(item[0]), float(item[1])
    return str(item), None


class HardwareBoostPWM:
    """Drives BCM GPIO 18 (Physical Pin 12) using Linux kernel hardware PWM."""
    def __init__(self, freq_hz=490, duty_pct=DEFAULT_BOOST_DUTY_PCT, soft_start=True):
        self.pwm_dir = None
        self.freq_hz = float(freq_hz)
        self.period_ns = int(round(1e9 / self.freq_hz))
        safe_pct = max(5.0, min(float(duty_pct), MAX_SAFE_DUTY_PCT))
        self.duty_pct = safe_pct
        self.duty_ns = int(round(self.period_ns * (safe_pct / 100.0)))
        self._setup_hardware_pwm(soft_start=soft_start)

    def _setup_hardware_pwm(self, soft_start=True):
        if not os.path.exists("/sys/class/pwm/pwmchip0"):
            print("Loading hardware PWM overlay on BCM GPIO 18 (Physical Pin 12)...")
            subprocess.run(["sudo", "dtoverlay", "pwm", "pin=18", "func=2"], check=False)
            time.sleep(0.3)

        chips = sorted(os.listdir("/sys/class/pwm")) if os.path.exists("/sys/class/pwm") else []
        if not chips:
            raise RuntimeError(
                "No /sys/class/pwm/pwmchip* found! Run: 'sudo dtoverlay pwm pin=18 func=2' in terminal."
            )

        chip_path = f"/sys/class/pwm/{chips[0]}"
        self.pwm_dir = f"{chip_path}/pwm0"

        if not os.path.exists(self.pwm_dir):
            self._write_sysfs(f"{chip_path}/export", "0")
            time.sleep(0.1)

        self._write_sysfs(f"{self.pwm_dir}/enable", "0")
        try:
            self._write_sysfs(f"{self.pwm_dir}/duty_cycle", "0")
        except Exception:
            pass
        self._write_sysfs(f"{self.pwm_dir}/period", str(self.period_ns))
        self._write_sysfs(f"{self.pwm_dir}/enable", "1")

        if soft_start and self.duty_pct > 25.0:
            ramp_steps = [p for p in (20.0, 30.0, 38.0) if p < self.duty_pct] + [self.duty_pct]
            for pct in ramp_steps:
                step_ns = int(round(self.period_ns * (pct / 100.0)))
                self._write_sysfs(f"{self.pwm_dir}/duty_cycle", str(step_ns))
                time.sleep(0.15)
        else:
            self._write_sysfs(f"{self.pwm_dir}/duty_cycle", str(self.duty_ns))

        print(
            f"Boost PWM ONLINE on GPIO 18 "
            f"({self.freq_hz:.0f} Hz, {self.duty_pct:.1f}% duty | period={self.period_ns}ns, duty={self.duty_ns}ns)!"
        )

    def _write_sysfs(self, path, value):
        try:
            with open(path, "w") as f:
                f.write(value)
        except PermissionError:
            subprocess.run(["sudo", "sh", "-c", f"echo {value} > {path}"], check=True)

    def stop(self):
        if self.pwm_dir and os.path.exists(self.pwm_dir):
            try:
                self._write_sysfs(f"{self.pwm_dir}/duty_cycle", "0")
                self._write_sysfs(f"{self.pwm_dir}/enable", "0")
                print("Boost Converter Hardware PWM safely turned OFF.")
            except Exception as e:
                print(f"Error stopping PWM: {e}")


class HardwareController:
    def __init__(self, use_solenoid_timings=False, boost_duty_pct=None, boost_freq_hz=490):
        self.use_solenoids = use_solenoid_timings
        self.timings = dict(SOLENOID_MODE_TIMINGS if use_solenoid_timings else LED_MODE_TIMINGS)
        self.cmd_queue = queue.Queue()
        self.running = True
        self.devices = {}
        self.boost_pwm = None
        self.last_action = None
        self._busy = False
        self._lock = threading.Lock()
        self._gen = 0
        self._cancel_ev = threading.Event()

        if boost_duty_pct is None:
            boost_duty_pct = DEFAULT_BOOST_DUTY_PCT if boost_freq_hz <= 2000 else FAST_31KHZ_DUTY_PCT

        if GPIO_BACKEND == "GPIOZERO":
            for name, pin in BUTTON_PINS.items():
                self.devices[name] = OutputDevice(pin, active_high=True, initial_value=False)
            print(f"HardwareController online via gpiozero (BCM Pins: {BUTTON_PINS})")
        elif GPIO_BACKEND == "RPIGPIO":
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for name, pin in BUTTON_PINS.items():
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            print(f"HardwareController online via RPi.GPIO (BCM Pins: {BUTTON_PINS})")
        else:
            print(f"GPIO library not found — running in MOCK mode (BCM Pins: {BUTTON_PINS})")

        if self.use_solenoids and ENABLE_BOOST_PWM and GPIO_BACKEND != "MOCK":
            try:
                self.boost_pwm = HardwareBoostPWM(freq_hz=boost_freq_hz, duty_pct=boost_duty_pct)
                print("Pre-charging boost capacitor C2 for 1.0s to reach full voltage...")
                time.sleep(1.0)
            except Exception as e:
                print(f"Could not start HardwareBoostPWM on GPIO 18: {e}")
        elif self.use_solenoids:
            print("Direct 12V mode: Boost PWM on GPIO 18 is OFF.")

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def strike_s(self):
        return float(self.timings["dpad_press_s"])

    def press_period_s(self):
        return float(
            self.timings["dpad_press_s"]
            + self.timings["release_s"]
            + self.timings.get("consecutive_same_extra_s", 0.040)
        )

    def _hold_for(self, action, custom_hold=None):
        if action == "DROP":
            h = self.timings["drop_hold_s"] if custom_hold is None else float(custom_hold)
            return max(MIN_DROP_HOLD_S, min(MAX_DROP_HOLD_S, h))
        if custom_hold is not None:
            return max(0.020, min(0.450, float(custom_hold)))
        if action in ("LEFT", "RIGHT"):
            return float(self.timings["dpad_press_s"])
        return float(self.timings["rot_press_s"])

    def estimate_sequence_s(self, actions):
        """Time from first strike start until last strike finishes (end of last press)."""
        if not actions:
            return 0.0
        parsed = [_parse_action(a) for a in actions]
        total = 0.0
        for idx, (action, custom_hold) in enumerate(parsed):
            total += self._hold_for(action, custom_hold)
            if idx + 1 < len(parsed):
                wait = self.timings["release_s"]
                if parsed[idx + 1][0] == action:
                    wait += self.timings.get("consecutive_same_extra_s", 0.040)
                total += wait
        return total

    def is_busy(self):
        return self._busy or not self.cmd_queue.empty()

    def _set_pin(self, action, state):
        if action not in BUTTON_PINS:
            return
        if action == "DROP" and not ENABLE_DROP_SOLENOID:
            return
        if action == "ROTATE_CCW" and not ENABLE_CCW_SOLENOID:
            return
        if GPIO_BACKEND == "GPIOZERO":
            if state:
                self.devices[action].on()
            else:
                self.devices[action].off()
        elif GPIO_BACKEND == "RPIGPIO":
            GPIO.output(BUTTON_PINS[action], GPIO.HIGH if state else GPIO.LOW)

    def _all_off(self):
        for action in BUTTON_PINS:
            self._set_pin(action, False)

    def _drain_queue(self):
        while not self.cmd_queue.empty():
            try:
                self.cmd_queue.get_nowait()
                self.cmd_queue.task_done()
            except queue.Empty:
                break

    def cancel(self):
        """Immediately aborts any in-progress or queued solenoid sequence (including an active DROP hold)
        so stale presses can never bleed into the next spawned piece."""
        with self._lock:
            self._gen += 1
            self._cancel_ev.set()
            self._drain_queue()
        self._all_off()

    def dispatch_sequence(self, actions):
        if not actions:
            return

        filtered = []
        for item in actions:
            action, custom_hold = _parse_action(item)
            if action not in BUTTON_PINS:
                continue
            if action == "ROTATE_CCW" and not ENABLE_CCW_SOLENOID:
                continue
            if action == "DROP":
                if not ENABLE_DROP_SOLENOID:
                    continue
                if custom_hold is not None and custom_hold < MIN_DROP_HOLD_S:
                    continue
                clamped = self._hold_for("DROP", custom_hold)
                filtered.append(("DROP", clamped))
            else:
                filtered.append((action, custom_hold))

        if not filtered:
            return

        with self._lock:
            self._gen += 1
            gen = self._gen
            self._cancel_ev.set()
            self._drain_queue()
            self.cmd_queue.put((gen, filtered))

    def _worker_loop(self):
        while self.running:
            try:
                gen, sequence = self.cmd_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            with self._lock:
                if gen != self._gen:
                    self.cmd_queue.task_done()
                    continue
                self._cancel_ev.clear()

            self._busy = True
            try:
                for idx, (action, custom_hold) in enumerate(sequence):
                    if not self.running or self._cancel_ev.is_set() or gen != self._gen:
                        break
                    if action == "DROP" and not ENABLE_DROP_SOLENOID:
                        continue
                    if action == "ROTATE_CCW" and not ENABLE_CCW_SOLENOID:
                        continue

                    hold_time = self._hold_for(action, custom_hold)

                    self._set_pin(action, True)
                    interrupted = self._cancel_ev.wait(hold_time)
                    self._set_pin(action, False)
                    self.last_action = action
                    if interrupted or not self.running or gen != self._gen:
                        break

                    next_action = sequence[idx + 1][0] if idx + 1 < len(sequence) else None
                    wait_time = self.timings["release_s"]
                    if next_action == action:
                        wait_time += self.timings.get("consecutive_same_extra_s", 0.040)
                    if self._cancel_ev.wait(wait_time):
                        break
            finally:
                self._all_off()
                self._busy = False
                self.cmd_queue.task_done()

    def cleanup(self):
        self.running = False
        self.cancel()
        if self.boost_pwm is not None:
            self.boost_pwm.stop()
        if GPIO_BACKEND == "GPIOZERO":
            for dev in self.devices.values():
                dev.close()
        elif GPIO_BACKEND == "RPIGPIO":
            GPIO.cleanup()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solenoids", action="store_true", help="Enable solenoid timings")
    ap.add_argument("--freq", type=float, default=490.0)
    ap.add_argument("--boost-duty", type=float, default=DEFAULT_BOOST_DUTY_PCT)
    ap.add_argument("--strike-ms", type=float, default=None)
    ap.add_argument("--recharge-ms", type=float, default=None)
    args = ap.parse_args()

    ctrl = HardwareController(
        use_solenoid_timings=args.solenoids,
        boost_duty_pct=args.boost_duty,
        boost_freq_hz=args.freq
    )
    if args.strike_ms is not None:
        ctrl.timings["dpad_press_s"] = args.strike_ms / 1000.0
        ctrl.timings["rot_press_s"] = args.strike_ms / 1000.0
    if args.recharge_ms is not None:
        ctrl.timings["release_s"] = args.recharge_ms / 1000.0

    try:
        print("\nTesting the 4 Active Solenoids on 12V (LEFT, RIGHT, ROTATE_CW, DROP):")
        for move in ["LEFT", "RIGHT", "ROTATE_CW", ("DROP", 0.15)]:
            name = move[0] if isinstance(move, tuple) else move
            print(f"  -> Striking {move} (BCM GPIO {BUTTON_PINS[name]})")
            ctrl.dispatch_sequence([move])
            time.sleep(0.6)
        print("Hardware Test Complete!")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ctrl.cleanup()
