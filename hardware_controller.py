import argparse
import os
import queue
import subprocess
import threading
import time

# Try gpiozero first (standard on Raspberry Pi OS Bookworm), then RPi.GPIO, or Mock mode on laptop
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
# Boost Converter PWM Signal -> BCM GPIO 18 (Physical Pin 12) = Arduino Pin 10
BOOST_PWM_PIN = 18

# Solenoid Control Pins -> BCM GPIO (Physical Pins 11, 13, 15, 16, 18) = Arduino Pins 2..6
BUTTON_PINS = {
    "LEFT": 17,        # Physical Pin 11 (Arduino sol1 = Pin 2)
    "RIGHT": 27,       # Physical Pin 13 (Arduino sol2 = Pin 3)
    "ROTATE_CW": 22,   # Physical Pin 15 (Arduino sol3 = Pin 4 -> SNES 'A' button)
    "ROTATE_CCW": 23,  # Physical Pin 16 (Arduino sol4 = Pin 5 -> SNES 'B' or 'Y' button)
    "DROP": 24,        # Physical Pin 18 (Arduino sol5 = Pin 6 -> DISABLED for safety!)
}

# Keep False so 'DROP' (Down D-Pad) is never actuated by the solenoids
ENABLE_DROP_SOLENOID = False

# --- BOOST CONVERTER HARDWARE PWM CONFIGURATION ---
# 31,250 Hz -> Period = 32,000 nanoseconds
# Note: Standard 8-bit Arduino analogWrite(10, 77) is 77 / 255 = 30.2% duty cycle (9,663 ns)!
# Previously we only ran at 15.0% (4,812 ns), which gave half the boost voltage!
PWM_PERIOD_NS = 32000
DEFAULT_BOOST_DUTY_PCT = 28.0  # 28% duty cycle (~8,960 ns) for hard D-pad strikes!
MAX_SAFE_DUTY_PCT = 35.0       # Safety ceiling to protect MOSFET Q1 & capacitor C2

# --- TIMING CONFIGURATION (Seconds) ---
# Mode 1: LED Placeholder Timings
LED_MODE_TIMINGS = {
    "dpad_press_s": 0.20,
    "rot_press_s": 0.20,
    "release_s": 0.10,
    "drop_hold_s": 0.35,
}

# Mode 2: High-Force Solenoid Timings
# D-Pad (LEFT/RIGHT) gets a longer 85ms full-saturation strike + 130ms capacitor recharge
SOLENOID_MODE_TIMINGS = {
    "dpad_press_s": 0.085,  # 85ms hard strike for stiff SNES D-Pad (LEFT / RIGHT)
    "rot_press_s": 0.060,   # 60ms crisp strike for A / B face buttons (ROTATE_CW / CCW)
    "release_s": 0.130,     # 130ms pause so C2 recharges to 100% voltage before the next strike!
    "drop_hold_s": 0.180,
}


class HardwareBoostPWM:
    """Drives BCM GPIO 18 (Physical Pin 12) using Linux hardware PWM at 31.25 kHz."""
    def __init__(self, period_ns=PWM_PERIOD_NS, duty_pct=DEFAULT_BOOST_DUTY_PCT):
        self.pwm_dir = None
        self.period_ns = int(period_ns)
        safe_pct = max(5.0, min(float(duty_pct), MAX_SAFE_DUTY_PCT))
        self.duty_pct = safe_pct
        self.duty_ns = int(round(self.period_ns * (safe_pct / 100.0)))
        self._setup_hardware_pwm()

    def _setup_hardware_pwm(self):
        if not os.path.exists("/sys/class/pwm/pwmchip0"):
            print("🔧 Loading hardware PWM overlay on BCM GPIO 18 (Physical Pin 12)...")
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
        self._write_sysfs(f"{self.pwm_dir}/period", str(self.period_ns))
        self._write_sysfs(f"{self.pwm_dir}/duty_cycle", str(self.duty_ns))
        self._write_sysfs(f"{self.pwm_dir}/enable", "1")
        print(
            f"⚡ HIGH-POWER Boost PWM ONLINE on GPIO 18 "
            f"(31.25 kHz, {self.duty_pct:.1f}% duty = {self.duty_ns} ns)!"
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
                print("🛑 Boost Converter Hardware PWM safely turned OFF.")
            except Exception as e:
                print(f"⚠️ Error stopping PWM: {e}")


class HardwareController:
    def __init__(self, use_solenoid_timings=False, boost_duty_pct=DEFAULT_BOOST_DUTY_PCT):
        self.use_solenoids = use_solenoid_timings
        self.timings = dict(SOLENOID_MODE_TIMINGS if use_solenoid_timings else LED_MODE_TIMINGS)
        self.cmd_queue = queue.Queue()
        self.running = True
        self.devices = {}
        self.boost_pwm = None

        # 1. Initialize Solenoid / LED Digital Output Pins (All pulled LOW initially)
        if GPIO_BACKEND == "GPIOZERO":
            for name, pin in BUTTON_PINS.items():
                self.devices[name] = OutputDevice(pin, active_high=True, initial_value=False)
            print(f"💡 HardwareController online via gpiozero (BCM Pins: {BUTTON_PINS})")
        elif GPIO_BACKEND == "RPIGPIO":
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for name, pin in BUTTON_PINS.items():
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            print(f"💡 HardwareController online via RPi.GPIO (BCM Pins: {BUTTON_PINS})")
        else:
            print(f"💻 GPIO library not found — running in MOCK mode (BCM Pins: {BUTTON_PINS})")

        # 2. Start 31.25 kHz Boost Converter PWM on GPIO 18 when using solenoids
        if self.use_solenoids and GPIO_BACKEND != "MOCK":
            try:
                self.boost_pwm = HardwareBoostPWM(duty_pct=boost_duty_pct)
                print("🔋 Pre-charging boost capacitor C2 for 0.7s to reach peak voltage...")
                time.sleep(0.7)
            except Exception as e:
                print(f"⚠️ Could not start HardwareBoostPWM on GPIO 18: {e}")

        # 3. Start non-blocking background worker thread
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _set_pin(self, action, state):
        if action not in BUTTON_PINS:
            return
        if action == "DROP" and not ENABLE_DROP_SOLENOID:
            return
        if GPIO_BACKEND == "GPIOZERO":
            if state:
                self.devices[action].on()
            else:
                self.devices[action].off()
        elif GPIO_BACKEND == "RPIGPIO":
            GPIO.output(BUTTON_PINS[action], GPIO.HIGH if state else GPIO.LOW)

    def dispatch_sequence(self, actions):
        """
        Queues a list of AI actions in exact order (e.g. ['ROTATE_CW', 'LEFT', 'LEFT'])
        while automatically stripping out 'DROP' so pieces fall naturally.
        """
        if not actions:
            return

        filtered_actions = [
            a for a in actions
            if (a in BUTTON_PINS and (a != "DROP" or ENABLE_DROP_SOLENOID))
        ]
        if not filtered_actions:
            return

        # Clear any stale unfinished commands if a new piece spawned
        while not self.cmd_queue.empty():
            try:
                self.cmd_queue.get_nowait()
                self.cmd_queue.task_done()
            except queue.Empty:
                break

        self.cmd_queue.put(filtered_actions)

    def _worker_loop(self):
        while self.running:
            try:
                sequence = self.cmd_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            for action in sequence:
                if not self.running:
                    break
                if action == "DROP" and not ENABLE_DROP_SOLENOID:
                    continue

                if action in ("LEFT", "RIGHT"):
                    hold_time = self.timings["dpad_press_s"]
                elif action == "DROP":
                    hold_time = self.timings["drop_hold_s"]
                else:
                    hold_time = self.timings["rot_press_s"]

                self._set_pin(action, True)
                time.sleep(hold_time)
                self._set_pin(action, False)
                time.sleep(self.timings["release_s"])

            self.cmd_queue.task_done()

    def cleanup(self):
        self.running = False
        for action in BUTTON_PINS:
            self._set_pin(action, False)
        if self.boost_pwm is not None:
            self.boost_pwm.stop()
        if GPIO_BACKEND == "GPIOZERO":
            for dev in self.devices.values():
                dev.close()
        elif GPIO_BACKEND == "RPIGPIO":
            GPIO.cleanup()


# --- STANDALONE SOLENOID + BOOST CONVERTER TESTER ---
# Run: `python3 hardware_controller.py --solenoids`
# Or tune force: `python3 hardware_controller.py --solenoids --boost-duty 30`
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solenoids", action="store_true", help="Enable 31.25kHz Boost PWM + high-force solenoid timings")
    ap.add_argument("--boost-duty", type=float, default=DEFAULT_BOOST_DUTY_PCT,
                    help=f"Boost PWM duty cycle %% (default: {DEFAULT_BOOST_DUTY_PCT}%%, max: {MAX_SAFE_DUTY_PCT}%%)")
    ap.add_argument("--dpad-ms", type=float, default=None, help="Override D-pad strike duration in ms (e.g. 95)")
    args = ap.parse_args()

    ctrl = HardwareController(use_solenoid_timings=args.solenoids, boost_duty_pct=args.boost_duty)
    if args.dpad_ms is not None:
        ctrl.timings["dpad_press_s"] = args.dpad_ms / 1000.0
        print(f"🔨 Overrode D-Pad strike time to {args.dpad_ms:.0f} ms")

    try:
        print("\n🔧 Testing the 4 Active Solenoids in Order (DROP is disabled):")
        active_moves = ["LEFT", "RIGHT", "ROTATE_CW", "ROTATE_CCW"]
        for move in active_moves:
            print(f"  -> Striking {move} (BCM GPIO {BUTTON_PINS[move]})")
            ctrl.dispatch_sequence([move])
            time.sleep(0.7)

        print("\n🎮 Simulating Multi-Tap D-Pad Sequence: ['ROTATE_CW', 'LEFT', 'LEFT', 'LEFT']")
        ctrl.dispatch_sequence(["ROTATE_CW", "LEFT", "LEFT", "LEFT"])
        time.sleep(2.0)
        print("✅ Hardware Test Complete!")
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        ctrl.cleanup()
