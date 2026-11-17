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
BOOST_PWM_PIN = 18     # Physical Pin 12 (Arduino Pin 10 = signalpin)

BUTTON_PINS = {
    "LEFT": 17,        # Physical Pin 11 (Arduino sol1 = Pin 2)
    "RIGHT": 27,       # Physical Pin 13 (Arduino sol2 = Pin 3)
    "ROTATE_CW": 22,   # Physical Pin 15 (Arduino sol3 = Pin 4 -> SNES 'A' button)
    "ROTATE_CCW": 23,  # Physical Pin 16 (Arduino sol4 = Pin 5 -> SNES 'B' or 'Y' button)
    "DROP": 24,        # Physical Pin 18 (Arduino sol5 = Pin 6 -> DISABLED)
}

ENABLE_DROP_SOLENOID = False

# --- BOOST CONVERTER PWM PRESETS ---
# Mode A (Default): Exact match to Smarika's Arduino `analogWrite(10, 77)`!
# Standard Arduino Pin 10 PWM is 490 Hz (Period = 2,040,816 ns) at 77/255 = 30.2% duty (616,246 ns).
ARDUINO_490HZ_PERIOD_NS = 2040816
ARDUINO_490HZ_DUTY_PCT = 30.2

# Mode B: Fast 31.25 kHz Timer1 mode (Period = 32,000 ns) at 15.0% duty (4,812 ns)
FAST_31KHZ_PERIOD_NS = 32000
FAST_31KHZ_DUTY_PCT = 15.0

MAX_SAFE_DUTY_PCT = 45.0

# --- TIMING CONFIGURATION (Seconds) ---
LED_MODE_TIMINGS = {
    "dpad_press_s": 0.20,
    "rot_press_s": 0.20,
    "release_s": 0.10,
    "drop_hold_s": 0.35,
}

# Tuned so each strike delivers peak impulse without draining C2 flat,
# followed by a 220ms recharge window so C2 reaches peak voltage before the next strike!
SOLENOID_MODE_TIMINGS = {
    "dpad_press_s": 0.065,  # 65ms strike (plunger bottoms out without draining C2 completely)
    "rot_press_s": 0.055,   # 55ms strike for A / B buttons
    "release_s": 0.220,     # 220ms capacitor C2 recharge window between consecutive presses!
    "drop_hold_s": 0.180,
}


class HardwareBoostPWM:
    """Drives BCM GPIO 18 (Physical Pin 12) using Linux kernel hardware PWM."""
    def __init__(self, freq_hz=490, duty_pct=ARDUINO_490HZ_DUTY_PCT):
        self.pwm_dir = None
        self.freq_hz = float(freq_hz)
        self.period_ns = int(round(1e9 / self.freq_hz))
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

        # Always reset duty_cycle to 0 before changing period so kernel sysfs never rejects the write!
        self._write_sysfs(f"{self.pwm_dir}/enable", "0")
        try:
            self._write_sysfs(f"{self.pwm_dir}/duty_cycle", "0")
        except Exception:
            pass
        self._write_sysfs(f"{self.pwm_dir}/period", str(self.period_ns))
        self._write_sysfs(f"{self.pwm_dir}/duty_cycle", str(self.duty_ns))
        self._write_sysfs(f"{self.pwm_dir}/enable", "1")
        print(
            f"⚡ Boost PWM ONLINE on GPIO 18 "
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
                print("🛑 Boost Converter Hardware PWM safely turned OFF.")
            except Exception as e:
                print(f"⚠️ Error stopping PWM: {e}")


DEFAULT_BOOST_DUTY_PCT = ARDUINO_490HZ_DUTY_PCT


class HardwareController:
    def __init__(self, use_solenoid_timings=False, boost_duty_pct=None, boost_freq_hz=490):
        self.use_solenoids = use_solenoid_timings
        self.timings = dict(SOLENOID_MODE_TIMINGS if use_solenoid_timings else LED_MODE_TIMINGS)
        self.cmd_queue = queue.Queue()
        self.running = True
        self.devices = {}
        self.boost_pwm = None

        if boost_duty_pct is None:
            boost_duty_pct = ARDUINO_490HZ_DUTY_PCT if boost_freq_hz <= 2000 else FAST_31KHZ_DUTY_PCT

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

        if self.use_solenoids and GPIO_BACKEND != "MOCK":
            try:
                self.boost_pwm = HardwareBoostPWM(freq_hz=boost_freq_hz, duty_pct=boost_duty_pct)
                print("🔋 Pre-charging boost capacitor C2 for 1.0s to reach full voltage...")
                time.sleep(1.0)
            except Exception as e:
                print(f"⚠️ Could not start HardwareBoostPWM on GPIO 18: {e}")

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
        if not actions:
            return

        filtered_actions = [
            a for a in actions
            if (a in BUTTON_PINS and (a != "DROP" or ENABLE_DROP_SOLENOID))
        ]
        if not filtered_actions:
            return

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solenoids", action="store_true", help="Enable Boost PWM + solenoid timings")
    ap.add_argument("--freq", type=float, default=490.0,
                    help="Boost PWM frequency in Hz (490 = standard Arduino analogWrite, 31250 = Timer1)")
    ap.add_argument("--boost-duty", type=float, default=None,
                    help="Boost PWM duty cycle %% (default: 30.2%% at 490Hz, 15.0%% at 31250Hz)")
    ap.add_argument("--strike-ms", type=float, default=None,
                    help="Override solenoid ON strike duration in ms (e.g. 80)")
    ap.add_argument("--recharge-ms", type=float, default=None,
                    help="Override capacitor recharge delay between strikes in ms (e.g. 300)")
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
        print("\n🔧 Testing the 4 Active Solenoids (1.0s recharge between each):")
        for move in ["LEFT", "RIGHT", "ROTATE_CW", "ROTATE_CCW"]:
            print(f"  -> Striking {move} (BCM GPIO {BUTTON_PINS[move]})")
            ctrl.dispatch_sequence([move])
            time.sleep(1.0)

        print("\n🎮 Simulating AI Sequence: ['ROTATE_CW', 'LEFT', 'LEFT']")
        ctrl.dispatch_sequence(["ROTATE_CW", "LEFT", "LEFT"])
        time.sleep(2.0)
        print("✅ Hardware Test Complete!")
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        ctrl.cleanup()
