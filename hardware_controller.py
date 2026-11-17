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
# Boost Converter PWM Signal -> BCM GPIO 18 (Physical Pin 12) = Arduino Pin 10 (analogWrite 77)
BOOST_PWM_PIN = 18

# Solenoid Control Pins -> BCM GPIO (Physical Pins 11, 13, 15, 16, 18) = Arduino Pins 2..6
BUTTON_PINS = {
    "LEFT": 17,        # Physical Pin 11 (Arduino sol1 = Pin 2)
    "RIGHT": 27,       # Physical Pin 13 (Arduino sol2 = Pin 3)
    "ROTATE_CW": 22,   # Physical Pin 15 (Arduino sol3 = Pin 4 -> SNES 'A' button)
    "ROTATE_CCW": 23,  # Physical Pin 16 (Arduino sol4 = Pin 5 -> SNES 'B' or 'Y' button)
    "DROP": 24,        # Physical Pin 18 (Arduino sol5 = Pin 6 -> DISABLED for safety!)
}

# Set to False so 'DROP' (Down D-Pad) is NEVER actuated by the solenoids
ENABLE_DROP_SOLENOID = False

# --- EXACT ARDUINO TIMER1 BOOST CONVERTER EQUIVALENT ---
# Arduino: 16MHz / 512 = 31,250 Hz -> Period = 32,000 ns
# Arduino: analogWrite(10, 77) -> 77 / 512 = 15.04% duty cycle -> 4,812 ns
PWM_PERIOD_NS = 32000
PWM_DUTY_NS = 4812     # 15.0% duty cycle (NEVER exceed 25% on open-loop boost converter!)

# --- TIMING CONFIGURATION (Seconds) ---
# Mode 1: LED Placeholder Timings
LED_MODE_TIMINGS = {
    "press_s": 0.18,
    "release_s": 0.10,
    "drop_hold_s": 0.30,
}

# Mode 2: Real Solenoid Timings
# 55ms strike time + 90ms capacitor recharge delay between consecutive strikes
SOLENOID_MODE_TIMINGS = {
    "press_s": 0.055,      # 55ms crisp mechanical strike
    "release_s": 0.090,    # 90ms spring return + C2 boost capacitor recharge time!
    "drop_hold_s": 0.18,
}


class HardwareBoostPWM:
    """Drives BCM GPIO 18 (Physical Pin 12) using Linux hardware PWM at 31.25 kHz (15% duty)."""
    def __init__(self, period_ns=PWM_PERIOD_NS, duty_ns=PWM_DUTY_NS):
        self.pwm_dir = None
        self.period_ns = period_ns
        self.duty_ns = duty_ns
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
        print("⚡ Boost Converter PWM ONLINE on BCM GPIO 18 (31.25 kHz, 15.0% duty = Arduino 77)!")

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
    def __init__(self, use_solenoid_timings=False):
        self.use_solenoids = use_solenoid_timings
        self.timings = SOLENOID_MODE_TIMINGS if use_solenoid_timings else LED_MODE_TIMINGS
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

        # 2. If running with real solenoids, start the 31.25 kHz Boost Converter PWM on GPIO 18!
        if self.use_solenoids and GPIO_BACKEND != "MOCK":
            try:
                self.boost_pwm = HardwareBoostPWM()
                print("🔋 Pre-charging boost capacitor C2 for 0.5s...")
                time.sleep(0.5)
            except Exception as e:
                print(f"⚠️ Could not start HardwareBoostPWM on GPIO 18: {e}")

        # 3. Start non-blocking background worker thread so camera FPS never drops
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
        while automatically stripping out 'DROP' so pieces fall naturally!
        """
        if not actions:
            return

        # Filter out 'DROP' unless explicitly re-enabled
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

                hold_time = self.timings["drop_hold_s"] if action == "DROP" else self.timings["press_s"]

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
# Run `python3 hardware_controller.py --solenoids` to test your 4 active solenoids + Boost PWM!
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solenoids", action="store_true", help="Enable 31.25kHz Boost PWM on GPIO 18 + solenoid timings")
    ap.add_argument("--arduino-loop", action="store_true", help="Run Smarika's exact 1s pair-test loop")
    args = ap.parse_args()

    ctrl = HardwareController(use_solenoid_timings=args.solenoids)
    try:
        if args.arduino_loop:
            print("\n🔄 Running Smarika's Arduino Solenoid Test Loop (Ctrl+C to stop)...")
            print("   Note: Firing 1-by-1 with 60ms strikes so solenoids don't overheat!")
            while True:
                for move in ["LEFT", "RIGHT", "ROTATE_CW", "ROTATE_CCW"]:
                    print(f"  -> Striking {move} (BCM GPIO {BUTTON_PINS[move]})")
                    ctrl.dispatch_sequence([move])
                    time.sleep(0.8)
        else:
            print("\n🔧 Testing the 4 Active Solenoids in Order (DROP is disabled):")
            active_moves = ["LEFT", "RIGHT", "ROTATE_CW", "ROTATE_CCW"]
            for move in active_moves:
                print(f"  -> Striking {move} (BCM GPIO {BUTTON_PINS[move]})")
                ctrl.dispatch_sequence([move])
                time.sleep(0.6)

            print("\n🎮 Simulating AI Sequence: ['ROTATE_CW', 'LEFT', 'LEFT', 'DROP'] (DROP will be ignored!)")
            ctrl.dispatch_sequence(["ROTATE_CW", "LEFT", "LEFT", "DROP"])
            time.sleep(1.5)
            print("✅ Hardware Test Complete!")
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        ctrl.cleanup()
