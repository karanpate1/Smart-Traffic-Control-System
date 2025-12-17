import subprocess
import time

# List of scripts to run in parallel
# The processor (main.py) and the detector(s) must run at the same time.
scripts_to_run = [
    # "src/script2(api_call).py",
    "src/script2(final_json_writer).py",
    "src/heavy_vehicle(with_queue)(roi_from_json).py",
    "src/red_light_violation(with_queue)(roi_from_json).py",
    "src/wrong_way(with_queue)(roi_from_json).py", # Or whichever detection script you are using
    # Add other detection scripts here if needed, e.g., "no_helmet.py"
]

# A list to hold the process objects
processes = []

print("🚀 Starting the ANPR system...")

# Launch each script as a separate, parallel process
for script in scripts_to_run:
    # subprocess.Popen is non-blocking; it starts the script and continues
    process = subprocess.Popen(['python', script])
    processes.append(process)
    print(f"   -> Launched {script}")
    time.sleep(3)
print("\n✅ System is running. Press Ctrl+C to stop all scripts.")

try:
    # This loop keeps the main 'final.py' script alive.
    # If this script exits, the child processes it started will also be terminated.
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Stopping all scripts...")
    for process in processes:
        process.terminate()  # Send a signal to stop each script
    print("All scripts have been terminated.")