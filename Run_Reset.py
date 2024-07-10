import os
import subprocess
import time
import logging
import psutil


#Set up training script name
train_file = 'BobRoss_ProGan_Train.py'

# Set up logging
logging.basicConfig(filename='training_monitor.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def reset_gpu():
    try:
        # Get the current script's PID
        current_pid = os.getpid()
        
        # Get a list of all running processes
        for proc in psutil.process_iter(['pid', 'name']):
            # Ensuring that process wont kill itself ( happened once at night - was very unhappy >:( )
            if proc.info['name'] == 'python.exe' and proc.info['pid'] != current_pid:
                logging.info(f"Killing process {proc.info['pid']} to free GPU memory")
                proc.kill()
        
        # Add a small delay to ensure processes are terminated
        time.sleep(5)
    except Exception as e:
        logging.error(f"Error resetting GPU: {e}")

def check_memory():
    memory = psutil.virtual_memory()
    logging.info(f"Available memory: {memory.available / (1024 * 1024)} MB")
    return memory.available > (1024 * 1024 * 500)  # Check if more than 500 MB is available

def run_training_script():
    while True:
        if check_memory():
            process = subprocess.Popen(['python', train_file])
            process.wait()

            if process.returncode != 0:
                logging.error("Training script crashed. Resetting GPU and restarting...")
                reset_gpu()
            else:
                logging.info("Training script completed successfully.")
                break
        else:
            logging.warning("Insufficient memory. Waiting before retrying...")
            time.sleep(60)  # Wait for a minute before checking again

if __name__ == "__main__":
    run_training_script()
