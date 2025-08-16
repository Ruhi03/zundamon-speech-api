import os
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))

project_dir = os.path.join(current_dir, "zundamon_sovits")
os.chdir(project_dir)

api_server_file = os.path.join(project_dir, "api_server.py")
subprocess.run(f"python3 {api_server_file}", shell=True)
