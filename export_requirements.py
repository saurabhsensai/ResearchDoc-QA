import subprocess

packages = subprocess.check_output(["pip", "freeze"]).decode().splitlines()

with open("requirements.txt", "w") as f:
    for pkg in packages:
        if "==" in pkg:
            f.write(pkg.split("==")[0] + "\n")